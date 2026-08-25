"""ImageNet-pretrained ResNet18 adapted for binary real vs AI classification."""

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class _BinaryHead(nn.Module):
    """Single-logit classification head for binary classification."""

    def __init__(self, in_features: int) -> None:
        """Initialize the binary classification head.

        Args:
            in_features: Number of input features produced by the backbone.
        """
        super().__init__()
        self.linear: nn.Linear = nn.Linear(in_features, 1)

    def forward(self, features: Tensor) -> Tensor:
        """Generate one binary classification logit per example.

        Args:
            features: Feature tensor produced by the backbone, with shape
                (batch, in_features).

        Returns:
            Tensor containing one logit per example, with shape (batch,).
        """
        return self.linear(features).squeeze(dim=1)


def build_resnet18(freeze_backbone: bool = False) -> nn.Module:
    """Build an ImageNet-pretrained ResNet18 for binary classification.

    The original fully connected layer is replaced with a single-logit
    classification head. The model therefore produces outputs compatible
    with BCEWithLogitsLoss and the shared training and evaluation
    pipeline.

    Args:
        freeze_backbone: If True, freeze all backbone parameters while
            keeping the classifier head trainable. This is used during
            the warmup phase before fine-tuning the full network.

    Returns:
        ImageNet-pretrained ResNet18 with a binary classification head.
        The model returns one logit per image with shape (batch,).
    """
    model: nn.Module = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = _BinaryHead(model.fc.in_features)

    if freeze_backbone:
        set_backbone_trainable(model, trainable=False)

    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze the ResNet18 backbone.

    The classifier head remains trainable regardless of the value of
    ``trainable``. This allows the model to be used in a two-phase
    fine-tuning setup: first training only the classifier head, followed
    by fine-tuning the pretrained backbone with a lower learning rate.

    Args:
        model: ResNet18 model created by ``build_resnet18``.
        trainable: If True, make the backbone parameters trainable.
            If False, freeze the backbone parameters.
    """
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = trainable
