"""ImageNet-pretrained ResNet18 adapted for binary real-vs-AI classification."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.models.heads import BinaryHead, set_backbone_trainable


def build_resnet18(freeze_backbone: bool = False, head_dropout: float = 0.0) -> nn.Module:
    """Build an ImageNet-pretrained ResNet18 with a binary classification head.

    Args:
        freeze_backbone: If True, freeze all backbone parameters while
            keeping the classification head trainable.
        head_dropout: Dropout probability applied before the classification
            layer. Defaults to 0.0.

    Returns:
        A ResNet18 module returning one logit per image with shape (batch,).
    """
    model: nn.Module = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features: int = model.fc.in_features
    model.fc = BinaryHead(in_features, dropout=head_dropout)

    if freeze_backbone:
        set_backbone_trainable(model, head_attr="fc", trainable=False)

    return model
