"""Classification heads shared across pretrained backbones."""

from torch import Tensor, nn


class BinaryHead(nn.Module):
    """Single-logit classification head for binary classification."""

    def __init__(self, in_features: int, dropout: float = 0.0) -> None:
        """Initialize the binary classification head.

        Args:
            in_features: Number of input features produced by the backbone.
            dropout: Dropout probability applied before the linear layer.
                Defaults to 0.0.
        """
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.linear: nn.Linear = nn.Linear(in_features, 1)

    def forward(self, features: Tensor) -> Tensor:
        """Generate one binary classification logit per example.

        Args:
            features: Feature tensor with shape (batch, in_features).

        Returns:
            Tensor containing one logit per example with shape (batch,).
        """
        return self.linear(self.dropout(features)).squeeze(dim=1)


def set_backbone_trainable(model: nn.Module, head_attr: str, trainable: bool) -> None:
    """Freeze or unfreeze backbone parameters while keeping the head unchanged.

    Args:
        model: Model containing a classification head.
        head_attr: Name of the classification head attribute, such as
            "fc" or "head".
        trainable: Whether backbone parameters should be trainable.
    """
    prefix: str = f"{head_attr}."
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix):
            parameter.requires_grad = trainable