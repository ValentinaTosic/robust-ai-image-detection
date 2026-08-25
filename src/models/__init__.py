"""Model architectures used in the image-detection experiments."""

from src.models.baseline_cnn import BaselineCNN, count_trainable_parameters
from src.models.resnet import build_resnet18, set_backbone_trainable

__all__ = [
    "BaselineCNN",
    "count_trainable_parameters",
    "build_resnet18",
    "set_backbone_trainable",
]
