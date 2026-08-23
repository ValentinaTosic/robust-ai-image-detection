"""Model architectures used in the image-detection experiments."""

from src.models.baseline_cnn import BaselineCNN, count_trainable_parameters

__all__ = ["BaselineCNN", "count_trainable_parameters"]
