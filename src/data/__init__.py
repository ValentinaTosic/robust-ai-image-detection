"""Core project code for dataset preparation, model training, and evaluation.

The package contains utilities for:
- loading experiment configuration
- image standardization and preprocessing
- dataset and split management
- model definitions
- training and evaluation
"""
"""Dataset preparation and loading utilities."""

from src.data.dataset import AIImageDataset, LABEL_TO_INDEX
from src.data.loaders import build_image_transform, create_dataloaders

__all__ = [
    "AIImageDataset",
    "LABEL_TO_INDEX",
    "build_image_transform",
    "create_dataloaders",
]
