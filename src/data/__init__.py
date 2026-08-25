"""Data loading, preprocessing, and dataset management utilities.

The package provides:
- dataset definitions and label mappings
- image preprocessing and transformations
- train, validation, and test data loader creation
"""

from src.data.dataset import AIImageDataset, LABEL_TO_INDEX
from src.data.loaders import build_image_transform, build_pretrained_transform, create_dataloaders

__all__ = [
    "AIImageDataset",
    "LABEL_TO_INDEX",
    "build_image_transform",
    "build_pretrained_transform",
    "create_dataloaders",
]
