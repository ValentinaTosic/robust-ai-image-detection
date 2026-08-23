"""Transforms and reproducible DataLoaders for image classification."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.data.dataset import AIImageDataset


NORMALIZATION_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
NORMALIZATION_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)


def build_image_transform() -> v2.Compose:
    """Build the deterministic transform used by the first CNN baseline.

    Standardized images are already 224 x 224 RGB, so the baseline only
    converts them to float tensors and maps pixel values from [0, 1] to
    approximately [-1, 1]. No augmentation is applied in this experiment.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )


def create_dataloaders(
    manifest: pd.DataFrame | Path | str,
    processed_root: Path | str,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Create reproducible train, validation, and test DataLoaders.

    Training examples are shuffled with a seeded generator. Validation and
    test order remains fixed so evaluation and saved predictions are stable.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    transform = build_image_transform()
    datasets = {
        split: AIImageDataset(manifest, processed_root, split, transform)
        for split in ("train", "val", "test")
    }

    train_generator = torch.Generator().manual_seed(seed)

    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
            generator=train_generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
