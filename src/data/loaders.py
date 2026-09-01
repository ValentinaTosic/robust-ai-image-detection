"""Transforms and reproducible DataLoaders for image classification."""

from pathlib import Path
from typing import Callable
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from src.data.dataset import AIImageDataset

NORMALIZATION_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
NORMALIZATION_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)

# Statistics used by ImageNet-pretrained backbones such as ResNet and ViT.
# Pretrained weights expect inputs normalized this way; 
# reusing the from-scratch baseline's normalization would provide inputs
# from a different distribution than the one used during pretraining.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

def build_image_transform() -> v2.Compose:
    """Build the deterministic transform used by the first CNN baseline.

    Standardized images are already 224 x 224 RGB, so the baseline only
    converts them to float tensors and maps pixel values from [0, 1] to
    approximately [-1, 1]. No augmentation is applied.

    Returns:
        A deterministic image transformation pipeline using the baseline
        normalization statistics.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
        ]
    )

def build_pretrained_transform() -> v2.Compose:
    """Build the deterministic transform for ImageNet-pretrained models.

    Standardized images are already 224 x 224 RGB, so this only converts
    images to float tensors and normalizes them using ImageNet statistics.
    No augmentation is applied, keeping preprocessing consistent with the
    baseline evaluation setup.

    Returns:
        A deterministic image transformation pipeline using ImageNet
        normalization statistics.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

def build_vit_transform(model_name: str, augment: bool = False) -> v2.Compose:
    """Build the transform expected by a timm ViT checkpoint.

    Normalization statistics are read from the selected checkpoint's
    pretrained configuration rather than hardcoded, since different timm
    checkpoints may use different statistics. Standardized images are already
    224 x 224 RGB, so only tensor conversion and normalization are required.

    Args:
        model_name: timm model identifier, e.g. "vit_small_patch16_224".
        augment: If True, apply a random horizontal flip for training-time
            augmentation. Validation and test transforms use the default
            ``False`` value to remain deterministic.

    Returns:
        An image transformation pipeline using the normalization statistics
        specified by the selected pretrained checkpoint.
    """
    pretrained_cfg: dict = timm.get_pretrained_cfg(model_name).to_dict()
    data_config: dict = timm.data.resolve_data_config({}, pretrained_cfg=pretrained_cfg)
    mean: tuple[float, ...] = data_config["mean"]
    std: tuple[float, ...] = data_config["std"]

    transforms: list = [v2.RandomHorizontalFlip(p=0.5)] if augment else []
    transforms += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]
    return v2.Compose(transforms)

def create_dataloaders(
    manifest: pd.DataFrame | Path | str,
    processed_root: Path | str,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool = False,
    transform_fn: Callable[[], v2.Compose] = build_image_transform,
    train_transform_fn: Callable[[], v2.Compose] | None = None,
) -> dict[str, DataLoader]:
    """Create reproducible train, validation, and test DataLoaders.

    Training examples are shuffled using a seeded generator, while
    validation and test examples retain a fixed order so evaluation results
    and saved predictions remain stable.

    Args:
        manifest: Dataset manifest containing image paths, labels, and split
            assignments.
        processed_root: Root directory containing the processed images.
        batch_size: Number of examples loaded per batch.
        num_workers: Number of worker processes used to load data.
        seed: Random seed used for deterministic training-data shuffling.
        pin_memory: Whether DataLoaders should use pinned memory for faster
            host-to-device transfers when training on a compatible device.
        transform_fn: Callable that builds the image transformation applied
            to the validation and test splits (and to the training split too
            if ``train_transform_fn`` is not given). Use
            ``build_pretrained_transform`` for ImageNet-pretrained backbones.
            Defaults to ``build_image_transform``.
        train_transform_fn: Optional callable that builds a separate,
            typically augmented, transform for the training split only.
            Defaults to ``None``, meaning the training split reuses
            ``transform_fn`` like the other splits.

    Returns:
        Dictionary containing train, val, and test DataLoaders.
    """
    if batch_size <= 0:
        raise ValueError("Batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    transform: v2.Compose = transform_fn()
    train_transform: v2.Compose = train_transform_fn() if train_transform_fn is not None else transform
    split_transforms: dict[str, v2.Compose] = {"train": train_transform, "val": transform, "test": transform}
    datasets: dict[str, AIImageDataset] = {
        split: AIImageDataset(manifest, processed_root, split, split_transforms[split])
        for split in ("train", "val", "test")
    }

    train_generator: torch.Generator = torch.Generator().manual_seed(seed)

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