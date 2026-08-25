"""PyTorch dataset utilities for the standardized image manifest."""

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


LABEL_TO_INDEX: dict[str, int] = {"real": 0, "ai": 1}
REQUIRED_COLUMNS: set[str] = {
    "filename",
    "label",
    "generator",
    "source_group",
    "split",
}


class AIImageDataset(Dataset):
    """Load one split of the standardized real-vs-AI image dataset.

    Each item contains the transformed image, a binary target where real is 0
    and AI is 1, and source metadata used for per-generator evaluation.
    """

    def __init__(
        self,
        manifest: pd.DataFrame | Path | str,
        processed_root: Path | str,
        split: str,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        """Create a dataset from the manifest and select one split.

        Args:
            manifest: Dataset manifest as a DataFrame or path to its CSV file.
            processed_root: Root containing the standardized ``ai`` and
                ``real`` image directories.
            split: One of ``train``, ``val``, or ``test``.
            transform: Image transformation that returns a tensor.

        Raises:
            ValueError: If the manifest schema, labels, or split are invalid.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split!r}; expected train, val, or test.")

        dataframe = (
            manifest.copy()
            if isinstance(manifest, pd.DataFrame)
            else pd.read_csv(manifest)
        )

        missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset manifest is missing columns: {sorted(missing_columns)}"
            )

        unknown_labels = set(dataframe["label"]).difference(LABEL_TO_INDEX)
        if unknown_labels:
            raise ValueError(f"Unknown labels in dataset manifest: {sorted(unknown_labels)}")

        split_rows = dataframe[dataframe["split"] == split].reset_index(drop=True)
        if split_rows.empty:
            raise ValueError(f"Dataset manifest contains no rows for split {split!r}.")

        self.dataframe = split_rows
        self.processed_root = Path(processed_root)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the selected split."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        """Load and transform one image together with its target and metadata."""
        row = self.dataframe.iloc[index]
        image_path = self.processed_root / row["filename"]

        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        return {
            "image": image_tensor,
            "label": torch.tensor(LABEL_TO_INDEX[row["label"]], dtype=torch.float32),
            "generator": row["generator"],
            "source_group": row["source_group"],
            "filename": row["filename"],
        }
