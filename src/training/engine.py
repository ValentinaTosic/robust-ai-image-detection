"""Reusable training and evaluation loops for binary image classification."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and binary-classification metrics collected over one epoch."""

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> dict[str, float]:
        """Return metric values in a serialization-friendly form."""
        return asdict(self)


@dataclass(frozen=True)
class FitResult:
    """Training history and location of the best validation checkpoint."""

    history: list[dict[str, float | int]]
    best_epoch: int
    best_val_loss: float
    epochs_completed: int
    checkpoint_path: Path


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _calculate_metrics(
    average_loss: float,
    labels: torch.Tensor,
    logits: torch.Tensor,
) -> EpochMetrics:
    """Calculate binary metrics using AI as the positive class."""
    labels_np = labels.numpy().astype(np.int64)
    probabilities = torch.sigmoid(logits).numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)

    return EpochMetrics(
        loss=average_loss,
        accuracy=accuracy_score(labels_np, predictions),
        precision=precision_score(labels_np, predictions, zero_division=0),
        recall=recall_score(labels_np, predictions, zero_division=0),
        f1=f1_score(labels_np, predictions, zero_division=0),
        roc_auc=roc_auc_score(labels_np, probabilities),
    )


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
) -> EpochMetrics:
    """Run one training or evaluation epoch."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    all_labels: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_labels.append(labels.detach().cpu())
        all_logits.append(logits.detach().cpu())

    if total_examples == 0:
        raise ValueError("Cannot run an epoch with an empty DataLoader.")

    return _calculate_metrics(
        average_loss=total_loss / total_examples,
        labels=torch.cat(all_labels),
        logits=torch.cat(all_logits),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """Train the model for one complete pass over a DataLoader."""
    return _run_epoch(model, dataloader, criterion, device, optimizer)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate the model without calculating gradients or updating weights."""
    return _run_epoch(model, dataloader, criterion, device, optimizer=None)


def fit(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    min_delta: float,
    checkpoint_path: Path | str,
) -> FitResult:
    """Train with early stopping and restore the best validation checkpoint.

    Args:
        model: Model to optimize.
        train_dataloader: Training split DataLoader.
        val_dataloader: Validation split DataLoader.
        criterion: Binary classification loss.
        optimizer: Parameter optimizer.
        device: CPU or CUDA device used for computation.
        epochs: Maximum number of epochs.
        patience: Consecutive non-improving epochs allowed before stopping.
        min_delta: Minimum validation-loss decrease considered an improvement.
        checkpoint_path: Destination for the best model checkpoint.

    Returns:
        Complete history and information about the restored best checkpoint.

    Raises:
        ValueError: If the stopping configuration is invalid.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if patience <= 0:
        raise ValueError("patience must be positive.")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative.")

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device
        )
        val_metrics = evaluate(model, val_dataloader, criterion, device)

        epoch_record: dict[str, float | int] = {"epoch": epoch}
        epoch_record.update(
            {f"train_{name}": value for name, value in train_metrics.to_dict().items()}
        )
        epoch_record.update(
            {f"val_{name}": value for name, value in val_metrics.to_dict().items()}
        )
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train loss {train_metrics.loss:.4f}, F1 {train_metrics.f1:.4f} | "
            f"val loss {val_metrics.loss:.4f}, F1 {val_metrics.f1:.4f}"
        )

        if val_metrics.loss < best_val_loss - min_delta:
            best_epoch = epoch
            best_val_loss = val_metrics.loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics.to_dict(),
                },
                destination,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    best_checkpoint = torch.load(destination, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    return FitResult(
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        epochs_completed=len(history),
        checkpoint_path=destination,
    )
