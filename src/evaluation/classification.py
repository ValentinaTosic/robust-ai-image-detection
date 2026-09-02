"""Test-set evaluation, per-generator metrics, and result export."""

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    log_loss,
)
from torch import nn
from torch.utils.data import DataLoader

from src.training.engine import EpochMetrics


@dataclass(frozen=True)
class EvaluationResult:
    """Overall metrics, individual predictions, and generator breakdown."""

    metrics: EpochMetrics
    predictions: pd.DataFrame
    per_generator: pd.DataFrame


def _metrics_from_frame(
    frame: pd.DataFrame,
    average_loss: float,
) -> EpochMetrics:
    """Calculate binary metrics from a prediction table."""
    labels = frame["label_index"].to_numpy(dtype=np.int64)
    predictions = frame["prediction_index"].to_numpy(dtype=np.int64)
    probabilities = frame["probability_ai"].to_numpy(dtype=np.float64)

    return EpochMetrics(
        loss=average_loss,
        accuracy=accuracy_score(labels, predictions),
        precision=precision_score(labels, predictions, zero_division=0),
        recall=recall_score(labels, predictions, zero_division=0),
        f1=f1_score(labels, predictions, zero_division=0),
        roc_auc=roc_auc_score(labels, probabilities),
    )


def _per_generator_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare each generator's AI images with its matched real source group."""
    generators = sorted(predictions.loc[predictions["label"] == "ai", "generator"].unique())
    rows: list[dict[str, float | int | str]] = []

    for generator in generators:
        group = predictions[
            (
                (predictions["label"] == "ai")
                & (predictions["generator"] == generator)
            )
            | (
                (predictions["label"] == "real")
                & (predictions["source_group"] == generator)
            )
        ]
        metrics = _metrics_from_frame(group, average_loss=float("nan"))
        rows.append(
            {
                "generator": generator,
                "n_ai": int((group["label"] == "ai").sum()),
                "n_real": int((group["label"] == "real").sum()),
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
            }
        )

    return pd.DataFrame(rows)


def evaluate_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate once and retain predictions required for detailed analysis."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in the range [0, 1].")

    model.eval()
    total_loss = 0.0
    total_examples = 0
    rows: list[dict[str, float | int | str]] = []

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            label_indices = labels.cpu().numpy().astype(np.int64)
            prediction_indices = (probabilities >= threshold).astype(np.int64)

            batch_size = len(label_indices)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            for index in range(batch_size):
                rows.append(
                    {
                        "filename": batch["filename"][index],
                        "label": "ai" if label_indices[index] == 1 else "real",
                        "label_index": int(label_indices[index]),
                        "generator": batch["generator"][index],
                        "source_group": batch["source_group"][index],
                        "probability_ai": float(probabilities[index]),
                        "prediction": "ai" if prediction_indices[index] == 1 else "real",
                        "prediction_index": int(prediction_indices[index]),
                    }
                )

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty DataLoader.")

    predictions = pd.DataFrame(rows)
    metrics = _metrics_from_frame(predictions, total_loss / total_examples)
    per_generator = _per_generator_metrics(predictions)
    return EvaluationResult(metrics, predictions, per_generator)


def evaluate_probability_predictions(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate probabilities produced by a non-PyTorch classifier.

    This keeps Logistic Regression and neural-network outputs in the same
    result format, so their metrics, predictions, and figures can be compared
    and saved by the same project utilities.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in the range [0, 1].")
    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities must have the same length.")

    predictions = rows[["filename", "label", "generator", "source_group"]].copy()
    predictions["label_index"] = predictions["label"].map({"real": 0, "ai": 1}).astype(np.int64)
    predictions["probability_ai"] = np.asarray(probabilities, dtype=np.float64)
    predictions["prediction_index"] = (
        predictions["probability_ai"] >= threshold
    ).astype(np.int64)
    predictions["prediction"] = predictions["prediction_index"].map({0: "real", 1: "ai"})

    average_loss = log_loss(
        predictions["label_index"], predictions["probability_ai"]
    )
    metrics = _metrics_from_frame(predictions, average_loss)
    per_generator = _per_generator_metrics(predictions)
    return EvaluationResult(metrics, predictions, per_generator)


def save_evaluation_outputs(
    result: EvaluationResult,
    output_dir: Path | str,
    figures_dir: Path | str | None = None,
    model_label: str = "Baseline CNN",
) -> dict[str, Path]:
    """Save metrics, predictions, and the two main evaluation figures."""
    destination = Path(output_dir)
    figure_destination = Path(figures_dir) if figures_dir is not None else destination
    destination.mkdir(parents=True, exist_ok=True)
    figure_destination.mkdir(parents=True, exist_ok=True)

    paths = {
        "metrics": destination / "test_metrics.json",
        "predictions": destination / "test_predictions.csv",
        "per_generator": destination / "per_generator_metrics.csv",
        "confusion_matrix": figure_destination / "confusion_matrix.png",
        "roc_curve": figure_destination / "roc_curve.png",
    }

    paths["metrics"].write_text(
        json.dumps(result.metrics.to_dict(), indent=2), encoding="utf-8"
    )
    result.predictions.to_csv(paths["predictions"], index=False)
    result.per_generator.to_csv(paths["per_generator"], index=False)

    matrix = confusion_matrix(
        result.predictions["label_index"],
        result.predictions["prediction_index"],
        labels=[0, 1],
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["real", "ai"],
        yticklabels=["real", "ai"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_label} confusion matrix")
    fig.tight_layout()
    fig.savefig(paths["confusion_matrix"], dpi=150)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        result.predictions["label_index"], result.predictions["probability_ai"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {result.metrics.roc_auc:.3f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_label} ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(paths["roc_curve"], dpi=150)
    plt.close(fig)

    return paths
