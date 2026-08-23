"""Evaluation utilities and result export."""

from src.evaluation.classification import (
    EvaluationResult,
    evaluate_predictions,
    save_evaluation_outputs,
)

__all__ = ["EvaluationResult", "evaluate_predictions", "save_evaluation_outputs"]
