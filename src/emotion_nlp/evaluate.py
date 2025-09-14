from __future__ import annotations

"""Evaluation helpers for classification tasks.

Provides basic metrics commonly used for multi-class text classification
including accuracy, confusion matrix, classification report, and macro/micro
ROC AUC when probabilistic predictions are available.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def evaluate_predictions(y_true, y_pred, labels=None) -> Dict[str, object]:
    """Compute accuracy, confusion matrix, and a text report.

    Args:
        y_true: Ground-truth labels (array-like).
        y_pred: Predicted labels (array-like).
        labels: Optional list of label names used for the report and matrix.

    Returns:
        Dict containing ``accuracy`` (float), ``confusion_matrix`` (ndarray),
        and ``report`` (str).
    """

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, target_names=labels if labels is not None else None)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report}


def compute_auc(y_true, proba, classes: list[int] | None) -> Dict[str, float]:
    """Compute macro/micro ROC AUC for multi-class classification.

    Args:
        y_true: Ground-truth labels (shape: [N]).
        proba: Predicted probabilities (shape: [N, C]).
        classes: List of class indices used to binarize labels.

    Returns:
        Dictionary with ``auc_macro`` and ``auc_micro``.

    Raises:
        ValueError: If ``classes`` is ``None``.
    """

    if classes is None:
        raise ValueError("classes must be provided for multi-class AUC computation")
    y_bin = label_binarize(y_true, classes=classes)
    auc_macro = roc_auc_score(y_bin, proba, average="macro")
    auc_micro = roc_auc_score(y_bin, proba, average="micro")
    return {"auc_macro": float(auc_macro), "auc_micro": float(auc_micro)}
