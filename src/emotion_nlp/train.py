from __future__ import annotations

"""CLI entrypoint and training orchestration for the baseline model.

This script wires together the data utilities, baseline model, and evaluation.
By default it trains a RandomForest-based pipeline and exports:
- CSV splits to ``artifacts/data/``
- metrics to ``artifacts/rf_metrics.txt``
- a simple training curve figure to ``assets/generated/training_curve_rf.png``
"""

import os
import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import build_splits, save_splits
from .models import build_rf_pipeline
from .evaluate import evaluate_predictions, compute_auc


def train_rf(
    data_dir: str,
    artifacts_dir: str = "artifacts",
    assets_dir: str = "assets/generated",
    n_per_file: int | None = None,
) -> None:
    """Train the RandomForest baseline and export artifacts.

    Args:
        data_dir: Directory containing the raw ``*_1000_clean.txt`` files.
        artifacts_dir: Directory where splits and metrics are written.
        assets_dir: Directory where generated figures are written.
        n_per_file: Optional per-class cap to speed up local iterations.
    """

    # Build randomized splits and persist them for inspection/reproducibility.
    splits = build_splits(data_dir=data_dir, n_per_file=n_per_file)
    save_splits(splits, out_dir=os.path.join(artifacts_dir, "data"))

    pipe = build_rf_pipeline()
    pipe.fit(splits.train["text"], splits.train["label"])

    # Evaluation
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    metrics = {}
    for name, df in {"train": splits.train, "val": splits.val, "test": splits.test}.items():
        pred = pipe.predict(df["text"])
        m = evaluate_predictions(df["label"], pred, labels=list(pipe.classes_))
        metrics[name] = m

    # Optionally compute AUC if the model exposes predict_proba (RF does).
    try:
        proba = pipe.predict_proba(splits.test["text"]).astype(float)
        auc = compute_auc(splits.test["label"], proba, classes=list(range(len(pipe.classes_))))
        metrics["test"].update(auc)
    except Exception:
        # Some models may not provide probabilities; skip silently.
        pass

    # Save metrics
    with open(os.path.join(artifacts_dir, "rf_metrics.txt"), "w", encoding="utf-8") as f:
        for split, m in metrics.items():
            f.write(f"[{split}]\n")
            f.write(f"accuracy: {m['accuracy']:.4f}\n")
            if "auc_macro" in m:
                f.write(f"auc_macro: {m['auc_macro']:.4f}\n")
                f.write(f"auc_micro: {m['auc_micro']:.4f}\n")
            f.write("classification report:\n")
            f.write(str(m["report"]) + "\n\n")

    # Produce a simple training curve by varying n_estimators.
    train_acc, val_acc = [], []
    est_range = list(range(10, 210, 10))
    from .models import build_rf_pipeline as _rf

    for n in est_range:
        p = _rf(n_estimators=n)
        p.fit(splits.train["text"], splits.train["label"])
        train_acc.append(p.score(splits.train["text"], splits.train["label"]))
        val_acc.append(p.score(splits.val["text"], splits.val["label"]))

    plt.figure(figsize=(8, 5))
    plt.plot(est_range, train_acc, label="Train")
    plt.plot(est_range, val_acc, label="Validation")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.title("RandomForest Training Curve")
    plt.legend()
    curve_path = os.path.join(assets_dir, "training_curve_rf.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()


def main():
    """Parse CLI arguments and launch training."""

    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument("--data-dir", default="data", help="Directory with *_1000_clean.txt files")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap per-class for quick runs")
    args = parser.parse_args()

    train_rf(data_dir=args.data_dir, n_per_file=args.limit)


if __name__ == "__main__":
    main()
