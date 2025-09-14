"""Emotion NLP
================

Duke ECE Data Science final project (modernized).

This package provides a small, modular NLP pipeline for multi-class
emotion classification. It is designed to be readable and easy to extend
for interviews, code reviews, and future coursework.

Subpackages
-----------
- ``data``: Loading, cleaning, and splitting utilities.
- ``preprocess``: Text preprocessing transformer(s).
- ``models``: Baseline scikit-learn pipeline(s) and optional BERT helper.
- ``train``: Simple CLI to train the baseline and export artifacts.
- ``evaluate``: Common evaluation helpers (accuracy, AUC, confusion matrix).
"""

__all__ = [
    "data",
    "preprocess",
    "models",
    "train",
    "evaluate",
]
