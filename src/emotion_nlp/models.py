from __future__ import annotations

"""Model builders for baselines and optional BERT initialization.

Provides a scikit-learn pipeline for a RandomForest baseline and a minimal
helper to instantiate DistilBERT components. The baseline mirrors the
original project choices while adding sensible defaults and documentation.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from .preprocess import TextProcessor


def build_rf_pipeline(
    max_features: int = 3000,
    n_estimators: int = 200,
    max_depth: int | None = 100,
    min_samples_split: int = 50,
    min_samples_leaf: int = 2,
    max_features_mode: str = "sqrt",
) -> Pipeline:
    """Construct a RandomForest text classification pipeline.

    The pipeline includes a lightweight text processor, a bag-of-words
    vectorizer, and a RandomForest classifier. Hyperparameter values are
    chosen to be stable for small-to-medium datasets.

    Args:
        max_features: Maximum vocabulary size for the vectorizer.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree; ``None`` for unlimited.
        min_samples_split: Min samples to split an internal node.
        min_samples_leaf: Min samples required at a leaf node.
        max_features_mode: Feature selection mode per split (e.g., ``"sqrt"``).

    Returns:
        A scikit-learn ``Pipeline`` ready to ``fit`` and ``predict``.
    """

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features_mode,
    )
    pipe = Pipeline([
        ("text_processing", TextProcessor(lower=True, stem=False)),
        ("vectorizer", CountVectorizer(max_features=max_features)),
        ("classifier", clf),
    ])
    return pipe


# Optional: lightweight wrapper for DistilBERT fine-tuning
try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    @dataclass
    class BertArtifacts:
        """Container for DistilBERT model resources.

        Attributes:
            model: Sequence classification head initialized for ``num_labels``.
            tokenizer: Matching DistilBERT tokenizer.
        """

        model: "DistilBertForSequenceClassification"
        tokenizer: "DistilBertTokenizer"

    def build_distilbert(num_labels: int) -> BertArtifacts:
        """Initialize DistilBERT for sequence classification.

        Args:
            num_labels: Number of target classes for the classifier head.

        Returns:
            ``BertArtifacts`` bundling the model and tokenizer.
        """

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        return BertArtifacts(model=model, tokenizer=tokenizer)
except Exception:  # pragma: no cover - optional dependency path
    pass
