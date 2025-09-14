from __future__ import annotations

"""Preprocessing transformers for text inputs.

Currently exposes a single scikit-learn compatible transformer, ``TextProcessor``,
which performs lightweight normalization mirroring the original project:
- remove non-alphabetic characters
- lowercasing
- stopword removal
- optional Porter stemming
"""

import re
from typing import Iterable, List

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextProcessor(BaseEstimator, TransformerMixin):
    """Lightweight text normalization for bag-of-words models.

    Args:
        lower: If True, convert text to lowercase.
        stem: If True, apply Porter stemming.
        remove_nonalpha: If True, drop all non A–Z characters.

    Notes:
        This transformer is intended for classic vectorizers (Count/Tf-idf).
        For transformer models (e.g., BERT), use the model tokenizer instead.
    """

    def __init__(self, lower: bool = True, stem: bool = False, remove_nonalpha: bool = True):
        self.lower = lower
        self.stem = stem
        self.remove_nonalpha = remove_nonalpha
        # Pre-load resources to avoid repeated lookups inside transform.
        self._stop = set(stopwords.words("english"))
        self._stemmer = PorterStemmer() if stem else None

    def fit(self, X: Iterable[str], y=None):  # noqa: N803 (sklearn API)
        """No-op fit for scikit-learn pipeline compatibility."""
        return self

    def transform(self, X: Iterable[str]) -> List[str]:  # noqa: N803 (sklearn API)
        """Transform raw texts into normalized token strings.

        Args:
            X: Iterable of raw text inputs.

        Returns:
            List of normalized strings, space-delimited tokens.
        """

        out: List[str] = []
        for text in X:
            t = text
            if self.remove_nonalpha:
                t = re.sub("[^a-zA-Z]", " ", t)
            if self.lower:
                t = t.lower()
            tokens = [w for w in t.split() if w not in self._stop]
            if self._stemmer is not None:
                tokens = [self._stemmer.stem(w) for w in tokens]
            out.append(" ".join(tokens))
        return out
