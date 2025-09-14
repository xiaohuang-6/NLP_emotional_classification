from __future__ import annotations

"""Data loading, cleaning, and split utilities.

This module reproduces (and slightly hardens) the original project logic for
constructing a supervised dataset from six per-class text files. Each line in a
file contains a sample and a class label separated by a semicolon (``;``).

Key behaviors preserved from the original code:
- Lines are cleaned to remove stray commas and tabs.
- If more than one semicolon is present, everything before the last semicolon
  is joined together (to avoid accidental extra field splits), and the final
  segment is treated as the label.
- Stratification is not applied; instead we do a shuffled split with fixed
  ratios for train/val/test.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import pandas as pd


EMOTION_FILES = [
    "love_1000_clean.txt",
    "joy_1000_clean.txt",
    "sad_1000_clean.txt",
    "fear_1000_clean.txt",
    "surprise_1000_clean.txt",
    "anger_1000_clean.txt",
]


@dataclass
class DatasetSplits:
    """Container for dataset splits.

    Attributes:
        train: Training split with columns ``text`` and ``label``.
        val: Validation split with columns ``text`` and ``label``.
        test: Test split with columns ``text`` and ``label``.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _clean_semicolon_line(line: str) -> str:
    """Normalize a raw line so it can be split into ``text`` and ``label``.

    - Trims whitespace and removes commas and tabs.
    - If the line contains 2+ semicolons, join everything before the last
      semicolon as the text field; keep the last segment as the label.

    Args:
        line: Raw line from the source file.

    Returns:
        A cleaned line with a single semicolon separating text and label.
    """

    line = line.strip().replace(",", "").replace("\t", "")
    # Guard against accidental extra semicolons in the text field.
    if line.count(";") >= 2:
        parts = line.split(";")
        line = ";".join(["".join(parts[:-1]), parts[-1]])
    return line


def load_all_lines(data_dir: str, n_per_file: int | None = None) -> List[str]:
    """Load, clean, and aggregate lines from all emotion files.

    Args:
        data_dir: Directory containing the six ``*_1000_clean.txt`` files.
        n_per_file: Optional cap per-class for quick experimentation.

    Returns:
        A shuffled list of cleaned lines where each line contains
        ``text;label``.
    """

    lines: List[str] = []
    for filename in EMOTION_FILES:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = f.readlines()
        if n_per_file is not None:
            raw = raw[: n_per_file]
        cleaned = [_clean_semicolon_line(x) for x in raw if x.strip()]
        lines.extend(cleaned)
    random.shuffle(lines)
    return lines


def to_dataframe(lines: List[str]) -> pd.DataFrame:
    """Convert cleaned lines to a two-column DataFrame.

    Args:
        lines: Cleaned lines with a single semicolon separator.

    Returns:
        DataFrame with columns ``text`` and ``label``.
    """

    df = pd.DataFrame([x.split(";", 1) for x in lines], columns=["text", "label"])  # type: ignore[arg-type]
    return df


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 2 / 3,
    val_ratio: float = 1 / 6,
    seed: int = 42,
) -> DatasetSplits:
    """Shuffle and split a dataset into train/val/test.

    This mirrors the original project’s 2/3 train, 1/6 validation, and the
    remainder as test, using a reproducible shuffle.

    Args:
        df: Input DataFrame with columns ``text`` and ``label``.
        train_ratio: Proportion of samples to allocate for training.
        val_ratio: Proportion of samples to allocate for validation.
        seed: Random seed used for shuffling.

    Returns:
        ``DatasetSplits`` containing train, val, and test DataFrames.
    """

    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return DatasetSplits(train=train, val=val, test=test)


def build_splits(
    data_dir: str,
    n_per_file: int | None = None,
    seed: int = 42,
) -> DatasetSplits:
    """Convenience wrapper to load, clean, and split the dataset.

    Args:
        data_dir: Directory containing the raw class files.
        n_per_file: Optional per-class cap for faster iterations.
        seed: Random seed for the shuffle.

    Returns:
        ``DatasetSplits`` with train/val/test DataFrames.
    """

    lines = load_all_lines(data_dir, n_per_file)
    df = to_dataframe(lines)
    return split_dataframe(df, seed=seed)


def save_splits(splits: DatasetSplits, out_dir: str) -> Dict[str, str]:
    """Persist dataset splits to CSV files.

    Args:
        splits: Train/val/test splits to save.
        out_dir: Destination directory for the CSV files.

    Returns:
        Mapping of split name to the written file path.
    """

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "train": os.path.join(out_dir, "train.csv"),
        "val": os.path.join(out_dir, "val.csv"),
        "test": os.path.join(out_dir, "test.csv"),
    }
    splits.train.to_csv(paths["train"], index=False)
    splits.val.to_csv(paths["val"], index=False)
    splits.test.to_csv(paths["test"], index=False)
    return paths
