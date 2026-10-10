"""Stratified, leakage-checked splits for independent synthetic samples."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from mlp.data.schema import FEATURES, validate_frame


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _fingerprints(frame: pd.DataFrame) -> set[int]:
    """Feature-only row identity; label disagreement must not bypass leakage checks."""
    columns = [*FEATURES, "channel"]
    hashes = pd.util.hash_pandas_object(frame.loc[:, columns], index=False)
    return set(int(value) for value in hashes.to_numpy())


def verify_no_leakage(splits: DataSplits) -> None:
    train, validation, test = (
        _fingerprints(splits.train),
        _fingerprints(splits.validation),
        _fingerprints(splits.test),
    )
    if train & validation or train & test or validation & test:
        raise ValueError("Exact feature rows overlap across dataset splits")


def split_dataset(
    frame: pd.DataFrame, *, seed: int, validation_fraction: float, test_fraction: float
) -> DataSplits:
    validate_frame(frame)
    if not (0 < validation_fraction < 0.5 and 0 < test_fraction < 0.5):
        raise ValueError("Validation and test fractions must be between 0 and 0.5")
    holdout_fraction = validation_fraction + test_fraction
    if holdout_fraction >= 0.5:
        raise ValueError("Validation plus test fractions must be below 0.5")
    if len(frame) < 40:
        raise ValueError("At least 40 labeled rows are required")
    counts = frame["target"].value_counts()
    if len(counts) != 2 or int(counts.min()) < 3:
        raise ValueError("Both classes need enough examples for stratified splitting")
    if len(_fingerprints(frame)) != len(frame):
        raise ValueError("Duplicate feature rows detected in source data")
    try:
        train, held = train_test_split(
            frame, test_size=holdout_fraction, random_state=seed, stratify=frame["target"]
        )
        validation, test = train_test_split(
            held,
            test_size=test_fraction / holdout_fraction,
            random_state=seed,
            stratify=held["target"],
        )
    except ValueError as exc:
        raise ValueError("Dataset is too small or imbalanced for stratified splits") from exc
    splits = DataSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
    verify_no_leakage(splits)
    return splits
