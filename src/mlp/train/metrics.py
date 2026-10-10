"""Small-sample-aware binary classifier evaluation and calibration."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    if len(y) != len(scores) or not len(y) or bins < 1:
        raise ValueError("Nonempty matched labels/scores and positive bin count required")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Scores must be finite probabilities in [0,1]")
    error = 0.0
    for indices in np.array_split(np.argsort(scores), min(bins, len(scores))):
        if len(indices):
            error += len(indices) / len(y) * abs(float(y[indices].mean() - scores[indices].mean()))
    return float(error)


def evaluate_binary(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) != 2:
        raise ValueError("Evaluation requires both classes")
    if len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid prediction vector")
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "ece": expected_calibration_error(y, scores),
    }


def slice_ap(frame, scores: np.ndarray) -> dict[str, float]:
    result = {}
    for channel, subset in frame.groupby("channel", sort=True):
        labels = subset["target"].to_numpy()
        if len(np.unique(labels)) == 2:
            result[str(channel)] = float(average_precision_score(labels, scores[subset.index]))
    return result
