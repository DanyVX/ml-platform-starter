"""Numerical edge cases and calibration calculations."""

from __future__ import annotations

import numpy as np
import pytest

from mlp.train.metrics import evaluate_binary, expected_calibration_error


def test_perfect_calibration_tiny() -> None:
    y = np.array([0, 1])
    scores = np.array([0.0, 1.0])
    assert evaluate_binary(y, scores)["pr_auc"] == pytest.approx(1.0)
    assert expected_calibration_error(y, scores) == pytest.approx(0.0)


@pytest.mark.parametrize("scores", [[0.3, float("nan")], [-0.1, 0.8], [1.1, 0.5]])
def test_invalid_probabilities_fail(scores: list[float]) -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 1]), np.array(scores))


def test_one_class_evaluation_is_rejected() -> None:
    with pytest.raises(ValueError, match="both classes"):
        evaluate_binary(np.array([0, 0]), np.array([0.1, 0.2]))
