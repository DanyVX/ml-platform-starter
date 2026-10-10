"""Drift, delayed labels, numerical pathologies, and privacy erase."""

from __future__ import annotations

import numpy as np
import pytest

from mlp.monitor.drift import categorical_psi, detect_numeric, population_stability_index
from mlp.monitor.feedback import FeedbackWindow


def test_psi_identical_is_zero() -> None:
    samples = np.linspace(-3, 3, 200)
    assert population_stability_index(samples, samples) == pytest.approx(0)


def test_shift_detected() -> None:
    rng = np.random.default_rng(12)
    reference = rng.normal(size=300)
    shifted = rng.normal(loc=2.0, size=300)
    result = detect_numeric(reference, shifted)
    assert result.status == "drift"
    assert result.psi is not None


def test_tiny_window_does_not_alert() -> None:
    result = detect_numeric(np.array([1.0]), np.array([20.0]))
    assert result.status == "insufficient_data"


def test_nan_heavy_is_insufficient() -> None:
    array = np.full(100, float("nan"))
    assert detect_numeric(array, array).status == "insufficient_data"


def test_categorical_psi() -> None:
    assert categorical_psi(["a", "b"], ["a", "b"]) == pytest.approx(0)
    assert categorical_psi(["a"] * 95 + ["b"] * 5, ["a"] * 10 + ["b"] * 90) > 0.1


def test_feedback_join_and_erasure() -> None:
    window = FeedbackWindow()
    for i in range(50):
        window.add_prediction(str(i), float(i) / 50)
        assert window.add_label(str(i), int(i > 25))
    assert window.performance()["status"] == "measured"
    window.erase("49")
    assert not window.add_label("49", 1)


def test_unmatched_label_returns_false() -> None:
    assert not FeedbackWindow().add_label("unknown", 1)
