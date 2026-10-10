"""Gate must reject ties, uncertainty, calibration and explicit smoke failures."""

from __future__ import annotations

import numpy as np
import pytest

from mlp.registry.promotion import evaluate_promotion, paired_bootstrap


def _kwargs() -> dict:
    return dict(
        champion_worst_slice=0.5,
        challenger_worst_slice=0.6,
        challenger_ece=0.03,
        challenger_p95_ms=20,
        challenger_bytes=2000,
        serving_smoke_passed=True,
        repeats=120,
    )


def test_accept_obviously_improved_challenger() -> None:
    y = np.array([0, 1] * 100)
    champion = np.full(200, 0.5)
    challenger = np.where(y == 1, 0.99, 0.01)
    decision = evaluate_promotion(y, champion, challenger, **_kwargs())
    assert decision.accepted
    assert decision.lower_bound > 0


def test_tie_rejected() -> None:
    y = np.array([0, 1] * 100)
    scores = np.linspace(0.01, 0.99, len(y))
    decision = evaluate_promotion(y, scores, scores, **_kwargs())
    assert not decision.accepted


def test_smoke_failure_is_rejected() -> None:
    y = np.array([0, 1] * 100)
    champion = np.full(200, 0.5)
    challenger = np.where(y, 0.99, 0.01)
    kw = _kwargs()
    kw["serving_smoke_passed"] = False
    assert not evaluate_promotion(y, champion, challenger, **kw).accepted


def test_calibration_regression_is_rejected() -> None:
    y = np.array([0, 1] * 100)
    kw = _kwargs()
    kw["challenger_ece"] = 0.5
    decision = evaluate_promotion(y, np.full(200, 0.5), np.where(y, 0.99, 0.01), **kw)
    assert not decision.accepted


def test_short_or_degenerate_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap(np.ones(100), np.zeros(100), np.ones(100))
