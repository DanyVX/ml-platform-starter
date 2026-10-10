"""Hash stability and rollback under intentionally degraded challengers."""

from __future__ import annotations

import pytest

from mlp.serve.routing import CanaryGuard, Observation, choose_route


def test_canary_sticky_edge_percentages() -> None:
    assert choose_route("same-user", 0) == "champion"
    assert choose_route("same-user", 100) == "challenger"
    assert choose_route("same-user", 10) == choose_route("same-user", 10)


@pytest.mark.parametrize("bad", [-1, 101, float("nan"), float("inf")])
def test_bad_percentage_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        choose_route("key", bad)


def test_rollback_after_bad_latency() -> None:
    guard = CanaryGuard(percent=20, min_samples=4, max_p95_ms=50)
    for _ in range(3):
        assert not guard.observe(Observation(False, 500, 0))
    assert guard.observe(Observation(False, 500, 0))
    assert guard.percent == 0
    assert guard.rollback_reason == "latency_p95"
    assert guard.route("any") == "champion"


def test_error_trigger_and_reset() -> None:
    guard = CanaryGuard(percent=100, min_samples=2)
    assert not guard.observe(Observation(True, 2, 0))
    assert guard.observe(Observation(True, 2, 0))
    guard.reset(10)
    assert guard.rollback_reason is None
    assert guard.percent == 10
