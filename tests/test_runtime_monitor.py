"""Privacy opt-in, bounded drift window and insufficient-data states."""

from __future__ import annotations

import numpy as np

from mlp.data.synthetic import generate_synthetic
from mlp.monitor.runtime import DriftRuntime


def test_no_raw_input_collection_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MLP_ENABLE_DRIFT_RECORDING", raising=False)
    monitor = DriftRuntime()
    frame = generate_synthetic(50, 12)
    monitor.observe(frame, np.repeat(0.5, 50))
    assert all(measured["status"] == "insufficient_data" for measured in monitor.report().values())


def test_explicit_collection_keeps_only_bounded_numeric_features(monkeypatch) -> None:
    monkeypatch.setenv("MLP_ENABLE_DRIFT_RECORDING", "true")
    monitor = DriftRuntime(limit=40)
    frame = generate_synthetic(80, 20)
    monitor.observe(frame, np.repeat(0.5, len(frame)))
    assert all(len(values) == 40 for values in monitor.windows.values())
    assert "channel" not in monitor.windows
    assert "target" not in monitor.windows
