"""Measured synthetic drift detection and false-alarm rates, never canned numbers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlp.common.environment import environment_report
from mlp.monitor.drift import detect_numeric


def main() -> None:
    rng = np.random.default_rng(42)
    trials = 50
    shifted = 0
    false_alarm = 0
    for _ in range(trials):
        ref = rng.normal(size=300)
        normal = rng.normal(size=300)
        abnormal = rng.normal(loc=1.0, size=300)
        false_alarm += detect_numeric(ref, normal).status == "drift"
        shifted += detect_numeric(ref, abnormal).status == "drift"
    report = {
        "experiment": "synthetic_shift_normal_mean_plus_one",
        "trials": trials,
        "seed": 42,
        "shift_detection_rate": shifted / trials,
        "false_alarm_rate": false_alarm / trials,
        "limitations": "Synthetic Gaussian drift only; thresholds not validated on production.",
        "environment": environment_report(),
        "numpy_version": np.__version__,
    }
    directory = Path("results")
    directory.mkdir(exist_ok=True)
    path = directory / "drift_injection.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Measurement written: {path}")


if __name__ == "__main__":
    main()
