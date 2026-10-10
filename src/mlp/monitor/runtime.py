"""Bounded in-memory aggregate drift reference; disabled by default for privacy."""

from __future__ import annotations

import os
import threading
from collections import deque

import numpy as np
import pandas as pd
from prometheus_client import Gauge, Histogram

from mlp.common.config import AppSettings
from mlp.data.schema import FEATURES
from mlp.data.synthetic import generate_synthetic
from mlp.monitor.drift import detect_numeric
from mlp.monitor.feedback import FeedbackWindow

DRIFT_PSI = Gauge("mlp_feature_psi", "Numeric feature PSI", ["feature"])
DRIFT_FLAG = Gauge("mlp_drift_alert", "Numeric drift alert", ["feature"])
PREDICTED_SCORE = Histogram(
    "mlp_prediction_score", "Predicted positive score",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0),
)


class DriftRuntime:
    def __init__(self, limit: int = 400, seed: int = 42, rows: int = 200) -> None:
        reference = generate_synthetic(rows, seed)
        self.reference = {name: reference[name].to_numpy() for name in FEATURES}
        self.windows: dict[str, deque[float]] = {
            name: deque(maxlen=limit) for name in FEATURES
        }
        self._lock = threading.Lock()
        self.feedback = FeedbackWindow()

    def observe(self, frame: pd.DataFrame, scores: np.ndarray) -> None:
        if os.getenv("MLP_ENABLE_DRIFT_RECORDING", "false").lower() != "true":
            return
        with self._lock:
            for name in FEATURES:
                self.windows[name].extend(float(x) for x in frame[name])
        for score in scores:
            PREDICTED_SCORE.observe(float(score))

    def report(self) -> dict:
        with self._lock:
            snapshots = {name: np.array(values, dtype=float) for name, values in self.windows.items()}
        result = {}
        for name, values in snapshots.items():
            measured = detect_numeric(self.reference[name], values)
            result[name] = measured.as_dict()
            if measured.psi is not None:
                DRIFT_PSI.labels(name).set(measured.psi)
            DRIFT_FLAG.labels(name).set(1 if measured.status == "drift" else 0)
        return result


settings = AppSettings()
runtime = DriftRuntime(seed=settings.random_seed, rows=settings.tiny_rows)
