"""Atomic registry-alias loader with integrity-checked last-known-good cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient

from mlp.data.schema import FEATURES
from mlp.train.pipeline import MODEL_NAME

LOG = logging.getLogger("mlp.serve")
CACHE = Path(os.getenv("MLP_CACHE_DIR", "artifacts/model-cache"))


@dataclass(frozen=True)
class Loaded:
    model: Any
    version: str
    data_version: str
    training_commit: str
    status: str


class ModelManager:
    def __init__(self, cache: Path | None = None) -> None:
        self.cache = cache or CACHE
        self._lock = threading.RLock()
        self._current: Loaded | None = None
        self._reload_lock = threading.Lock()

    def snapshot(self) -> Loaded | None:
        with self._lock:
            return self._current

    def _persist(self, candidate: Loaded) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        tmp = self.cache / "model.tmp"
        joblib.dump(candidate.model, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        metadata = {
            "sha256": digest,
            "version": candidate.version,
            "data_version": candidate.data_version,
            "training_commit": candidate.training_commit,
        }
        os.replace(tmp, self.cache / "model.joblib")
        manifest = self.cache / "manifest.tmp"
        manifest.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        os.replace(manifest, self.cache / "manifest.json")

    def _cached(self) -> Loaded:
        metadata = json.loads((self.cache / "manifest.json").read_text(encoding="utf-8"))
        path = self.cache / "model.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Corrupt last-known-good model checksum")
        # Only a locally produced trusted artifact must ever be placed in this directory.
        model = joblib.load(path)
        return Loaded(
            model, str(metadata["version"]), str(metadata["data_version"]),
            str(metadata["training_commit"]), "degraded",
        )

    def reload(self, alias: str = "champion") -> bool:
        if not self._reload_lock.acquire(blocking=False):
            return False
        try:
            try:
                mlflow.set_tracking_uri(os.getenv("MLP_TRACKING_URI", "sqlite:///mlflow.db"))
                client = MlflowClient()
                found = client.get_model_version_by_alias(MODEL_NAME, alias)
                current = self.snapshot()
                if current and current.version == str(found.version) and current.status == "ready":
                    return False
                model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{found.version}")
                smoke = {
                    **{feature: [0.0] for feature in FEATURES},
                    "channel": ["web"],
                }
                import pandas as pd

                scores = model.predict_proba(pd.DataFrame(smoke))[:, 1]
                if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                    raise ValueError("Smoke prediction returned invalid probabilities")
                candidate = Loaded(
                    model,
                    str(found.version),
                    found.tags.get("data_version", "unknown"),
                    found.tags.get("training_commit", "unknown"),
                    "ready",
                )
                self._persist(candidate)
            except (OSError, ValueError, KeyError, RuntimeError, mlflow.exceptions.MlflowException):
                LOG.exception("registry_load_failed")
                if self.snapshot() is not None:
                    return False
                try:
                    candidate = self._cached()
                except (OSError, ValueError, KeyError):
                    LOG.exception("cache_integrity_failed")
                    return False
            with self._lock:
                self._current = candidate
            return True
        finally:
            self._reload_lock.release()


manager = ModelManager()
