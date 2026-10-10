"""Serving contract and race-safe immutable snapshot checks."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
from fastapi.testclient import TestClient

from mlp.data.schema import FEATURES
from mlp.serve import api
from mlp.serve.manager import Loaded, ModelManager


def _valid() -> dict:
    return {**{key: 0.1 for key in FEATURES}, "channel": "web"}


def test_ready_requires_model(monkeypatch) -> None:
    monkeypatch.setattr(api.manager, "_current", None)
    with TestClient(api.app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503


def test_predictions_single_batch_and_schema(monkeypatch) -> None:
    import numpy as np

    fake = Mock()
    fake.predict_proba.side_effect = lambda frame: np.column_stack(
        [np.repeat(0.4, len(frame)), np.repeat(0.6, len(frame))]
    )
    loaded = Loaded(fake, "1", "test-sha", "test-commit", "ready")
    monkeypatch.setattr(api.manager, "reload", lambda *args: True)
    monkeypatch.setattr(api.manager, "_current", loaded)
    with TestClient(api.app) as client:
        assert client.get("/readyz").json()["status"] == "ready"
        one = client.post("/v1/predict", json={"records": _valid()})
        assert one.status_code == 200
        assert one.json()["scores"] == [0.6]
        many = client.post("/v1/predict", json={"records": [_valid()] * 2})
        assert many.status_code == 200
        assert len(many.json()["scores"]) == 2
        assert client.post("/v1/predict", json={"records": [_valid()] * 129}).status_code == 422
        assert client.post("/v1/predict", json={"records": {"oops": 1}}).status_code == 422
        assert client.get("/metrics").status_code == 200


def test_cache_tamper_refused(tmp_path) -> None:
    from sklearn.dummy import DummyClassifier

    manager = ModelManager(cache=tmp_path)
    classifier = DummyClassifier(strategy="constant", constant=1).fit(
        pd.DataFrame({"f": [0, 1]}), [0, 1]
    )
    manager._persist(Loaded(classifier, "1", "data", "commit", "ready"))
    assert manager._cached().status == "degraded"
    (tmp_path / "model.joblib").write_bytes(b"corrupt")
    import pytest

    with pytest.raises(ValueError, match="checksum"):
        manager._cached()
