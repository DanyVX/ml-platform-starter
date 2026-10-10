"""Opt-in MLflow registration integration test; local SQLite only."""

from __future__ import annotations

from mlflow import MlflowClient

from mlp.common.config import AppSettings
from mlp.train.pipeline import MODEL_NAME, train_and_register


def test_train_and_register(tmp_path, monkeypatch) -> None:
    tracking = (tmp_path / "tracking.db").as_posix()
    monkeypatch.setenv("MLP_TRACKING_URI", f"sqlite:///{tracking}")
    monkeypatch.chdir(tmp_path)
    report = train_and_register(AppSettings(_env_file=None, tiny_rows=200))
    assert report["test"]["pr_auc"] > 0
    assert report["version"]
    assert MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
