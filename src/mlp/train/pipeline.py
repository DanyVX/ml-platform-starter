"""Train on synthetic data; evaluate before registering any model version."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlp.common.config import AppSettings
from mlp.data.manifest import build_manifest
from mlp.data.schema import FEATURES
from mlp.data.split import split_dataset
from mlp.data.synthetic import generate_synthetic
from mlp.train.metrics import evaluate_binary, slice_ap

MODEL_NAME = "synthetic-churn-pilot"


def make_pipeline(seed: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(FEATURES)),
            ("channel", OneHotEncoder(handle_unknown="ignore"), ["channel"]),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=300, solver="liblinear", random_state=seed)),
        ]
    )


def train_and_register(settings: AppSettings | None = None) -> dict[str, Any]:
    cfg = settings or AppSettings()
    dataset = generate_synthetic(cfg.tiny_rows, cfg.random_seed)
    parts = split_dataset(
        dataset,
        seed=cfg.random_seed,
        validation_fraction=cfg.validation_fraction,
        test_fraction=cfg.test_fraction,
    )
    x_cols = [*FEATURES, "channel"]
    x_train = parts.train[x_cols]
    y_train = parts.train["target"].to_numpy()
    if len(np.unique(y_train)) != 2 or min(np.bincount(y_train)) < 3:
        raise ValueError("Insufficient examples in minority class for training")
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.random_seed)
    grid = GridSearchCV(
        make_pipeline(cfg.random_seed),
        {"model__C": [0.1, 1.0, 10.0]},
        scoring="average_precision",
        cv=folds,
        n_jobs=1,
        error_score="raise",
    )
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    validation_scores = model.predict_proba(parts.validation[x_cols])[:, 1]
    test_scores = model.predict_proba(parts.test[x_cols])[:, 1]
    validation = evaluate_binary(parts.validation["target"].to_numpy(), validation_scores)
    final = evaluate_binary(parts.test["target"].to_numpy(), test_scores)
    slices = slice_ap(parts.test, test_scores)
    manifest = build_manifest(dataset, parts, seed=cfg.random_seed)
    with tempfile.TemporaryDirectory() as measured_dir:
        artifact_path = Path(measured_dir) / "sized.joblib"
        joblib.dump(model, artifact_path)
        model_bytes = artifact_path.stat().st_size
    latency_samples = []
    for _ in range(20):
        started = perf_counter()
        model.predict_proba(parts.test[x_cols])
        latency_samples.append((perf_counter() - started) * 1000)
    mlflow.set_tracking_uri(os.getenv("MLP_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("synthetic-churn-pilot")
    with mlflow.start_run(tags={"commit": str(manifest["git_commit"]), "data_sha256": manifest["data_version"]}) as run:
        mlflow.log_params({"seed": cfg.random_seed, "rows": cfg.tiny_rows, **grid.best_params_})
        mlflow.log_metrics({f"val_{k}": v for k, v in validation.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in final.items()})
        mlflow.log_metrics({"worst_slice_ap": min(slices.values()) if slices else 0.0})
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "data_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(manifest_path))
            report = {
                "validation": validation,
                "test": final,
                "slices": slices,
                "run_id": run.info.run_id,
                "data_version": manifest["data_version"],
                "git_commit": manifest["git_commit"],
                "python": platform.python_version(),
                "seed": cfg.random_seed,
            }
            (Path(temporary) / "evaluation.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            mlflow.log_artifact(str(Path(temporary) / "evaluation.json"))
            holdout = {
                "data_version": manifest["data_version"],
                "labels": parts.test["target"].astype(int).tolist(),
                "scores": test_scores.tolist(),
                "worst_slice": min(slices.values()) if slices else 0.0,
                "ece": final["ece"],
                "latency_p95_ms": float(np.percentile(latency_samples, 95)),
                "model_bytes": model_bytes,
                "smoke": bool(np.isfinite(test_scores).all()),
            }
            (Path(temporary) / "holdout.json").write_text(
                json.dumps(holdout), encoding="utf-8"
            )
            mlflow.log_artifact(str(Path(temporary) / "holdout.json"))
            (Path(temporary) / "MODEL_CARD.md").write_text(
                "# Synthetic model card\n\nFor synthetic MLOps infrastructure testing only. "
                "No demonstrated real-world accuracy, safety, or subgroup fairness.\n\n"
                + json.dumps(report, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(Path(temporary) / "MODEL_CARD.md"))
        signature = infer_signature(x_train, model.predict_proba(x_train)[:, 1])
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=x_train.head(2),
            registered_model_name=MODEL_NAME,
        )
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    matching = sorted(
        (item for item in versions if item.run_id == run.info.run_id),
        key=lambda item: int(item.version),
    )
    if not matching:
        raise RuntimeError("Model was logged but registry version was not found")
    version = matching[-1].version
    client.set_model_version_tag(MODEL_NAME, version, "data_version", str(manifest["data_version"]))
    client.set_model_version_tag(
        MODEL_NAME, version, "training_commit", str(manifest["git_commit"])
    )
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    try:
        client.get_model_version_by_alias(MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
    return {**report, "version": version, "model_name": MODEL_NAME}
