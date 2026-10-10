"""Recreate the synthetic training inputs from a registry version's run metadata."""

from __future__ import annotations

import argparse
import json
import os

import mlflow
from mlflow import MlflowClient

from mlp.data.manifest import build_manifest
from mlp.data.split import split_dataset
from mlp.data.synthetic import generate_synthetic
from mlp.train.pipeline import MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_version")
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLP_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()
    version = client.get_model_version(MODEL_NAME, args.model_version)
    run = client.get_run(version.run_id)
    seed, rows = int(run.data.params["seed"]), int(run.data.params["rows"])
    frame = generate_synthetic(rows, seed)
    splits = split_dataset(frame, seed=seed, validation_fraction=0.2, test_fraction=0.2)
    actual = build_manifest(frame, splits, seed=seed)["data_version"]
    expected = version.tags.get("data_version")
    if actual != expected:
        raise SystemExit("Reproduction failed: generated dataset hash mismatch")
    print(
        json.dumps(
            {
                "model_version": args.model_version,
                "data_version": actual,
                "matched": True,
                "scope": "data recreation only; retraining equivalence is not yet asserted",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
