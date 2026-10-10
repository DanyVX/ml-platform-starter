"""Explicit, audited rollback to a known version in the MLflow registry."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from mlp.train.pipeline import MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Explicit previously evaluated champion version")
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLP_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()
    old = client.get_model_version_by_alias(MODEL_NAME, "champion")
    chosen = client.get_model_version(MODEL_NAME, args.version)
    if not chosen.run_id:
        raise SystemExit("Cannot roll back to a version without a tracked run")
    client.set_registered_model_alias(MODEL_NAME, "champion", chosen.version)
    decision = {
        "timestamp": datetime.now(UTC).isoformat(),
        "action": "manual_rollback",
        "from_version": old.version,
        "to_version": chosen.version,
        "model": MODEL_NAME,
    }
    target = Path("results/rollback_decision.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
