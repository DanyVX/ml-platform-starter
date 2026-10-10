"""Manual fail-closed promotion command using aligned synthetic holdout artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient

from mlp.registry.promotion import evaluate_promotion, save_decision
from mlp.train.pipeline import MODEL_NAME


def _holdout(client: MlflowClient, version: str) -> dict:
    entity = client.get_model_version(MODEL_NAME, version)
    path = client.download_artifacts(entity.run_id, "holdout.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/promotion_decision.json"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLP_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()
    challenger = client.get_model_version_by_alias(MODEL_NAME, "challenger")
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        client.set_registered_model_alias(MODEL_NAME, "champion", challenger.version)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"accepted": True, "reason": "first deployment, no prior champion"}, indent=2
            ),
            encoding="utf-8",
        )
        return
    if champion.version == challenger.version:
        raise SystemExit("Rejected: challenger is already champion")
    reference = _holdout(client, champion.version)
    candidate = _holdout(client, challenger.version)
    if reference["data_version"] != candidate["data_version"]:
        raise SystemExit("Rejected: datasets differ; paired comparison is invalid")
    if reference["labels"] != candidate["labels"]:
        raise SystemExit("Rejected: holdout labels/order differ")
    decision = evaluate_promotion(
        np.array(candidate["labels"]),
        np.array(reference["scores"]),
        np.array(candidate["scores"]),
        champion_worst_slice=float(reference["worst_slice"]),
        challenger_worst_slice=float(candidate["worst_slice"]),
        challenger_ece=float(candidate["ece"]),
        challenger_p95_ms=float(candidate["latency_p95_ms"]),
        challenger_bytes=int(candidate["model_bytes"]),
        serving_smoke_passed=bool(candidate["smoke"]),
    )
    save_decision(decision, args.output)
    print(decision.to_json())
    if not decision.accepted:
        raise SystemExit("Promotion rejected; champion alias unchanged")
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger.version)


if __name__ == "__main__":
    main()
