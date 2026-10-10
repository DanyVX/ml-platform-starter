"""Content-addressed schema and dataset metadata; no personal inputs retained."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from mlp.data.schema import CHANNELS, FEATURES, validate_frame
from mlp.data.split import DataSplits


def canonical_csv(frame: pd.DataFrame) -> str:
    csv_text = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n")
    if not isinstance(csv_text, str):
        raise TypeError("Expected an in-memory CSV string")
    return csv_text


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def schema_hash() -> str:
    definition = {
        "features": list(FEATURES),
        "bounds": [-8.0, 8.0],
        "dtype": "float64",
        "channel": list(CHANNELS),
        "target": [0, 1],
        "nullable": False,
        "schema_version": 1,
    }
    return sha256_text(json.dumps(definition, sort_keys=True, separators=(",", ":")))


def _git_commit() -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return output.stdout.strip() if output.returncode == 0 else None


def build_manifest(
    original: pd.DataFrame, splits: DataSplits, *, seed: int
) -> dict[str, Any]:
    validate_frame(original)
    versions = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "pandera": version("pandera"),
    }
    partition_frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    return {
        "schema_version": 1,
        "data_version": sha256_text(canonical_csv(original)),
        "schema_sha256": schema_hash(),
        "rows": len(original),
        "seed": seed,
        "git_commit": _git_commit(),
        "libraries": versions,
        "partitions": {
            name: {"rows": len(data), "sha256": sha256_text(canonical_csv(data))}
            for name, data in partition_frames.items()
        },
    }


def verify_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_sha256") != schema_hash():
        raise ValueError("Dataset schema fingerprint does not match this code revision")
    counts = 0
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.csv"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        spec = manifest["partitions"][name]
        if digest != spec["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}.csv")
        frame = pd.read_csv(path)
        validate_frame(frame)
        if len(frame) != spec["rows"]:
            raise ValueError(f"Row count mismatch for {name}.csv")
        counts += len(frame)
    if counts != manifest["rows"]:
        raise ValueError("Total partition row count mismatch")
    return manifest
