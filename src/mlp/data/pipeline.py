"""One-command deterministic synthetic fixture build."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mlp.data.manifest import build_manifest, canonical_csv, verify_manifest
from mlp.data.split import split_dataset
from mlp.data.synthetic import generate_synthetic


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def build_data(
    directory: Path,
    *,
    rows: int = 200,
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    frame = generate_synthetic(rows, seed)
    splits = split_dataset(
        frame,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    directory.mkdir(parents=True, exist_ok=True)
    parts = {"train": splits.train, "validation": splits.validation, "test": splits.test}
    for name, data in parts.items():
        _atomic_text(directory / f"{name}.csv", canonical_csv(data))
    manifest = build_manifest(frame, splits, seed=seed)
    _atomic_text(
        directory / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    verify_manifest(directory)
    return manifest
