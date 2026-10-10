"""Deterministic splits, integrity checks and CLI smoke tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mlp import __main__
from mlp.data.manifest import build_manifest, schema_hash, verify_manifest
from mlp.data.pipeline import build_data
from mlp.data.split import DataSplits, split_dataset, verify_no_leakage
from mlp.data.synthetic import generate_synthetic


def test_stratified_splits_are_reproducible_and_disjoint() -> None:
    frame = generate_synthetic(200, 42)
    first = split_dataset(frame, seed=42, validation_fraction=0.2, test_fraction=0.2)
    again = split_dataset(frame, seed=42, validation_fraction=0.2, test_fraction=0.2)
    for name in ("train", "validation", "test"):
        part = getattr(first, name)
        pd.testing.assert_frame_equal(part, getattr(again, name))
        assert set(part["target"]) == {0, 1}
    assert sum(map(len, [first.train, first.validation, first.test])) == len(frame)
    verify_no_leakage(first)


def test_overlap_is_detected() -> None:
    frame = generate_synthetic(80, 12)
    splits = DataSplits(frame.iloc[:20], frame.iloc[:20], frame.iloc[20:])
    with pytest.raises(ValueError, match="overlap"):
        verify_no_leakage(splits)


def test_duplicate_source_rows_detected() -> None:
    frame = generate_synthetic(200, 8)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        split_dataset(doubled, seed=8, validation_fraction=0.2, test_fraction=0.2)


@pytest.mark.parametrize("val,test", [(0.0, 0.2), (0.2, 0.0), (0.3, 0.3), (0.5, 0.1)])
def test_invalid_split_fractions(val: float, test: float) -> None:
    frame = generate_synthetic(80, 42)
    with pytest.raises(ValueError):
        split_dataset(frame, seed=42, validation_fraction=val, test_fraction=test)


def test_tiny_and_degenerate_frames_fail_loudly() -> None:
    frame = generate_synthetic(100, 42)
    with pytest.raises(ValueError, match="At least 40"):
        split_dataset(frame.iloc[:10], seed=42, validation_fraction=0.2, test_fraction=0.2)
    frame["target"] = 0
    with pytest.raises(ValueError, match="Both classes"):
        split_dataset(frame, seed=42, validation_fraction=0.2, test_fraction=0.2)


def test_manifest_includes_reproducibility_provenance() -> None:
    frame = generate_synthetic(200, 42)
    splits = split_dataset(frame, seed=42, validation_fraction=0.2, test_fraction=0.2)
    manifest = build_manifest(frame, splits, seed=42)
    assert manifest["schema_sha256"] == schema_hash()
    assert len(manifest["data_version"]) == 64
    assert manifest["rows"] == 200
    assert manifest["seed"] == 42
    assert "python" in manifest["libraries"]
    assert manifest["partitions"]["train"]["rows"] == 120


def test_build_verify_and_tampering(tmp_path) -> None:
    directory = tmp_path / "data"
    manifest = build_data(directory, rows=200, seed=42)
    assert verify_manifest(directory) == manifest
    assert build_data(directory, rows=200, seed=42) == manifest

    path = directory / "test.csv"
    path.write_text(path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_manifest(directory)


def test_schema_hash_mismatch_fails(tmp_path) -> None:
    directory = tmp_path / "data"
    build_data(directory, rows=100, seed=42)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_sha256"] = "00" * 32
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema fingerprint"):
        verify_manifest(directory)


def test_rowcount_mismatch_fails(tmp_path) -> None:
    directory = tmp_path / "data"
    build_data(directory, rows=100, seed=42)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["rows"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Total partition"):
        verify_manifest(directory)


def test_cli_build_and_verify(tmp_path, capsys) -> None:
    directory = tmp_path / "data"
    assert __main__.main(["data-build", "--output-dir", str(directory), "--rows", "80"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 80
    assert __main__.main(["data-verify", "--data-dir", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
