"""Train/serve schema edge cases. No real people or remote data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from mlp.data.schema import FEATURES, validate_frame
from mlp.data.synthetic import generate_synthetic

INVALID_SCHEMA = (SchemaError, SchemaErrors)


def test_generator_is_deterministic() -> None:
    first = generate_synthetic(200, 42)
    pd.testing.assert_frame_equal(first, generate_synthetic(200, 42))
    assert not first.equals(generate_synthetic(200, 43))
    assert set(first["target"]) == {0, 1}


@pytest.mark.parametrize("rows", [0, 1, 39, 10001])
def test_generator_refuses_unsupported_batch_sizes(rows: int) -> None:
    with pytest.raises(ValueError, match="row count"):
        generate_synthetic(rows, 42)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_generator_rejects_bad_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="Seed"):
        generate_synthetic(100, seed)


def test_serve_schema_excludes_target() -> None:
    training = generate_synthetic(100, 2)
    serve = training.drop(columns="target")
    pd.testing.assert_frame_equal(validate_frame(serve, training=False), serve)
    with pytest.raises(INVALID_SCHEMA):
        validate_frame(serve, training=True)
    with pytest.raises(INVALID_SCHEMA):
        validate_frame(training, training=False)


@pytest.mark.parametrize(
    "change", ["unknown", "string_number", "nan", "inf", "too_large", "null_channel"]
)
def test_invalid_input_rejected(change: str) -> None:
    frame = generate_synthetic(80, 11)
    if change == "unknown":
        frame.loc[0, "channel"] = "unseen_category"
    elif change == "string_number":
        frame["f_0"] = frame["f_0"].astype(str)
    elif change == "nan":
        frame.loc[0, "f_0"] = np.nan
    elif change == "inf":
        frame.loc[0, "f_0"] = np.inf
    elif change == "too_large":
        frame.loc[0, "f_0"] = 99.0
    else:
        frame.loc[0, "channel"] = None
    with pytest.raises((ValueError, *INVALID_SCHEMA)):
        validate_frame(frame)


def test_extra_and_missing_columns_fail() -> None:
    frame = generate_synthetic(80, 12)
    with pytest.raises(INVALID_SCHEMA):
        validate_frame(frame.assign(unexpected="private"))
    with pytest.raises(INVALID_SCHEMA):
        validate_frame(frame.drop(columns=FEATURES[0]))


def test_empty_batch_rejected() -> None:
    frame = generate_synthetic(80, 13)
    with pytest.raises(ValueError, match="at least one row"):
        validate_frame(frame.iloc[0:0])
