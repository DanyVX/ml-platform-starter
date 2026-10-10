"""Shared train/serve contract for the synthetic classification pilot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

FEATURES = tuple(f"f_{index}" for index in range(6))
CHANNELS = ("web", "mobile", "kiosk")

FEATURE_COLUMNS = {
    name: pa.Column(float, pa.Check.in_range(-8.0, 8.0), nullable=False)
    for name in FEATURES
}
CHANNEL_COLUMN = {"channel": pa.Column(str, pa.Check.isin(CHANNELS), nullable=False)}
LABEL_COLUMN = {"target": pa.Column(int, pa.Check.isin([0, 1]), nullable=False)}

TRAIN_SCHEMA = pa.DataFrameSchema(
    {**FEATURE_COLUMNS, **CHANNEL_COLUMN, **LABEL_COLUMN},
    strict=True,
    coerce=False,
    ordered=True,
)
SERVE_SCHEMA = pa.DataFrameSchema(
    {**FEATURE_COLUMNS, **CHANNEL_COLUMN},
    strict=True,
    coerce=False,
    ordered=True,
)


def validate_frame(frame: pd.DataFrame, *, training: bool = True) -> pd.DataFrame:
    """Reject empty, non-finite, malformed, unexpected, or out-of-range inputs."""
    if frame.empty:
        raise ValueError("Input batch must contain at least one row")
    schema = TRAIN_SCHEMA if training else SERVE_SCHEMA
    validated: pd.DataFrame = schema.validate(frame, lazy=True)
    numeric = validated.loc[:, list(FEATURES)].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric features must be finite")
    return validated
