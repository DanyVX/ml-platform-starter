"""Seeded, license-clear synthetic classification fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from mlp.data.schema import CHANNELS, FEATURES, validate_frame


def generate_synthetic(rows: int, seed: int) -> pd.DataFrame:
    if rows < 40 or rows > 10_000:
        raise ValueError("Synthetic row count must be between 40 and 10000")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must fit in unsigned 32 bits")

    features, target = make_classification(
        n_samples=rows,
        n_features=len(FEATURES),
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],
        flip_y=0.0,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        np.clip(features, -8.0, 8.0).astype("float64"), columns=list(FEATURES)
    )
    frame["channel"] = rng.choice(CHANNELS, size=rows, p=[0.6, 0.3, 0.1])
    frame["target"] = target.astype("int64")
    return validate_frame(frame)
