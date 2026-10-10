"""Explicit PSI/KS drift monitoring with insufficient-data state."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class DriftMetric:
    status: str
    psi: float | None
    ks_statistic: float | None
    ks_pvalue: float | None
    reference_count: int
    observed_count: int

    def as_dict(self) -> dict:
        return asdict(self)


def _clean(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def population_stability_index(reference: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
    reference = _clean(reference)
    observed = _clean(observed)
    if len(reference) < 2 or len(observed) < 2 or bins < 2:
        raise ValueError("PSI requires at least two finite samples per window")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError("PSI reference has too few unique quantiles")
    edges[0] = -np.inf
    edges[-1] = np.inf
    baseline, _ = np.histogram(reference, bins=edges)
    current, _ = np.histogram(observed, bins=edges)
    expected = np.maximum(baseline / len(reference), 1e-6)
    actual = np.maximum(current / len(observed), 1e-6)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def categorical_psi(reference: list[str], observed: list[str]) -> float:
    if not reference or not observed:
        raise ValueError("Categorical PSI needs nonempty reference and observed sets")
    categories = set(reference) | set(observed)
    if len(categories) > 100:
        raise ValueError("Too many categorical levels (maximum 100)")
    return float(sum(
        ((observed.count(category) / len(observed)) - (reference.count(category) / len(reference)))
        * math.log(
            max(observed.count(category) / len(observed), 1e-6)
            / max(reference.count(category) / len(reference), 1e-6)
        )
        for category in categories
    ))


def detect_numeric(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    min_size: int = 40,
    psi_threshold: float = 0.25,
    alpha: float = 0.01,
) -> DriftMetric:
    base, window = _clean(reference), _clean(observed)
    if min(len(base), len(window)) < min_size:
        return DriftMetric("insufficient_data", None, None, None, len(base), len(window))
    if len(np.unique(base)) < 3:
        return DriftMetric("insufficient_variance", None, None, None, len(base), len(window))
    psi = population_stability_index(base, window)
    result = ks_2samp(base, window, method="auto")
    status = "drift" if psi > psi_threshold and float(result.pvalue) < alpha else "stable"
    return DriftMetric(status, psi, float(result.statistic), float(result.pvalue), len(base), len(window))
