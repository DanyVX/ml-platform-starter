"""Fail-closed model promotion with paired bootstrap and explicit guardrails."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    delta_pr_auc: float
    lower_bound: float
    upper_bound: float
    bootstrap_samples: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def paired_bootstrap(
    labels: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
    *,
    seed: int = 42,
    repeats: int = 500,
) -> tuple[float, float, float]:
    if not (len(labels) == len(champion) == len(challenger)) or len(labels) < 30:
        raise ValueError("Paired bootstrap needs at least 30 aligned samples")
    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes must be present")
    if not (np.isfinite(champion).all() and np.isfinite(challenger).all()):
        raise ValueError("Nonfinite predictions cannot be evaluated")
    observed = float(
        average_precision_score(labels, challenger) - average_precision_score(labels, champion)
    )
    rng = np.random.default_rng(seed)
    outcomes: list[float] = []
    for _ in range(repeats):
        idx = rng.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        outcomes.append(
            float(
                average_precision_score(labels[idx], challenger[idx])
                - average_precision_score(labels[idx], champion[idx])
            )
        )
    if len(outcomes) < repeats * 0.8:
        raise ValueError("Bootstrap samples repeatedly lacked one target class")
    lo, hi = np.quantile(outcomes, [0.025, 0.975])
    return observed, float(lo), float(hi)


def evaluate_promotion(
    labels: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
    *,
    min_margin: float = 0.01,
    champion_worst_slice: float,
    challenger_worst_slice: float,
    challenger_ece: float,
    challenger_p95_ms: float,
    challenger_bytes: int,
    serving_smoke_passed: bool,
    repeats: int = 500,
) -> GateDecision:
    delta, low, high = paired_bootstrap(labels, champion, challenger, repeats=repeats)
    problems = []
    if not serving_smoke_passed:
        problems.append("serving smoke test failed")
    if delta <= min_margin or low <= 0:
        problems.append("primary metric lacks paired-bootstrap support")
    if challenger_worst_slice < champion_worst_slice - 0.05:
        problems.append("worst slice regression")
    if challenger_ece > 0.15:
        problems.append("calibration exceeds limit")
    if challenger_p95_ms > 250:
        problems.append("latency budget exceeded")
    if challenger_bytes > 20_000_000:
        problems.append("model size exceeded")
    if not np.isfinite([challenger_worst_slice, challenger_ece, challenger_p95_ms]).all():
        problems.append("nonfinite guardrail")
    return GateDecision(
        accepted=not problems,
        reason="accepted" if not problems else "; ".join(problems),
        delta_pr_auc=delta,
        lower_bound=low,
        upper_bound=high,
        bootstrap_samples=repeats,
    )


def save_decision(decision: GateDecision, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(decision.to_json() + "\n", encoding="utf-8")
