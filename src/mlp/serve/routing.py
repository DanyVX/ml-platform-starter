"""Deterministic sticky routing and bounded canary rollback evaluation."""

from __future__ import annotations

import hashlib
import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal


def choose_route(key: str, percent: float) -> Literal["champion", "challenger"]:
    if not 0 <= percent <= 100 or not math.isfinite(percent):
        raise ValueError("Canary percentage must be finite and between 0 and 100")
    bucket = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % 10_000
    return "challenger" if bucket < int(percent * 100) else "champion"


@dataclass(frozen=True)
class Observation:
    failed: bool
    elapsed_ms: float
    divergence: float


class CanaryGuard:
    def __init__(
        self,
        percent: float = 0.0,
        *,
        min_samples: int = 20,
        max_error_rate: float = 0.05,
        max_p95_ms: float = 500.0,
        max_divergence: float = 0.35,
    ) -> None:
        choose_route("validate", percent)
        self.percent = percent
        self.min_samples = min_samples
        self.max_error_rate = max_error_rate
        self.max_p95_ms = max_p95_ms
        self.max_divergence = max_divergence
        self._samples: deque[Observation] = deque(maxlen=200)
        self._lock = threading.Lock()
        self.rollback_reason: str | None = None

    def route(self, key: str) -> Literal["champion", "challenger"]:
        with self._lock:
            percent = self.percent
        return choose_route(key, percent)

    def observe(self, observed: Observation) -> bool:
        with self._lock:
            self._samples.append(observed)
            if len(self._samples) < self.min_samples:
                return False
            error_rate = sum(item.failed for item in self._samples) / len(self._samples)
            times = sorted(item.elapsed_ms for item in self._samples)
            p95 = times[math.ceil(0.95 * len(times)) - 1]
            divergence = statistics.mean(item.divergence for item in self._samples)
            breaches = [
                (error_rate > self.max_error_rate, "error_rate"),
                (p95 > self.max_p95_ms, "latency_p95"),
                (divergence > self.max_divergence, "prediction_divergence"),
            ]
            for breached, reason in breaches:
                if breached and self.percent:
                    self.percent = 0.0
                    self.rollback_reason = reason
                    return True
        return False

    def reset(self, percent: float) -> None:
        choose_route("validate", percent)
        with self._lock:
            self.percent = percent
            self._samples.clear()
            self.rollback_reason = None
