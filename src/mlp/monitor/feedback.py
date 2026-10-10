"""Privacy-minimizing delayed-label join keyed only by ephemeral prediction IDs."""

from __future__ import annotations

import threading
from collections import deque

from sklearn.metrics import average_precision_score


class FeedbackWindow:
    def __init__(self, limit: int = 2000) -> None:
        self.limit = limit
        self._lock = threading.Lock()
        self._scores: dict[str, float] = {}
        self._labels: dict[str, int] = {}
        self._order: deque[str] = deque()

    def add_prediction(self, prediction_id: str, score: float) -> None:
        if not 0 <= score <= 1:
            raise ValueError("Invalid probability")
        with self._lock:
            if prediction_id in self._scores:
                raise ValueError("Duplicate prediction identifier")
            self._scores[prediction_id] = score
            self._order.append(prediction_id)
            while len(self._order) > self.limit:
                old = self._order.popleft()
                self._scores.pop(old, None)
                self._labels.pop(old, None)

    def add_label(self, prediction_id: str, label: int) -> bool:
        if label not in (0, 1):
            raise ValueError("Label must be binary")
        with self._lock:
            if prediction_id not in self._scores:
                return False
            self._labels[prediction_id] = label
            return True

    def erase(self, prediction_id: str) -> None:
        with self._lock:
            self._scores.pop(prediction_id, None)
            self._labels.pop(prediction_id, None)

    def performance(self) -> dict[str, float | int | str]:
        with self._lock:
            ids = [i for i in self._labels if i in self._scores]
            positives = [self._labels[i] for i in ids]
            if len(ids) < 20 or len(set(positives)) != 2:
                return {"status": "insufficient_data", "labeled": len(ids)}
            return {
                "status": "measured",
                "labeled": len(ids),
                "pr_auc": float(average_precision_score(
                    positives, [self._scores[i] for i in ids]
                )),
            }
