"""Load and run the active BETO+BiLSTM classifier."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .model import BETOBiLSTMClassifier, LABELS


class ActiveModelLoader:
    """Coordinate inference for the single active classifier."""

    def __init__(self, model_dir: Path) -> None:
        self.model = BETOBiLSTMClassifier(model_dir)

    @property
    def is_loaded(self) -> bool:
        return self.model.is_loaded

    def load(self) -> None:
        self.model.load()

    def predict(self, text: str) -> list[dict[str, str | int | float]]:
        if not text.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        started = time.perf_counter()
        results, _ = self.model.predict([text])
        label_id, confidence = results[0]
        return [self._prediction(label_id, confidence, (time.perf_counter() - started) * 1000)]

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Every text must contain at least one non-whitespace character.")
        started = time.perf_counter()
        results, _ = self.model.predict(texts)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return [
            {"text": text, "predictions": [self._prediction(label_id, confidence, elapsed_ms / len(texts))]}
            for text, (label_id, confidence) in zip(texts, results)
        ]

    @staticmethod
    def _prediction(label_id: int, confidence: float, latency_ms: float) -> dict[str, str | int | float]:
        return {
            "model_name": "BETO + BiLSTM",
            "label": LABELS[int(label_id)],
            "label_id": int(label_id),
            "confidence": round(float(confidence), 6),
            "latency_ms": round(float(latency_ms), 3),
            "rank": 1,
        }
