"""Load and run BETO plus serialized classical baseline models."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any

from joblib import load

from .model import BETOBiLSTMClassifier, LABELS, RequirementClassifier


class BenchmarkModelLoader:
    """Coordinate inference for all three local classifiers."""

    def __init__(self, beto: RequirementClassifier, model_dir: Path) -> None:
        self.beto = beto
        self.model_dir = model_dir
        self.beto_lstm = BETOBiLSTMClassifier(model_dir / "beto_lstm_rf_rnf")
        self._baselines: dict[str, Any] = {}
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether core benchmark models are ready."""
        return self.beto.is_loaded and set(self._baselines) == {"random_forest", "naive_bayes"}

    @property
    def lstm_loaded(self) -> bool:
        """Return whether optional BETO+BiLSTM artifact is available."""
        return self.beto_lstm.is_loaded

    def load(self) -> None:
        """Load baseline joblib artifacts from the local models directory."""
        required = {
            "random_forest": self.model_dir / "random_forest_pipeline.joblib",
            "naive_bayes": self.model_dir / "naive_bayes_pipeline.joblib",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing baseline model files: {', '.join(missing)}")
        with self._lock:
            self._baselines = {name: load(path) for name, path in required.items()}
        try:
            self.beto_lstm.load()
        except FileNotFoundError:
            # Keep existing three-model service usable until LSTM training finishes.
            pass

    def predict(self, text: str) -> list[dict[str, str | int | float]]:
        """Predict one raw requirement with each model, sorted by confidence."""
        if not text.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        if not self.is_loaded:
            raise RuntimeError("Benchmark models are not loaded.")

        predictions: list[dict[str, str | int | float]] = []
        started = time.perf_counter()
        beto_results, _ = self.beto.predict([text])
        beto_elapsed = (time.perf_counter() - started) * 1000
        beto_label, beto_confidence = beto_results[0]
        predictions.append(
            self._prediction("BETO", beto_label, beto_confidence, beto_elapsed)
        )

        if self.beto_lstm.is_loaded:
            started = time.perf_counter()
            lstm_results, _ = self.beto_lstm.predict([text])
            lstm_label, lstm_confidence = lstm_results[0]
            predictions.append(
                self._prediction(
                    "BETO + BiLSTM",
                    lstm_label,
                    lstm_confidence,
                    (time.perf_counter() - started) * 1000,
                )
            )

        for key, display_name in (
            ("random_forest", "Random Forest"),
            ("naive_bayes", "Naive Bayes"),
        ):
            started = time.perf_counter()
            probabilities = self._baselines[key].predict_proba([text])[0]
            label_id = int(self._baselines[key].classes_[probabilities.argmax()])
            confidence = float(probabilities.max())
            elapsed_ms = (time.perf_counter() - started) * 1000
            predictions.append(self._prediction(display_name, label_id, confidence, elapsed_ms))

        predictions.sort(key=lambda prediction: float(prediction["confidence"]), reverse=True)
        for rank, prediction in enumerate(predictions, start=1):
            prediction["rank"] = rank
        return predictions

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Run the same three-model comparison for every input text."""
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Every text must contain at least one non-whitespace character.")
        return [{"text": text, "predictions": self.predict(text)} for text in texts]

    @staticmethod
    def _prediction(
        model_name: str, label_id: int, confidence: float, latency_ms: float
    ) -> dict[str, str | int | float]:
        return {
            "model_name": model_name,
            "label": LABELS[int(label_id)],
            "label_id": int(label_id),
            "confidence": round(float(confidence), 6),
            "latency_ms": round(float(latency_ms), 3),
        }
