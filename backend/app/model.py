"""Local CPU BETO inference service."""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel

from .beto_lstm import BETOBiLSTM

LABELS = {0: "RF", 1: "RNF"}


class RequirementClassifier:
    """Load and execute trained BETO model without network access."""

    def __init__(self, model_dir: Path, max_length: int = 128) -> None:
        self.model_dir = model_dir
        self.max_length = max_length
        self._model: BertForSequenceClassification | None = None
        self._tokenizer = None
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether model artifacts are loaded in memory."""
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        """Load tokenizer and weights from local artifact directory."""
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing model files in {self.model_dir}: {', '.join(missing)}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self._model = BertForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self._model.to(torch.device("cpu"))
        self._model.eval()

    def predict(self, texts: list[str]) -> tuple[list[tuple[int, float]], float]:
        """Predict RF/RNF labels and return results plus elapsed milliseconds."""
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Every text must contain at least one non-whitespace character.")
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            assert self._tokenizer is not None and self._model is not None
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            confidence, labels = probabilities.max(dim=-1)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return list(zip(labels.tolist(), confidence.tolist())), elapsed_ms


def default_model_dir() -> Path:
    """Resolve model path, allowing deployment-time override."""
    configured = os.getenv("MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "models" / "beto_rf_rnf"


class BETOBiLSTMClassifier:
    """Load the separately trained BETO+BiLSTM artifact for local inference."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.max_length = 128
        self._model: BETOBiLSTM | None = None
        self._tokenizer = None
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Return whether LSTM artifact is ready."""
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        """Load encoder, tokenizer, and BiLSTM head from local files."""
        required = ("metadata.json", "head.pt", "base_encoder", "tokenizer.json")
        missing = [name for name in required if not (self.model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing BETO+BiLSTM files: {', '.join(missing)}")
        import json

        metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.max_length = int(metadata.get("max_length", 128))
        encoder = BertModel.from_pretrained(self.model_dir / "base_encoder", local_files_only=True)
        model = BETOBiLSTM(
            encoder,
            hidden_size=int(metadata.get("lstm_hidden_size", 256)),
            dropout=float(metadata.get("dropout", 0.3)),
        )
        state = torch.load(self.model_dir / "head.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model.to(torch.device("cpu"))
        model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self._model = model

    def predict(self, texts: list[str]) -> tuple[list[tuple[int, float]], float]:
        """Predict RF/RNF labels with the BETO+BiLSTM model."""
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Every text must contain at least one non-whitespace character.")
        if not self.is_loaded:
            raise RuntimeError("BETO+BiLSTM model is not loaded.")
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            assert self._tokenizer is not None and self._model is not None
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            confidence, labels = probabilities.max(dim=-1)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return list(zip(labels.tolist(), confidence.tolist())), elapsed_ms
