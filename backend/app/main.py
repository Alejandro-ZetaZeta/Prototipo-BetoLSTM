"""FastAPI entrypoint for Spanish requirement classification."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import LABELS
from .model_loader import ActiveModelLoader
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BenchmarkBatchResponse,
    BenchmarkResponse,
    ModelPrediction,
    PredictionRequest,
    PredictionResponse,
)


def _model_dir() -> Path:
    configured = os.getenv("MODEL_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "models" / "beto_lstm_rf_rnf"


classifier = ActiveModelLoader(_model_dir())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load active model once when application starts."""
    classifier.load()
    yield


app = FastAPI(
    title="RF/RNF Requirements Classifier",
    description="Classifies Spanish software requirements as functional or non-functional.",
    version="1.0.0",
    lifespan=lifespan,
)


def allowed_origins() -> list[str]:
    """Read comma-separated browser origins from the environment."""
    configured = os.getenv("ALLOWED_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    local_origins = ["http://localhost:4321", "http://127.0.0.1:4321"]
    return list(dict.fromkeys(origins + local_origins))


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _response(label_id: int, confidence: float, elapsed_ms: float) -> PredictionResponse:
    return PredictionResponse(
        label=LABELS[label_id],
        label_id=label_id,
        confidence=round(float(confidence), 6),
        execution_time_ms=round(elapsed_ms, 3),
    )


def _model_prediction(data: dict[str, str | int | float]) -> ModelPrediction:
    return ModelPrediction(**data)


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report active model state."""
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Active model is not loaded")
    return {"status": "ok", "model_loaded": True, "beto_lstm_loaded": True}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Classify one requirement with active model."""
    started = time.perf_counter()
    try:
        results, _ = classifier.model.predict([request.text])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    label_id, confidence = results[0]
    return _response(int(label_id), float(confidence), (time.perf_counter() - started) * 1000)


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Classify up to 256 requirements with active model."""
    started = time.perf_counter()
    try:
        results, _ = classifier.model.predict(request.texts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    return BatchPredictionResponse(
        predictions=[_response(label_id, confidence, elapsed_ms / len(results)) for label_id, confidence in results],
        execution_time_ms=round(elapsed_ms, 3),
    )


def _comparison_response(request: PredictionRequest) -> BenchmarkResponse:
    started = time.perf_counter()
    try:
        prediction = classifier.predict(request.text)[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    prediction["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return BenchmarkResponse(text=request.text, predictions=[_model_prediction(prediction)])


@app.post("/api/predict/benchmark", response_model=BenchmarkResponse)
def predict_benchmark(request: PredictionRequest) -> BenchmarkResponse:
    """Compatibility route returning active model only."""
    return _comparison_response(request)


@app.post("/api/predict/benchmark/batch", response_model=BenchmarkBatchResponse)
def predict_benchmark_batch(request: BatchPredictionRequest) -> BenchmarkBatchResponse:
    """Compatibility route returning active model only."""
    try:
        predictions = classifier.predict_batch(request.texts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return BenchmarkBatchResponse(
        predictions=[
            BenchmarkResponse(
                text=item["text"],
                predictions=[_model_prediction(item["predictions"][0])],
            )
            for item in predictions
        ]
    )
