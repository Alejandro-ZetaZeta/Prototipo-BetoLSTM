"""FastAPI entrypoint for RF/RNF prediction."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import LABELS, RequirementClassifier, default_model_dir
from .model_loader import BenchmarkModelLoader
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BenchmarkBatchResponse,
    BenchmarkResponse,
    PredictionRequest,
    PredictionResponse,
)

classifier = RequirementClassifier(default_model_dir())
benchmark_loader = BenchmarkModelLoader(classifier, default_model_dir().parent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once when application starts."""
    classifier.load()
    benchmark_loader.load()
    yield


app = FastAPI(
    title="RF/RNF BETO Classifier",
    description="Classifies Spanish software requirements as functional or non-functional.",
    version="1.0.0",
    lifespan=lifespan,
)


def allowed_origins() -> list[str]:
    """Read comma-separated browser origins from the environment."""
    configured = os.getenv("ALLOWED_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return origins or ["http://localhost:4321", "http://127.0.0.1:4321"]


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


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report service and model state."""
    models_ready = benchmark_loader.is_loaded
    if not models_ready:
        raise HTTPException(status_code=503, detail="Models are not loaded")
    return {
        "status": "ok",
        "model_loaded": models_ready,
        "beto_lstm_loaded": benchmark_loader.lstm_loaded,
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Classify one requirement."""
    started = time.perf_counter()
    try:
        results, _ = classifier.predict([request.text])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    label_id, confidence = results[0]
    return _response(label_id, confidence, (time.perf_counter() - started) * 1000)


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Classify up to 256 requirements in one model call."""
    started = time.perf_counter()
    try:
        results, _ = classifier.predict(request.texts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    return BatchPredictionResponse(
        predictions=[_response(label_id, confidence, elapsed_ms / len(results)) for label_id, confidence in results],
        execution_time_ms=round(elapsed_ms, 3),
    )


@app.post("/api/predict/benchmark", response_model=BenchmarkResponse)
def predict_benchmark(request: PredictionRequest) -> BenchmarkResponse:
    """Run all locally available benchmark models on one requirement."""
    try:
        predictions = benchmark_loader.predict(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return BenchmarkResponse(text=request.text, predictions=predictions)


@app.post("/api/predict/benchmark/batch", response_model=BenchmarkBatchResponse)
def predict_benchmark_batch(request: BatchPredictionRequest) -> BenchmarkBatchResponse:
    """Run all three models for each requirement in a batch."""
    try:
        predictions = benchmark_loader.predict_batch(request.texts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return BenchmarkBatchResponse(predictions=predictions)
