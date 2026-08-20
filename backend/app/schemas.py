"""API request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single Spanish software requirement."""

    text: str = Field(min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    """Classification result for one requirement."""

    label: str
    label_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    execution_time_ms: float = Field(ge=0.0)


class BatchPredictionRequest(BaseModel):
    """Batch of requirements sharing one inference call."""

    texts: list[str] = Field(min_length=1, max_length=256)


class BatchPredictionResponse(BaseModel):
    """Batch classification response."""

    predictions: list[PredictionResponse]
    execution_time_ms: float = Field(ge=0.0)


class ModelPrediction(BaseModel):
    """Prediction produced by one benchmark model."""

    model_name: str
    label: str
    label_id: int = Field(ge=0, le=1)
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    rank: int = Field(ge=1, le=4)


class BenchmarkResponse(BaseModel):
    """Side-by-side predictions for all available models."""

    text: str
    predictions: list[ModelPrediction] = Field(min_length=3, max_length=4)


class BenchmarkBatchResponse(BaseModel):
    """Benchmark predictions for multiple requirements."""

    predictions: list[BenchmarkResponse] = Field(min_length=1, max_length=256)
