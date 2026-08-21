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
    """Prediction produced by active model."""

    model_name: str
    label: str
    label_id: int = Field(ge=0, le=1)
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    rank: int = Field(ge=1, le=1)


class BenchmarkResponse(BaseModel):
    """Prediction response for compatibility routes."""

    text: str
    predictions: list[ModelPrediction] = Field(min_length=1, max_length=1)


class BenchmarkBatchResponse(BaseModel):
    """Batch response for compatibility route."""

    predictions: list[BenchmarkResponse] = Field(min_length=1, max_length=256)
