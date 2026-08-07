"""Core domain types, enums, and base Pydantic models."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StatusEnum(StrEnum):
    """Execution status for benchmarks, agents, and evaluations."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MetricScore(BaseModel):
    """Model representing a single calculated score."""

    name: str
    value: float
    unit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetadata(BaseModel):
    """Metadata describing a single-cell biology benchmark."""

    id: str
    name: str
    description: str
    version: str = "1.0.0"
    tags: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Complete summary result of an agent benchmark evaluation."""

    benchmark_id: str
    agent_id: str
    status: StatusEnum
    scores: list[MetricScore] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
