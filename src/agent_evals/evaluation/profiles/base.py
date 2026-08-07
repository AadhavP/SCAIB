"""Typed benchmark-specific metric profile contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MetricProfileEntry(BaseModel):
    """One frozen metric weight within a domain."""

    model_config = ConfigDict(extra="forbid")

    weight: float = Field(gt=0)
    required: bool = True


class MetricGroupProfile(BaseModel):
    """Weighted geometric domain definition."""

    model_config = ConfigDict(extra="forbid")

    weight: float = Field(gt=0)
    metrics: dict[str, MetricProfileEntry] = Field(default_factory=dict)
    external_score: str | None = None


class BenchmarkMetricProfile(BaseModel):
    """Complete frozen metric profile for one benchmark family."""

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    metric_groups: dict[str, MetricGroupProfile]
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_metric_profile(path: str | Path) -> BenchmarkMetricProfile:
    """Load and validate a YAML metric profile."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return BenchmarkMetricProfile.model_validate(payload)


__all__ = [
    "BenchmarkMetricProfile",
    "MetricGroupProfile",
    "MetricProfileEntry",
    "load_metric_profile",
]
