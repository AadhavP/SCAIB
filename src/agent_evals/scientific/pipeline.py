"""Declarative pipeline definitions for scientific benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PipelineStep(BaseModel):
    """One ordered scientific operation and its parameters."""

    model_config = ConfigDict(extra="forbid")

    operation: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class PipelineSpecification(BaseModel):
    """Validated pipeline document consumed by the scientific runner."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dataset: dict[str, Any]
    steps: list[PipelineStep] = Field(min_length=1)


def load_pipeline(path: Path | str) -> PipelineSpecification:
    """Load and validate a YAML pipeline document."""
    pipeline_path = Path(path)
    with pipeline_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline file '{pipeline_path}' must contain a mapping")
    return PipelineSpecification.model_validate(payload)

