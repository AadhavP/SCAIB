"""Baseline execution contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BaselineResult(BaseModel):
    """Portable baseline output suitable for score comparison."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    baseline_id: str
    status: str
    actions: list[str] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaselineRunner(ABC):
    """Abstract deterministic baseline runner."""

    baseline_id: str

    @abstractmethod
    def run(self, context: dict[str, Any] | None = None, seed: int = 0) -> BaselineResult:
        """Execute a baseline under the supplied reproducibility context."""


__all__ = ["BaselineResult", "BaselineRunner"]
