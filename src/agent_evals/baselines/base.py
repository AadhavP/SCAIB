"""Baseline execution contracts."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaselineResult(BaseModel):
    """Portable baseline output suitable for score comparison."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    baseline_id: str
    status: str
    seed: int = 0
    actions: list[str] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0, le=1)
    implementation_digest: str | None = None
    environment_digest: str | None = None
    benchmark_digest: str | None = None
    configuration_digest: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "implementation_digest",
        "environment_digest",
        "benchmark_digest",
        "configuration_digest",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        """Reject mutable or malformed baseline identity claims."""
        if value is not None and re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError("baseline digests must be 64-character hexadecimal values")
        return value.lower() if value is not None else None

    @property
    def reproducible_contract_complete(self) -> bool:
        """Whether this result identifies the code and execution condition."""
        return all(
            digest is not None
            for digest in (
                self.implementation_digest,
                self.environment_digest,
                self.benchmark_digest,
                self.configuration_digest,
            )
        )


class BaselineRunner(ABC):
    """Abstract deterministic baseline runner."""

    baseline_id: str

    @abstractmethod
    def run(self, context: dict[str, Any] | None = None, seed: int = 0) -> BaselineResult:
        """Execute a baseline under the supplied reproducibility context."""


__all__ = ["BaselineResult", "BaselineRunner"]
