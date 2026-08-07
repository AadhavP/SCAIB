"""Sandboxed, replayable tool contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    """Public tool description supplied to an agent."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """Persisted call/result envelope for replay and auditing."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    call_id: str = Field(default_factory=lambda: str(uuid4()))
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


ToolHandler = Callable[[dict[str, Any], Any], Any | Awaitable[Any]]


__all__ = ["ToolCallRecord", "ToolDefinition", "ToolHandler"]
