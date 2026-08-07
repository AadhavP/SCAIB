"""Environment runtime context state management."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EnvironmentContext(BaseModel):
    """Holds runtime environment context and variables for evaluation."""

    session_id: str
    workdir: Path
    env_vars: dict[str, str] = Field(default_factory=dict)
    active_dataset_path: Path | None = None
    step_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def increment_step(self) -> int:
        """Increment execution step counter."""
        self.step_count += 1
        return self.step_count
