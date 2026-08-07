"""Abstract interface for isolated execution sandboxes."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """Result of running code inside the sandbox environment."""

    stdout: str
    stderr: str
    exit_code: int
    execution_time_seconds: float
    metadata: dict[str, Any] = {}


class BaseSandbox(ABC):
    """Abstract Base Class for isolated environment sandboxes."""

    def __init__(self, workdir: Path, timeout_seconds: int = 300) -> None:
        self.workdir = workdir
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def setup(self) -> None:
        """Initialize and start the sandbox environment (e.g. Docker container, venv)."""
        pass

    @abstractmethod
    async def execute_code(
        self, code: str, language: str = "python"
    ) -> ExecutionResult:
        """Execute snippet of code inside the sandbox.

        Args:
            code: Source code or shell script string to execute.
            language: Programming language ("python", "bash").

        Returns:
            ExecutionResult containing stdout, stderr, exit code, and time.
        """
        pass

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up and shutdown the sandbox environment."""
        pass
