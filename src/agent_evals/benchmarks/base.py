"""Abstract base class interface for benchmarks."""

from abc import ABC, abstractmethod
from typing import Any

from agent_evals.core.types import BenchmarkMetadata, EvaluationResult


class BaseBenchmark(ABC):
    """Abstract Base Class defining the benchmark execution contract."""

    def __init__(self, metadata: BenchmarkMetadata) -> None:
        self.metadata = metadata

    @abstractmethod
    async def prepare(self, config: dict[str, Any]) -> None:
        """Prepare dataset, workspace, and sandbox environment for benchmark run."""
        pass

    @abstractmethod
    async def evaluate_agent(
        self, agent_adapter: Any, sandbox: Any
    ) -> EvaluationResult:
        """Execute the evaluation protocol against the provided agent adapter inside sandbox.

        Args:
            agent_adapter: The agent adapter instance under evaluation.
            sandbox: The execution sandbox context.

        Returns:
            EvaluationResult model containing scores and status.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup transient execution resources post evaluation."""
        pass
