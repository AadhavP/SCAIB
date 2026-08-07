"""Abstract base class interface for metric evaluators."""

from abc import ABC, abstractmethod
from typing import Any

from agent_evals.core.types import MetricScore


class BaseEvaluator(ABC):
    """Abstract Base Class for scoring agent outputs against ground truth data."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    async def evaluate(
        self, predicted_output: Any, ground_truth: Any
    ) -> list[MetricScore]:
        """Compute metrics comparing predicted agent output against ground truth.

        Args:
            predicted_output: Result produced by agent.
            ground_truth: Reference dataset or annotations.

        Returns:
            List of MetricScore items.
        """
        pass
