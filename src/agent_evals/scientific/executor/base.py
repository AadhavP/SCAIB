"""Backend-neutral contract for concrete scientific executors."""

from abc import ABC, abstractmethod

from agent_evals.environment.models import ActionExecutionResult, ActionIntent
from agent_evals.scientific.context import ScientificContext


class ScientificExecutor(ABC):
    """Execute typed action intents against a scientific context."""

    @abstractmethod
    def execute(self, action: ActionIntent, context: ScientificContext) -> ActionExecutionResult:
        """Execute one operation and return the stable framework result."""

