"""Route each intent to the executor its action kind calls for.

A benchmark may offer typed actions, free execution, or both, and the choice is
per-action rather than per-benchmark. ``ScientificEnvironment`` holds exactly one
``ActionExecutor``, so something has to stand in that slot and decide. This is
that something, and being an ``ActionExecutor`` itself is the whole point: the
environment cannot tell it apart from a single backend, so nothing above the port
learns that two execution models now coexist.

Mixing the kinds in one benchmark is the interesting case, not an edge case. It
lets a task hand the agent a reliable typed loader and still require it to write
its own clustering, which is how a benchmark isolates the step it means to
measure from the boilerplate it does not.

The one refusal here is at construction: a benchmark that declares free-execution
actions with no workspace executor to run them is a wiring mistake, and it raises
rather than failing each action in turn. A per-action failure would read as the
agent's code not working, which is exactly the wrong diagnosis to record.
"""

from __future__ import annotations

from collections.abc import Iterable

from agent_evals.benchmarks.schema import ActionKind, BenchmarkSpecification
from agent_evals.environment.models import ActionExecutionResult, ActionIntent
from agent_evals.environment.ports import ActionExecutor, ExecutionContext


def free_execution_action_ids(specification: BenchmarkSpecification) -> frozenset[str]:
    """Return the ids of actions the agent implements itself."""
    return frozenset(
        action.id
        for action in specification.actions
        if action.kind is ActionKind.FREE_EXECUTION
    )


class ActionKindRouter:
    """Dispatch typed intents and free-execution intents to their executors."""

    def __init__(
        self,
        *,
        typed: ActionExecutor,
        free: ActionExecutor | None = None,
        free_execution_ids: Iterable[str] = (),
    ) -> None:
        self.typed = typed
        self.free = free
        self.free_execution_ids = frozenset(free_execution_ids)
        if self.free_execution_ids and free is None:
            declared = ", ".join(sorted(self.free_execution_ids))
            raise ValueError(
                "free-execution action(s) were declared with no workspace "
                f"executor to run them: {declared}. Provision an environment for "
                "the task, or the agent would be failed for the benchmark's "
                "missing configuration."
            )

    @classmethod
    def from_specification(
        cls,
        specification: BenchmarkSpecification,
        *,
        typed: ActionExecutor,
        free: ActionExecutor | None = None,
    ) -> ActionKindRouter:
        """Build a router from the benchmark's own declaration of action kinds."""
        return cls(
            typed=typed,
            free=free,
            free_execution_ids=free_execution_action_ids(specification),
        )

    def executor_for(self, action_id: str) -> ActionExecutor:
        """Return the executor that owns an action.

        Anything not declared free-execution goes to the typed executor,
        including an unknown id: the environment's validator has already
        rejected undeclared actions, and duplicating that rejection here would
        put two different error messages on one failure.
        """
        if action_id in self.free_execution_ids and self.free is not None:
            return self.free
        return self.typed

    async def execute(
        self,
        intent: ActionIntent,
        context: ExecutionContext,
    ) -> ActionExecutionResult:
        """Execute one intent through whichever executor its action kind names."""
        return await self.executor_for(intent.action_id).execute(intent, context)


__all__ = ["ActionKindRouter", "free_execution_action_ids"]
