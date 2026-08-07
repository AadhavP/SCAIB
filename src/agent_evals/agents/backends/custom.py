"""Adapter for user-owned Python agents."""

from __future__ import annotations

import inspect
from typing import Any

from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentObservation,
    AgentSession,
    FinalSubmission,
)


class CustomPythonRuntime(AgentRuntime):
    """Wrap an object implementing initialize/act/terminate or observe/act."""

    def __init__(
        self,
        agent: Any,
        *,
        agent_id: str = "custom-python",
        manifest: AgentManifest | None = None,
    ) -> None:
        self.agent = agent
        self.agent_id = agent_id
        self.manifest = manifest or AgentManifest(
            name=agent_id,
            type="custom_python",
            capabilities=["custom_protocol"],
        )

    async def initialize(self, context: AgentContext) -> AgentSession:
        """Initialize the wrapped object when it provides a lifecycle hook."""
        state: dict[str, Any] = {}
        hook = getattr(self.agent, "initialize", None)
        if callable(hook):
            value = hook(context)
            state["native_session"] = await value if inspect.isawaitable(value) else value
        return AgentSession(context=context, state=state)

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        """Call the wrapped agent using the most specific supported method."""
        hook = getattr(self.agent, "act", None)
        if not callable(hook):
            hook = getattr(self.agent, "step", None)
        if not callable(hook):
            raise TypeError("custom agent must implement act() or step()")
        value = hook(observation, session) if _accepts_two_arguments(hook) else hook(observation)
        value = await value if inspect.isawaitable(value) else value
        return AgentAction.model_validate(value)

    async def terminate(
        self,
        session: AgentSession,
        observation: AgentObservation | None = None,
    ) -> FinalSubmission:
        """Call the optional finalization hook and normalize its output."""
        hook = getattr(self.agent, "terminate", None)
        if not callable(hook):
            return FinalSubmission()
        value = hook(session, observation) if _accepts_two_arguments(hook) else hook(session)
        value = await value if inspect.isawaitable(value) else value
        return FinalSubmission.model_validate(value)


def _accepts_two_arguments(callable_value: Any) -> bool:
    """Best-effort signature check for simple custom agent objects."""
    try:
        return len(inspect.signature(callable_value).parameters) >= 2
    except (TypeError, ValueError):
        return False


__all__ = ["CustomPythonRuntime"]
