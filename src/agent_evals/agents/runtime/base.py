"""Abstract runtime contract implemented by native and external agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentObservation,
    AgentPlan,
    AgentSession,
    FinalSubmission,
)


class AgentRuntime(ABC):
    """Framework-neutral lifecycle for one agent session."""

    agent_id: str
    manifest: AgentManifest

    @abstractmethod
    async def initialize(self, context: AgentContext) -> AgentSession:
        """Create a clean session for one benchmark episode."""

    @abstractmethod
    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        """Return one observable action for the current observation."""

    async def plan(self, context: AgentContext, observation: AgentObservation) -> AgentPlan | None:
        """Optionally propose an observable high-level plan before acting."""
        del context, observation
        return None

    @abstractmethod
    async def terminate(
        self,
        session: AgentSession,
        observation: AgentObservation | None = None,
    ) -> FinalSubmission:
        """Return the final observable submission and release resources."""


__all__ = ["AgentRuntime"]
