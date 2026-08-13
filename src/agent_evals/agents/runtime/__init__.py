"""Universal agent runtime layer."""

from agent_evals.agents.backends import (
    DEFAULT_RUNTIMES,
    AnthropicRuntime,
    CustomPythonRuntime,
    ExternalProcessRuntime,
    HttpStepRuntime,
    OpenAICompatibleRuntime,
    OpenAIRuntime,
    OpenRouterRuntime,
)
from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.events import (
    AgentEvent,
    AgentEventType,
    AgentTrajectory,
)
from agent_evals.agents.runtime.manager import (
    AgentRuntimeManager,
    RuntimeAgentAdapter,
    RuntimeRun,
)
from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentModelInfo,
    AgentObservation,
    AgentPlan,
    AgentSession,
    FinalSubmission,
)
from agent_evals.agents.runtime.registry import (
    AgentRuntimeRegistry,
    agent_runtime_registry,
)


def _register_defaults() -> None:
    """Register the adapter layer's table without importing provider SDKs.

    The table itself lives in :mod:`agent_evals.agents.backends.aliases`, because
    an alias name and its default model are provider knowledge. This function
    reads it and learns nothing about who is behind each name.
    """
    if agent_runtime_registry.list():
        return
    for registration in DEFAULT_RUNTIMES:
        agent_runtime_registry.register(
            registration.name,
            registration.factory,
            capabilities=list(registration.capabilities),
        )


_register_defaults()

__all__ = [
    "AgentAction",
    "AgentContext",
    "AgentEvent",
    "AgentEventType",
    "AgentManifest",
    "AgentModelInfo",
    "AgentObservation",
    "AgentPlan",
    "AgentRuntime",
    "AgentRuntimeManager",
    "AgentRuntimeRegistry",
    "AgentSession",
    "AgentTrajectory",
    "AnthropicRuntime",
    "CustomPythonRuntime",
    "ExternalProcessRuntime",
    "FinalSubmission",
    "HttpStepRuntime",
    "OpenAICompatibleRuntime",
    "OpenAIRuntime",
    "OpenRouterRuntime",
    "RuntimeAgentAdapter",
    "RuntimeRun",
    "agent_runtime_registry",
]
