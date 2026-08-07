"""Universal agent runtime layer."""

from agent_evals.agents.backends import (
    AnthropicRuntime,
    CustomPythonRuntime,
    ExternalProcessRuntime,
    OpenAICompatibleRuntime,
    OpenAIRuntime,
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
    AgentSession,
    FinalSubmission,
)
from agent_evals.agents.runtime.registry import (
    AgentRuntimeRegistry,
    agent_runtime_registry,
)


def _register_defaults() -> None:
    """Register provider-neutral names without importing provider SDKs."""
    if agent_runtime_registry.list():
        return
    agent_runtime_registry.register("openai", OpenAIRuntime, capabilities=["tool_use"])
    agent_runtime_registry.register("gpt-5", lambda **config: OpenAIRuntime(model="gpt-5", **config), capabilities=["tool_use"])
    agent_runtime_registry.register("anthropic", AnthropicRuntime, capabilities=["tool_use"])
    agent_runtime_registry.register(
        "claude-sonnet",
        lambda **config: AnthropicRuntime(model="claude-sonnet", **config),
        capabilities=["tool_use"],
    )
    agent_runtime_registry.register("openai-compatible", OpenAICompatibleRuntime, capabilities=["tool_use"])
    agent_runtime_registry.register("external-process", ExternalProcessRuntime, capabilities=["external_process"])
    agent_runtime_registry.register("custom", CustomPythonRuntime, capabilities=["custom_protocol"])


_register_defaults()

__all__ = [
    "AgentAction",
    "AgentContext",
    "AgentEvent",
    "AgentEventType",
    "AgentManifest",
    "AgentModelInfo",
    "AgentObservation",
    "AgentRuntime",
    "AgentRuntimeManager",
    "AgentRuntimeRegistry",
    "AgentSession",
    "AgentTrajectory",
    "AnthropicRuntime",
    "CustomPythonRuntime",
    "ExternalProcessRuntime",
    "FinalSubmission",
    "OpenAICompatibleRuntime",
    "OpenAIRuntime",
    "RuntimeAgentAdapter",
    "RuntimeRun",
    "agent_runtime_registry",
]
