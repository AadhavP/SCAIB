"""Legacy agents and framework-neutral harness adapters."""

from agent_evals.agents.base import AgentObservation, BaseAgentAdapter
from agent_evals.agents.baselines import RuleBasedSingleCellAgent
from agent_evals.agents.harness import (
    AgentAdapter,
    AgentHarness,
    DefaultTraceNormalizer,
    TraceNormalizer,
)
from agent_evals.agents.mock import (
    MockActionExecutor,
    MockAgent,
    MockAgentAdapter,
    MockObservationBuilder,
)
from agent_evals.agents.openhands import OpenHandsAdapter, OpenHandsTraceNormalizer
from agent_evals.agents.registry import (
    AgentAdapterRegistry,
    AgentRegistry,
    agent_adapter_registry,
    agent_registry,
)
from agent_evals.agents.runtime import (
    AgentAction,
    AgentContext,
    AgentEvent,
    AgentEventType,
    AgentRuntime,
    AgentRuntimeManager,
    AgentSession,
    AgentTrajectory,
    FinalSubmission,
    RuntimeAgentAdapter,
    RuntimeRun,
    agent_runtime_registry,
)
from agent_evals.agents.trajectory import *  # noqa: F403

agent_adapter_registry.register("mock")(MockAgentAdapter)
agent_adapter_registry.register("openhands")(OpenHandsAdapter)
agent_adapter_registry.register("rule-based")(RuleBasedSingleCellAgent)

__all__ = [
    "AgentAction",
    "AgentAdapter",
    "AgentAdapterRegistry",
    "AgentContext",
    "AgentEvent",
    "AgentEventType",
    "AgentHarness",
    "AgentObservation",
    "AgentRegistry",
    "AgentRuntime",
    "AgentRuntimeManager",
    "AgentSession",
    "AgentTrajectory",
    "BaseAgentAdapter",
    "DefaultTraceNormalizer",
    "FinalSubmission",
    "MockActionExecutor",
    "MockAgent",
    "MockAgentAdapter",
    "MockObservationBuilder",
    "OpenHandsAdapter",
    "OpenHandsTraceNormalizer",
    "RuleBasedSingleCellAgent",
    "RuntimeAgentAdapter",
    "RuntimeRun",
    "TraceNormalizer",
    "agent_adapter_registry",
    "agent_registry",
    "agent_runtime_registry",
]
