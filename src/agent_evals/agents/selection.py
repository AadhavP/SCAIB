"""Turning an ``--agent`` name into something the benchmark can drive.

Three places used to answer this question and they had drifted. The CLI and the
scientific loop each carried a copy of the same ``{"gpt-5", "claude-sonnet"}``
guard -- a branch on provider identity in code that is supposed to know nothing
about providers -- and the loop additionally resolved one provider's credentials
inline. The guard existed only because two registry factories fixed their model
rather than defaulting it; :mod:`agent_evals.agents.backends.aliases` fixes that,
and this module is the single conversion the callers now share.

Selection order is deliberate: a universal runtime wins over a legacy adapter of
the same name, because the runtime tier is the one that records decisions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent_evals.agents.backends.aliases import build_runtime
from agent_evals.agents.backends.credentials import (
    compatible_runtime_from_environment,
)
from agent_evals.agents.registry import agent_adapter_registry
from agent_evals.agents.runtime.manager import RuntimeAgentAdapter
from agent_evals.agents.runtime.registry import agent_runtime_registry

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


def is_universal_runtime(agent_type: str) -> bool:
    """Report whether ``agent_type`` names a registered universal runtime."""
    return agent_type in agent_runtime_registry.list()


def build_agent_adapter(
    agent_type: str,
    *,
    model: str | None = None,
    agent_endpoint: str | None = None,
    event_callback: EventCallback | None = None,
    test_mode: bool = False,
) -> Any:
    """Build the adapter that will drive one episode.

    ``test_mode`` substitutes an OpenAI-compatible runtime configured from the
    backend environment, which is how the API server offers a hosted trial run
    without the caller holding a key. It overrides ``agent_type`` on purpose --
    the point is to exercise the universal path regardless of what was asked for
    -- and it raises rather than silently degrading when no key is present.
    """
    if test_mode:
        if agent_endpoint is not None:
            raise ValueError(
                "agent_endpoint cannot be combined with test_mode; test_mode selects "
                "the configured hosted runtime"
            )
        runtime = compatible_runtime_from_environment(model=model)
        return RuntimeAgentAdapter(runtime, event_callback=event_callback)
    if agent_endpoint is not None and agent_type != "http-step":
        raise ValueError(
            f"agent_endpoint is only valid for the 'http-step' runtime, not '{agent_type}'"
        )
    if is_universal_runtime(agent_type):
        return RuntimeAgentAdapter(
            build_runtime(agent_type, model=model, endpoint=agent_endpoint),
            event_callback=event_callback,
        )
    return agent_adapter_registry.create(agent_type)


__all__ = ["EventCallback", "build_agent_adapter", "is_universal_runtime"]
