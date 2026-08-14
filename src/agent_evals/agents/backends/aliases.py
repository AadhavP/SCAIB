"""Which runtime names exist, and what each one defaults to.

The benchmark's rule is that nothing above the adapter layer branches on which
provider is answering. Two things broke that rule, and both had one cause. The
``gpt-5`` and ``claude-sonnet`` factories were registered as
``lambda **config: OpenAIRuntime(model="gpt-5", **config)``, so a caller passing
``model=`` hit a duplicate-keyword ``TypeError`` -- and rather than fix the
factory, two call sites grew the same ``{"gpt-5", "claude-sonnet"}`` guard to
avoid passing it. Defaulting instead of fixing the model removes the reason for
the guard rather than the guard itself.

The registration table lives here for the same reason: an alias name *is*
provider knowledge, so the layer that knows about providers is the layer that
should hold it. :mod:`agent_evals.agents.runtime` reads the table and registers
it, and learns nothing about who is behind each name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from agent_evals.agents.backends.anthropic import AnthropicRuntime
from agent_evals.agents.backends.custom import CustomPythonRuntime
from agent_evals.agents.backends.external_process import ExternalProcessRuntime
from agent_evals.agents.backends.http_step import HttpStepRuntime
from agent_evals.agents.backends.openai import OpenAIRuntime
from agent_evals.agents.backends.openai_compatible import OpenAICompatibleRuntime
from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.registry import agent_runtime_registry

RuntimeFactory = Callable[..., AgentRuntime]


def with_default_model(factory: RuntimeFactory, model: str) -> RuntimeFactory:
    """Bind a default model to ``factory`` while leaving it overridable.

    ``setdefault`` rather than a fixed keyword: an alias says which model the
    name means by default, not which model the name is allowed to run. Fixing it
    is what made ``model=`` unpassable and pushed the workaround upstream.
    """

    def build(**config: Any) -> AgentRuntime:
        config.setdefault("model", model)
        return factory(**config)

    return build


@dataclass(frozen=True)
class RuntimeRegistration:
    """One entry in the default runtime table."""

    name: str
    factory: RuntimeFactory
    capabilities: Sequence[str]


#: Runtimes registered at import time. Ordered for readability only; the
#: registry sorts its own listing.
DEFAULT_RUNTIMES: tuple[RuntimeRegistration, ...] = (
    RuntimeRegistration("openai", OpenAIRuntime, ("tool_use",)),
    RuntimeRegistration(
        "gpt-5", with_default_model(OpenAIRuntime, "gpt-5"), ("tool_use",)
    ),
    RuntimeRegistration("anthropic", AnthropicRuntime, ("tool_use",)),
    RuntimeRegistration(
        "claude-sonnet",
        with_default_model(AnthropicRuntime, "claude-sonnet"),
        ("tool_use",),
    ),
    RuntimeRegistration("openai-compatible", OpenAICompatibleRuntime, ("tool_use",)),
    RuntimeRegistration(
        "external-process", ExternalProcessRuntime, ("external_process",)
    ),
    RuntimeRegistration("custom", CustomPythonRuntime, ("custom_protocol",)),
    RuntimeRegistration("http-step", HttpStepRuntime, ("http_step",)),
)


def build_runtime(
    name: str,
    *,
    model: str | None = None,
    endpoint: str | None = None,
    timeout_seconds: float | None = None,
) -> AgentRuntime:
    """Instantiate a runtime with provider-neutral overrides.

    ``endpoint`` is intentionally an adapter-level option rather than a
    benchmark concern. It makes the black-box ``http-step`` tier selectable from
    a URL while keeping provider-specific construction inside the runtime
    registry. Other runtimes reject it instead of silently ignoring the URL.
    """
    config: dict[str, Any] = {}
    if model is not None:
        config["model"] = model
    if endpoint is not None:
        if name != "http-step":
            raise ValueError(
                f"agent endpoint configuration is only supported by 'http-step', not '{name}'"
            )
        config["endpoint"] = endpoint
    if timeout_seconds is not None:
        config["timeout_seconds"] = timeout_seconds
    try:
        return agent_runtime_registry.create(name, **config)
    except TypeError as error:
        option = "endpoint" if endpoint is not None else "model"
        raise ValueError(
            f"agent runtime '{name}' does not accept the {option} override ({error})"
        ) from error


__all__ = [
    "DEFAULT_RUNTIMES",
    "RuntimeFactory",
    "RuntimeRegistration",
    "build_runtime",
    "with_default_model",
]
