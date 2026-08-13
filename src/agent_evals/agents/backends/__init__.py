"""Built-in provider/runtime adapters with lazy SDK boundaries.

This package is the only layer allowed to name a provider. :mod:`aliases` holds
which runtime names exist and what each defaults to; :mod:`credentials` holds
where an OpenAI-compatible endpoint's model, URL, and key come from. Both used to
live in the CLI and the scientific loop, which is what the spec's
zero-provider-branching-outside-adapters rule exists to prevent.
"""

from agent_evals.agents.backends.aliases import (
    DEFAULT_RUNTIMES,
    RuntimeRegistration,
    build_runtime,
    with_default_model,
)
from agent_evals.agents.backends.anthropic import AnthropicRuntime
from agent_evals.agents.backends.credentials import (
    CompatibleCredentials,
    MissingCredentialsError,
    compatible_runtime_from_environment,
    resolve_compatible_credentials,
)
from agent_evals.agents.backends.custom import CustomPythonRuntime
from agent_evals.agents.backends.external_process import ExternalProcessRuntime
from agent_evals.agents.backends.http_step import HttpStepError, HttpStepRuntime
from agent_evals.agents.backends.openai import OpenAIRuntime
from agent_evals.agents.backends.openai_compatible import OpenAICompatibleRuntime
from agent_evals.agents.backends.openrouter import OpenRouterRuntime

__all__ = [
    "DEFAULT_RUNTIMES",
    "AnthropicRuntime",
    "CompatibleCredentials",
    "CustomPythonRuntime",
    "ExternalProcessRuntime",
    "HttpStepError",
    "HttpStepRuntime",
    "MissingCredentialsError",
    "OpenAICompatibleRuntime",
    "OpenAIRuntime",
    "OpenRouterRuntime",
    "RuntimeRegistration",
    "build_runtime",
    "compatible_runtime_from_environment",
    "resolve_compatible_credentials",
    "with_default_model",
]
