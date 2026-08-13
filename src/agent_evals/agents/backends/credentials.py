"""Where an OpenAI-compatible runtime's model, endpoint, and key come from.

This resolution used to sit inside ``scientific_loop._create_scientific_adapter``,
which meant the orchestration layer knew the names of four providers' environment
variables and one provider's default endpoint. It resolves nothing the benchmark
cares about, so it belongs beside the adapter it configures.

Every value is read from :class:`Settings` or the process environment. Nothing is
hardcoded except the model name and endpoint, which are defaults rather than
secrets, and a missing key raises rather than falling through to an unauthenticated
call that would fail further away from its cause.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agent_evals.agents.backends.openai_compatible import OpenAICompatibleRuntime
from agent_evals.core.config import get_settings

#: Used when neither settings nor the environment names a model.
DEFAULT_COMPATIBLE_MODEL = "z-ai/glm-5.2"

#: Used when neither settings nor the environment names an endpoint.
DEFAULT_COMPATIBLE_BASE_URL = "https://openrouter.ai/api/v1"

#: Environment variables consulted for the API key, in order. Named here so the
#: error message and the lookup cannot drift apart.
API_KEY_VARIABLES: tuple[str, ...] = (
    "LLM_API_KEY",
    "GLM_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
)


class MissingCredentialsError(RuntimeError):
    """Raised when no API key is available for an OpenAI-compatible runtime."""


@dataclass(frozen=True)
class CompatibleCredentials:
    """A resolved endpoint configuration, with the key deliberately unprinted."""

    model: str
    base_url: str
    api_key: str

    def __repr__(self) -> str:
        """Redact the key so a traceback or log line cannot leak it."""
        return (
            f"CompatibleCredentials(model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key=<redacted>)"
        )


def _first(*candidates: str | None) -> str | None:
    return next((value for value in candidates if value), None)


def resolve_compatible_credentials(
    *, model: str | None = None
) -> CompatibleCredentials:
    """Resolve model, endpoint, and key for an OpenAI-compatible provider.

    Settings are consulted before ``os.getenv`` because pydantic-settings parses
    ``.env``, which ``os.getenv`` cannot see; the ``os.getenv`` tier still matters
    for variables exported after the settings object was built. ``model`` ranks
    below both on purpose -- it preserves the precedence this resolution had
    inline, where an explicitly configured backend model outranked a caller's
    per-run suggestion.
    """
    settings = get_settings()
    resolved_model = (
        _first(
            settings.llm_model,
            settings.glm_model,
            os.getenv("LLM_MODEL"),
            os.getenv("GLM_MODEL"),
            model,
        )
        or DEFAULT_COMPATIBLE_MODEL
    )
    resolved_base_url = (
        _first(
            settings.llm_base_url,
            settings.glm_base_url,
            settings.openrouter_base_url,
            os.getenv("LLM_BASE_URL"),
            os.getenv("GLM_BASE_URL"),
            os.getenv("OPENROUTER_BASE_URL"),
        )
        or DEFAULT_COMPATIBLE_BASE_URL
    )
    resolved_key = _first(
        settings.llm_api_key,
        settings.glm_api_key,
        settings.openrouter_api_key,
        settings.openai_api_key,
        *(os.getenv(name) for name in API_KEY_VARIABLES),
    )
    if not resolved_key:
        raise MissingCredentialsError(
            "an OpenAI-compatible runtime requires one of "
            f"{', '.join(API_KEY_VARIABLES)} in the backend environment"
        )
    return CompatibleCredentials(
        model=resolved_model, base_url=resolved_base_url, api_key=resolved_key
    )


def compatible_runtime_from_environment(
    *, model: str | None = None
) -> OpenAICompatibleRuntime:
    """Build an OpenAI-compatible runtime from resolved credentials."""
    credentials = resolve_compatible_credentials(model=model)
    return OpenAICompatibleRuntime(
        model=credentials.model,
        base_url=credentials.base_url,
        api_key=credentials.api_key,
    )


__all__ = [
    "API_KEY_VARIABLES",
    "DEFAULT_COMPATIBLE_BASE_URL",
    "DEFAULT_COMPATIBLE_MODEL",
    "CompatibleCredentials",
    "MissingCredentialsError",
    "compatible_runtime_from_environment",
    "resolve_compatible_credentials",
]
