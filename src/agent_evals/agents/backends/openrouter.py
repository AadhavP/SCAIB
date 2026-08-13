"""OpenRouter runtime using the OpenAI-compatible chat-completions API."""

from __future__ import annotations

import os
from typing import Any

from agent_evals.agents.backends.credentials import MissingCredentialsError
from agent_evals.agents.backends.openai_compatible import OpenAICompatibleRuntime
from agent_evals.agents.runtime.protocol import AgentManifest, AgentModelInfo
from agent_evals.core.config import get_settings

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "z-ai/glm-5.2"


class OpenRouterRuntime(OpenAICompatibleRuntime):
    """Provider-named OpenRouter runtime without leaking credentials upstream."""

    def __init__(
        self,
        *,
        model: str | None = None,
        client: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        http_referer: str | None = None,
        app_title: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        settings = get_settings()
        resolved_model = (
            model
            or settings.openrouter_model
            or os.getenv("OPENROUTER_MODEL")
            or settings.llm_model
            or os.getenv("LLM_MODEL")
            or DEFAULT_OPENROUTER_MODEL
        )
        resolved_base_url = (
            base_url
            or settings.openrouter_base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or DEFAULT_OPENROUTER_BASE_URL
        )
        resolved_key = (
            api_key
            or settings.openrouter_api_key
            or os.getenv("OPENROUTER_API_KEY")
            or settings.llm_api_key
            or os.getenv("LLM_API_KEY")
        )
        if client is None and not resolved_key:
            raise MissingCredentialsError(
                "OpenRouter runtime requires OPENROUTER_API_KEY or LLM_API_KEY "
                "in the backend environment"
            )
        super().__init__(
            model=resolved_model,
            client=client,
            base_url=resolved_base_url,
            api_key=resolved_key,
            tools=tools,
            default_headers=_openrouter_headers(
                http_referer=http_referer,
                app_title=app_title,
            ),
            system_prompt=system_prompt,
        )
        self.agent_id = "openrouter"
        self.manifest = AgentManifest(
            name="OpenRouter scientific agent",
            type="llm_tool_agent",
            model=AgentModelInfo(provider="openrouter", name=resolved_model),
            capabilities=["tool_use", "structured_actions"],
        )


def _openrouter_headers(
    *,
    http_referer: str | None = None,
    app_title: str | None = None,
) -> dict[str, str]:
    settings = get_settings()
    referer = (
        http_referer
        or settings.openrouter_http_referer
        or os.getenv("OPENROUTER_HTTP_REFERER")
    )
    title = (
        app_title
        or settings.openrouter_app_title
        or os.getenv("OPENROUTER_APP_TITLE")
    )
    return {
        **({"HTTP-Referer": referer} if referer else {}),
        **({"X-OpenRouter-Title": title} if title else {}),
    }


__all__ = [
    "DEFAULT_OPENROUTER_BASE_URL",
    "DEFAULT_OPENROUTER_MODEL",
    "OpenRouterRuntime",
]
