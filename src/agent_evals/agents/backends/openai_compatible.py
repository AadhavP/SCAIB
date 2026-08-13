"""Generic OpenAI-compatible chat-completions runtime."""

from __future__ import annotations

from typing import Any

from agent_evals.agents.backends.openai import OpenAIRuntime
from agent_evals.agents.runtime.protocol import AgentManifest, AgentModelInfo


class OpenAICompatibleRuntime(OpenAIRuntime):
    """Support GLM, vLLM, LM Studio, Together, OpenRouter, and similar clients."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        default_headers: dict[str, str] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            client=client,
            base_url=base_url,
            api_key=api_key,
            tools=tools,
            default_headers=default_headers,
            system_prompt=system_prompt,
            use_chat_completions=True,
        )
        self.agent_id = "openai-compatible"
        self.manifest = AgentManifest(
            name="OpenAI-compatible scientific agent",
            type="llm_tool_agent",
            model=AgentModelInfo(provider="openai-compatible", name=model),
            capabilities=["tool_use", "structured_actions"],
        )


__all__ = ["OpenAICompatibleRuntime"]
