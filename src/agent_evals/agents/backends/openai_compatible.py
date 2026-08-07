"""Generic OpenAI-compatible chat-completions runtime."""

from __future__ import annotations

from typing import Any

from agent_evals.agents.backends.openai import OpenAIRuntime
from agent_evals.agents.runtime.protocol import AgentManifest, AgentModelInfo


class OpenAICompatibleRuntime(OpenAIRuntime):
    """Support vLLM, LM Studio, Together, OpenRouter, and similar clients."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        base_url: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(model=model, client=client, tools=tools)
        self.agent_id = "openai-compatible"
        self.manifest = AgentManifest(
            name="OpenAI-compatible scientific agent",
            type="llm_tool_agent",
            model=AgentModelInfo(provider="openai-compatible", name=model),
            capabilities=["tool_use", "structured_actions"],
            metadata={"base_url": base_url},
        )
        self.base_url = base_url


__all__ = ["OpenAICompatibleRuntime"]
