"""OpenAI Responses/chat tool-use runtime with injectable clients."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentModelInfo,
    AgentObservation,
    AgentSession,
    FinalSubmission,
)


class OpenAIRuntime(AgentRuntime):
    """Use an injected OpenAI-compatible client; no SDK is required at import time."""

    def __init__(
        self,
        *,
        model: str = "gpt-5",
        client: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.agent_id = "openai"
        self.client = client
        self.model = model
        self.tools = tools or []
        self.manifest = AgentManifest(
            name="OpenAI scientific agent",
            type="llm_tool_agent",
            model=AgentModelInfo(provider="openai", name=model),
            capabilities=["tool_use", "structured_actions"],
        )

    async def initialize(self, context: AgentContext) -> AgentSession:
        """Create a message history without contacting the provider."""
        return AgentSession(context=context, state={"messages": []})

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        """Request one structured action from Responses or Chat Completions."""
        if self.client is None:
            raise RuntimeError("OpenAI client is not configured; inject a client or install the provider SDK")
        session.state.setdefault("messages", []).append({"role": "user", "content": observation.model_dump(mode="json")})
        response = await _call_openai(self.client, self.model, session.state["messages"], self.tools)
        action = _parse_provider_action(response)
        session.state["messages"].append({"role": "assistant", "content": response})
        return action

    async def terminate(
        self,
        session: AgentSession,
        observation: AgentObservation | None = None,
    ) -> FinalSubmission:
        del session, observation
        return FinalSubmission()


async def _call_openai(client: Any, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
    """Call either Responses API or Chat Completions through a duck-typed client."""
    if hasattr(client, "responses"):
        result = client.responses.create(model=model, input=messages, tools=tools)
    elif hasattr(client, "chat"):
        result = client.chat.completions.create(model=model, messages=messages, tools=tools)
    else:
        raise TypeError("client must expose responses or chat.completions")
    return await result if hasattr(result, "__await__") else result


def _parse_provider_action(response: Any) -> AgentAction:
    """Parse structured JSON or a function/tool call from a provider response."""
    tool_call = _first_tool_call(response)
    if tool_call is not None:
        name = _read(tool_call, "name") or _read(tool_call, "function.name")
        arguments = _read(tool_call, "arguments") or _read(tool_call, "function.arguments") or {}
        return AgentAction(action_type=str(name), parameters=_json_object(arguments))
    text = _read(response, "output_text") or _read(response, "choices.0.message.content") or response
    return AgentAction.model_validate(_json_object(text))


def _first_tool_call(response: Any) -> Any | None:
    return _read(response, "output.0.content.0") or _read(response, "choices.0.message.tool_calls.0")


def _read(value: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("provider response must contain a JSON object")
        return parsed
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    raise ValueError("provider response did not contain structured action JSON")


__all__ = ["OpenAIRuntime"]
