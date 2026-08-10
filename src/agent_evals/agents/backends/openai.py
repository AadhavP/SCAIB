"""OpenAI Responses/chat tool-use runtime with injectable clients."""

from __future__ import annotations

import json
import os
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

DEFAULT_OPENAI_ACTION_PROMPT = (
    "You are controlling a scientific benchmark environment. Return exactly one "
    "structured action. Use a provider tool call when tools are supplied; otherwise "
    'return JSON matching {"action_type": "...", "parameters": {}, '
    '"reasoning_metadata": {"summary": "..."}}. Choose action_type from the '
    "available_actions in the latest observation. Do not include private reasoning."
)


class OpenAIRuntime(AgentRuntime):
    """Use an injected OpenAI-compatible client; no SDK is required at import time."""

    def __init__(
        self,
        *,
        model: str = "gpt-5",
        client: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.agent_id = "openai"
        self.client = client
        self.model = model
        self.tools = tools or []
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.system_prompt = system_prompt or DEFAULT_OPENAI_ACTION_PROMPT
        self.manifest = AgentManifest(
            name="OpenAI scientific agent",
            type="llm_tool_agent",
            model=AgentModelInfo(provider="openai", name=model),
            capabilities=["tool_use", "structured_actions"],
        )

    async def initialize(self, context: AgentContext) -> AgentSession:
        """Create a message history without contacting the provider."""
        return AgentSession(
            context=context,
            state={"messages": [{"role": "system", "content": self.system_prompt}]},
        )

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        """Request one structured action from Responses or Chat Completions."""
        if self.client is None:
            self.client = _build_openai_client(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
            )
        session.state.setdefault("messages", []).append({"role": "user", "content": observation.model_dump(mode="json")})
        response = await _call_openai(self.client, self.model, session.state["messages"], self.tools)
        action = _parse_provider_action(response)
        session.state["messages"].append({"role": "assistant", "content": action.model_dump_json()})
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
        instructions, input_messages = _split_instructions(messages)
        kwargs: dict[str, Any] = {"model": model, "input": input_messages, "tools": tools}
        if instructions:
            kwargs["instructions"] = instructions
        result = client.responses.create(**kwargs)
    elif hasattr(client, "chat"):
        result = client.chat.completions.create(model=model, messages=messages, tools=tools)
    else:
        raise TypeError("client must expose responses or chat.completions")
    return await result if hasattr(result, "__await__") else result


def _split_instructions(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    instruction_parts = [
        str(message.get("content"))
        for message in messages
        if message.get("role") == "system" and message.get("content") is not None
    ]
    input_messages = [message for message in messages if message.get("role") != "system"]
    return "\n\n".join(instruction_parts) or None, input_messages


def _build_openai_client(
    *,
    api_key: str | None,
    base_url: str | None,
    organization: str | None,
) -> Any:
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "OpenAI client is not configured; set OPENAI_API_KEY, pass api_key, "
            "or inject a client"
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "OpenAI SDK is not installed; install the provider extra or inject a client"
        ) from error
    kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    resolved_organization = organization or os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
    if resolved_organization:
        kwargs["organization"] = resolved_organization
    return OpenAI(**kwargs)


def _parse_provider_action(response: Any) -> AgentAction:
    """Parse structured JSON or a function/tool call from a provider response."""
    tool_call = _first_tool_call(response)
    if tool_call is not None:
        name = _read(tool_call, "name") or _read(tool_call, "function.name")
        arguments = _read(tool_call, "arguments") or _read(tool_call, "function.arguments") or {}
        return AgentAction(action_type=str(name), parameters=_json_object(arguments))
    text = (
        _read(response, "output_text")
        or _read(response, "output.0.content.0.text")
        or _read(response, "choices.0.message.content")
        or response
    )
    return AgentAction.model_validate(_json_object(text))


def _first_tool_call(response: Any) -> Any | None:
    output_item = _read(response, "output.0")
    if _read(output_item, "type") in {"function_call", "tool_call"}:
        return output_item
    content_item = _read(output_item, "content.0")
    if _read(content_item, "type") in {"function_call", "tool_call"}:
        return content_item
    return _read(response, "choices.0.message.tool_calls.0")


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
