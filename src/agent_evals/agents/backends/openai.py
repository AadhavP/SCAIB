"""OpenAI Responses/chat tool-use runtime with injectable clients."""

from __future__ import annotations

import asyncio
import inspect
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
    AgentPlan,
    AgentSession,
    FinalSubmission,
)

DEFAULT_OPENAI_PLAN_PROMPT = (
    "You are the planning scientist for a benchmark. Read the scenario goal, "
    "objective, observations, required artifacts, and success criteria. Return only "
    'JSON matching {"goal":"...", "steps":["..."], '
    '"success_criteria":["..."], "adaptation_policy":"..."}. '
    "Make the plan concrete, evidence-driven, and revisable after every environment result."
)

DEFAULT_OPENAI_ACTION_PROMPT = (
    "You are an agent solving a scientific benchmark, not merely selecting arbitrary tools. "
    "Use the scenario.goal, objective, success criteria, current observations, and pipeline "
    "history to make a defensible plan toward the benchmark goal. Return exactly one "
    "structured action. Use a provider tool call when tools are supplied; otherwise "
    'return JSON matching {"action_type": "...", "parameters": {}, '
    '"reasoning_metadata": {"summary": "..."}}. Choose an action_type from the '
    "available_actions, or return finish/terminate when the goal is satisfied. "
    "Do not repeat completed actions without a reason, and do not include private reasoning."
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
        default_headers: dict[str, str] | None = None,
        system_prompt: str | None = None,
        use_chat_completions: bool = False,
    ) -> None:
        self.agent_id = "openai"
        self.client = client
        self.model = model
        self.tools = tools or []
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
        self.default_headers = default_headers or {}
        self.system_prompt = system_prompt or DEFAULT_OPENAI_ACTION_PROMPT
        self.use_chat_completions = use_chat_completions
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

    async def plan(self, context: AgentContext, observation: AgentObservation) -> AgentPlan:
        """Ask the provider for an observable high-level plan before acting."""
        messages = [
            {"role": "system", "content": DEFAULT_OPENAI_PLAN_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "context": context.model_dump(mode="json"),
                        "observation": observation.model_dump(mode="json"),
                    }
                ),
            },
        ]
        if self.client is None:
            self.client = _build_openai_client(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                default_headers=self.default_headers,
            )
        response = await _call_openai(
            self.client,
            self.model,
            messages,
            [],
            prefer_chat_completions=self.use_chat_completions,
        )
        content = (
            _read(response, "output_text")
            or _read(response, "output.0.content.0.text")
            or _read(response, "choices.0.message.content")
            or response
        )
        return AgentPlan.model_validate(_json_object(content))

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        """Request one structured action from Responses or Chat Completions."""
        if self.client is None:
            self.client = _build_openai_client(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                default_headers=self.default_headers,
            )
        session.state.setdefault("messages", []).append({
            "role": "user",
            "content": json.dumps(observation.model_dump(mode="json")),
        })
        response = await _call_openai(
            self.client,
            self.model,
            session.state["messages"],
            self.tools,
            prefer_chat_completions=self.use_chat_completions,
        )
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


async def _call_openai(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    prefer_chat_completions: bool = False,
) -> Any:
    """Call either Responses API or Chat Completions through a duck-typed client."""
    if not prefer_chat_completions and hasattr(client, "responses"):
        instructions, input_messages = _split_instructions(messages)
        kwargs: dict[str, Any] = {"model": model, "input": input_messages, "tools": tools}
        if instructions:
            kwargs["instructions"] = instructions

        def call_responses() -> Any:
            return client.responses.create(**kwargs)

        result = await asyncio.to_thread(call_responses)
    elif hasattr(client, "chat"):
        def call_chat() -> Any:
            return client.chat.completions.create(model=model, messages=messages, tools=tools)

        result = await asyncio.to_thread(call_chat)
    else:
        raise TypeError("client must expose responses or chat.completions")
    return await result if inspect.isawaitable(result) else result


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
    default_headers: dict[str, str] | None = None,
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
    if default_headers:
        kwargs["default_headers"] = dict(default_headers)
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
        candidate = value.strip()
        if candidate.startswith("```"):
            candidate = candidate.removeprefix("```").removeprefix("json").removesuffix("```").strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("provider response must contain a JSON object") from None
            parsed = json.loads(candidate[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("provider response must contain a JSON object")
        return parsed
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    raise ValueError("provider response did not contain structured action JSON")


__all__ = ["OpenAIRuntime"]
