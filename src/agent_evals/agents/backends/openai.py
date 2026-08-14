"""OpenAI Responses/chat tool-use runtime with injectable clients."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Mapping
from typing import Any

from agent_evals.agents.decisions import extract_action_response
from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentModelInfo,
    AgentObservation,
    AgentPlan,
    AgentSession,
    AgentUsage,
    FinalSubmission,
)

DEFAULT_OPENAI_PLAN_PROMPT = (
    "You are the planning scientist for a benchmark. Read the scenario goal, "
    "objective, observations, required artifacts, and stopping criteria. Return only "
    'JSON matching {"goal":"...", "steps":["..."], '
    '"success_criteria":["..."], "stopping_criteria":["..."], "adaptation_policy":"..."}. '
    "Make the plan concrete, evidence-driven, and revisable after every environment result. "
    "Do not invent an exhaustive failure-mode checklist; use observations to discover and correct problems."
)

DEFAULT_OPENAI_ACTION_PROMPT = (
    "You are an AI scientist operating an iterative, typed benchmark environment. "
    "Treat task_package as the authoritative experimental brief: inspect the data "
    "summary, method choices, parameter constraints, required inputs, expected outputs, "
    "workflow dependencies, and artifact contract before acting. After every result, "
    "review interaction.last_action and pipeline history, then keep or revise the plan "
    "based on observable evidence. Return exactly one structured action. Use a provider "
    "tool call when tools are supplied; otherwise return JSON matching "
    '{"action_type":"...","parameters":{},"reasoning_metadata":{'
    '"decision":{"intent":"...","hypothesis":"...","method":"...",'
    '"evidence_used":["..."],"alternatives_considered":["..."],'
    '"expected_effect":{"metric":0.0},"downstream_dependency":{"feeds":"..."},'
    '"confidence":0.0},"summary":"..."},"plan_update":null}. '
    "Choose action_type from the current available_actions, and supply method explicitly "
    "when the action contract declares one. Use finish/terminate only when the required "
    "artifact contract is satisfied. A failed action is feedback to diagnose, not a reason "
    "to hide the failure. Do not include private chain-of-thought."
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
        use_chat_completions: bool = False,
    ) -> None:
        self.agent_id = "openai"
        self.client = client
        self.model = model
        self.tools = tools or []
        self.api_key = api_key
        self.base_url = base_url
        self.organization = organization
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
        payload = _json_object(content)
        usage = _provider_usage(response)
        if usage is not None and "usage" not in payload:
            payload["usage"] = usage.model_dump(mode="json")
        return AgentPlan.model_validate(payload)

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        """Request one structured action from Responses or Chat Completions."""
        if self.client is None:
            self.client = _build_openai_client(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
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
    usage = _provider_usage(response)
    tool_call = _first_tool_call(response)
    if tool_call is not None:
        name = _read(tool_call, "name") or _read(tool_call, "function.name")
        arguments = _read(tool_call, "arguments") or _read(tool_call, "function.arguments") or {}
        payload, evidence = extract_action_response(
            {"action_type": str(name), "parameters": _json_object(arguments)},
        )
        if usage is not None:
            payload["usage"] = usage
        payload["extraction_evidence"] = evidence.model_copy(
            update={"source": "provider_tool_call"}
        )
        return AgentAction.model_validate(payload)
    text = (
        _read(response, "output_text")
        or _read(response, "output.0.content.0.text")
        or _read(response, "choices.0.message.content")
        or response
    )
    payload, evidence = extract_action_response(text)
    if usage is not None and "usage" not in payload:
        payload["usage"] = usage
    payload["extraction_evidence"] = evidence.model_copy(
        update={"source": "provider_text"}
    )
    return AgentAction.model_validate(payload)


def _first_tool_call(response: Any) -> Any | None:
    output_item = _read(response, "output.0")
    if _read(output_item, "type") in {"function_call", "tool_call"}:
        return output_item
    content_item = _read(output_item, "content.0")
    if _read(content_item, "type") in {"function_call", "tool_call"}:
        return content_item
    return _read(response, "choices.0.message.tool_calls.0")


def _provider_usage(response: Any) -> AgentUsage | None:
    """Translate provider usage fields into the boundary's public envelope."""
    raw = _read(response, "usage")
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    if not isinstance(raw, Mapping):
        return None
    usage: dict[str, Any] = {
        key: raw[key]
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if isinstance(raw.get(key), int) and raw[key] >= 0
    }
    if "input_tokens" not in usage and isinstance(raw.get("prompt_tokens"), int):
        usage["input_tokens"] = raw["prompt_tokens"]
    if "output_tokens" not in usage and isinstance(raw.get("completion_tokens"), int):
        usage["output_tokens"] = raw["completion_tokens"]
    if "total_tokens" not in usage and "input_tokens" in usage and "output_tokens" in usage:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return AgentUsage.model_validate({**usage, "source": "provider"}) if usage else None


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
