"""Anthropic tool-use runtime with an injectable client."""

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
    AgentSession,
    AgentUsage,
    FinalSubmission,
)

DEFAULT_ANTHROPIC_ACTION_PROMPT = (
    "You are an AI scientist operating an iterative, typed benchmark environment. "
    "Use task_package as the experimental brief, including its end goal and stopping "
    "criteria, and follow its inputs, method choices, "
    "parameter constraints, workflow dependencies, and artifact contract. After every "
    "result, inspect interaction.last_action and pipeline history before choosing the next "
    "evidence-producing action. Return exactly one structured action. Use a provider tool "
    "call when tools are supplied; otherwise return JSON with action_type, parameters, "
    "reasoning_metadata, and optional plan_update. In reasoning_metadata include a public "
    "decision object with intent, hypothesis, method, evidence_used, alternatives_considered, "
    "expected_effect, downstream_dependency, and confidence. Choose action_type from the "
    "current available_actions and use finish/terminate only after the required artifacts "
    "validate. Failed actions are diagnostic feedback. Discover failure modes and confounders "
    "from the evidence instead of assuming a benchmark checklist. Never include private chain-of-thought."
)


class AnthropicRuntime(AgentRuntime):
    """Use the Anthropic messages API without making the SDK mandatory."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet",
        client: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.agent_id = "anthropic"
        self.client = client
        self.model = model
        self.tools = tools or []
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt or DEFAULT_ANTHROPIC_ACTION_PROMPT
        self.manifest = AgentManifest(
            name="Anthropic scientific agent",
            type="llm_tool_agent",
            model=AgentModelInfo(provider="anthropic", name=model),
            capabilities=["tool_use", "structured_actions"],
        )

    async def initialize(self, context: AgentContext) -> AgentSession:
        return AgentSession(context=context, state={"messages": []})

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        if self.client is None:
            self.client = _build_anthropic_client(api_key=self.api_key, base_url=self.base_url)
        session.state.setdefault("messages", []).append(
            {"role": "user", "content": json.dumps(observation.model_dump(mode="json"))}
        )
        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=2048,
            messages=session.state["messages"],
            system=self.system_prompt,
            tools=self.tools,
        )
        if inspect.isawaitable(response):
            response = await response
        usage = _provider_usage(response)
        block = _first_content_block(response)
        if block is not None and _read(block, "type") == "tool_use":
            payload, evidence = extract_action_response(
                {
                    "action_type": str(_read(block, "name")),
                    "parameters": _read(block, "input") or {},
                },
            )
            if usage is not None:
                payload["usage"] = usage
            payload["extraction_evidence"] = evidence.model_copy(
                update={"source": "provider_tool_call"}
            )
            return AgentAction.model_validate(payload)
        text = _read(block, "text") if block is not None else _read(response, "content.0.text")
        payload, evidence = extract_action_response(text)
        if usage is not None and "usage" not in payload:
            payload["usage"] = usage
        payload["extraction_evidence"] = evidence.model_copy(
            update={"source": "provider_text"}
        )
        return AgentAction.model_validate(payload)

    async def terminate(
        self,
        session: AgentSession,
        observation: AgentObservation | None = None,
    ) -> FinalSubmission:
        del session, observation
        return FinalSubmission()


def _build_anthropic_client(*, api_key: str | None, base_url: str | None) -> Any:
    resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Anthropic client is not configured; set ANTHROPIC_API_KEY, pass api_key, "
            "or inject a client"
        )
    try:
        from anthropic import Anthropic
    except ImportError as error:
        raise RuntimeError(
            "Anthropic SDK is not installed; install the provider extra or inject a client"
        ) from error
    kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    resolved_base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return Anthropic(**kwargs)


def _first_content_block(response: Any) -> Any | None:
    content = _read(response, "content")
    return content[0] if isinstance(content, list) and content else None


def _provider_usage(response: Any) -> AgentUsage | None:
    """Translate Anthropic per-message usage into the public envelope."""
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
    if "total_tokens" not in usage and "input_tokens" in usage and "output_tokens" in usage:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return AgentUsage.model_validate({**usage, "source": "provider"}) if usage else None


def _read(value: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            value = getattr(value, part, None)
        if value is None:
            return None
    return value


__all__ = ["AnthropicRuntime"]
