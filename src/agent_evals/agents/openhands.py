"""OpenHands SDK adapter and trace normalization."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast
from uuid import uuid4

from agent_evals.agents.harness import DefaultTraceNormalizer, build_agent_run
from agent_evals.agents.trajectory import (
    AgentConfiguration,
    AgentFailure,
    AgentRun,
    EstimatedCost,
    FailureKind,
    RawTraceEvent,
    RunTerminationStatus,
    TokenUsage,
)
from agent_evals.benchmarks.schema import TaskSpecification
from agent_evals.environment.models import EpisodeSnapshot, EpisodeStatus
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.environment.workspace import LocalWorkspace


class OpenHandsTraceNormalizer(DefaultTraceNormalizer):
    """Normalize OpenHands SDK event objects without importing SDK types."""

    _EVENT_ALIASES: ClassVar[dict[str, str]] = {
        "systempromptevent": "message",
        "messageevent": "message",
        "actionevent": "action",
        "observationevent": "observation",
        "agenterrorevent": "error",
        "conversationerrorevent": "error",
        "acptoolcallevent": "tool_call",
        "hookexecutionevent": "tool_result",
    }

    def coerce(self, events: Sequence[Any]) -> list[RawTraceEvent]:
        """Convert SDK event objects or dictionaries into raw events."""
        raw: list[RawTraceEvent] = []
        for sequence, event in enumerate(events):
            if isinstance(event, RawTraceEvent):
                raw.append(event)
                continue
            payload = _event_payload(event)
            kind = str(
                payload.get("event_type")
                or payload.get("type")
                or payload.get("kind")
                or type(event).__name__
                or "message"
            )
            normalized_kind = self._EVENT_ALIASES.get(kind.lower(), kind)
            timestamp = _event_timestamp(payload.get("timestamp"))
            raw.append(
                RawTraceEvent(
                    event_id=str(payload.get("id") or payload.get("event_id") or uuid4()),
                    source="openhands",
                    sequence=sequence,
                    timestamp=timestamp,
                    event_type=normalized_kind,
                    payload=payload,
                    parent_event_id=(
                        payload.get("parent_event_id") or payload.get("parent_id")
                    ),
                )
            )
        return raw


class OpenHandsAdapter:
    """Run a benchmark task through the OpenHands Python SDK.

    The SDK is imported lazily so the core harness remains usable without the
    optional OpenHands extra. ``session_factory`` remains available for tests
    and for controlled deployments that provide a custom Conversation object.
    """

    adapter_name = "openhands"
    adapter_version = "1.22.1"

    def __init__(
        self,
        *,
        session_factory: Callable[..., Any] | None = None,
        trace_normalizer: OpenHandsTraceNormalizer | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.trace_normalizer = trace_normalizer or OpenHandsTraceNormalizer()

    @property
    def available(self) -> bool:
        """Report whether the SDK import surface is installed."""
        return self.session_factory is not None or (
            importlib.util.find_spec("openhands.sdk") is not None
            and importlib.util.find_spec("openhands.tools") is not None
        )

    async def run(
        self,
        task: TaskSpecification,
        environment: ScientificEnvironment,
        configuration: AgentConfiguration,
    ) -> AgentRun:
        """Run one SDK conversation inside a controlled local workspace."""
        started_at = datetime.now(UTC)
        initial = await environment.reset(
            seed=configuration.seed,
            dataset_id=configuration.metadata.get("dataset_id")
            or (task.datasets[0] if task.datasets else None),
        )
        if not self.available:
            failure = AgentFailure(
                kind=FailureKind.ADAPTER_UNAVAILABLE,
                message=(
                    "OpenHands is unavailable; install the 'openhands' extra "
                    "or provide session_factory"
                ),
            )
            environment.terminate(status=EpisodeStatus.FAILED, reason=failure.message)
            final = environment.episode.snapshot() if environment.episode is not None else initial
            return build_agent_run(
                adapter_name=self.adapter_name,
                adapter_version=self.adapter_version,
                configuration=configuration,
                task=task,
                snapshot=final,
                raw_events=[],
                started_at=started_at,
                finished_at=datetime.now(UTC),
                termination_status=RunTerminationStatus.UNAVAILABLE,
                termination_reason=failure.message,
                failures=[failure],
                normalizer=self.trace_normalizer,
            )

        workspace = self._create_workspace(task, configuration)
        await workspace.initialize()
        session: Any = None
        raw_events: list[RawTraceEvent] = []
        failures: list[AgentFailure] = []
        status = RunTerminationStatus.COMPLETED
        reason = "OpenHands conversation complete"
        token_usage: TokenUsage | None = None
        estimated_cost: EstimatedCost | None = None
        metadata: dict[str, Any] = {
            "workspace": workspace.manifest.model_dump(mode="json"),
        }
        try:
            if self.session_factory is not None:
                session = self.session_factory(
                    task=task,
                    environment=environment,
                    configuration=configuration,
                    workspace=workspace.manifest,
                )
                session = await _resolve(session)
                result = await _run_in_session(
                    session,
                    prompt=_task_prompt(task, initial, workspace.manifest.root),
                    timeout_seconds=configuration.timeout_seconds,
                )
                candidate_events = (
                    result if isinstance(result, Sequence) and not isinstance(result, str) else None
                )
            else:
                session = self._create_sdk_session(task, configuration, workspace)
                await _run_in_session(
                    session,
                    prompt=_task_prompt(task, initial, workspace.manifest.root),
                    timeout_seconds=configuration.timeout_seconds,
                )
                candidate_events = None

            raw_events = self.trace_normalizer.coerce(
                list(candidate_events)
                if candidate_events is not None
                else _session_events(session)
            )
            token_usage, estimated_cost = _session_usage(session)
            conversation_id = getattr(session, "id", None)
            if conversation_id is not None:
                metadata["conversation_id"] = str(conversation_id)
            environment.terminate(status=EpisodeStatus.COMPLETED, reason=reason)
        except TimeoutError:
            status = RunTerminationStatus.TIMEOUT
            reason = "OpenHands conversation timed out"
            failures.append(AgentFailure(kind=FailureKind.TIMEOUT, message=reason))
            if environment.episode is not None:
                environment.terminate(status=EpisodeStatus.FAILED, reason=reason)
        except Exception as error:
            status = RunTerminationStatus.FAILED
            reason = str(error)
            failures.append(AgentFailure(kind=FailureKind.AGENT_ERROR, message=reason))
            raw_events = self.trace_normalizer.coerce(_session_events(session))
            if environment.episode is not None:
                environment.terminate(status=EpisodeStatus.FAILED, reason=reason)
        finally:
            await _close_session(session)
            await workspace.close()
            metadata["workspace"] = workspace.manifest.model_dump(mode="json")

        final = environment.episode.snapshot() if environment.episode is not None else initial
        return build_agent_run(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            configuration=configuration,
            task=task,
            snapshot=final,
            raw_events=raw_events,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            termination_status=status,
            termination_reason=reason,
            failures=failures,
            normalizer=self.trace_normalizer,
            token_usage=token_usage,
            estimated_cost=estimated_cost,
            metadata=metadata,
        )

    def _create_workspace(
        self,
        task: TaskSpecification,
        configuration: AgentConfiguration,
    ) -> LocalWorkspace:
        """Create or reuse the run workspace requested by the configuration."""
        configured_root = configuration.workspace.get("root")
        root = Path(configured_root) if configured_root else Path("runs") / "openhands" / (
            f"{task.id}-{uuid4().hex[:8]}"
        )
        return LocalWorkspace(root, workspace_id=f"openhands-{uuid4().hex}")

    def _create_sdk_session(
        self,
        task: TaskSpecification,
        configuration: AgentConfiguration,
        workspace: LocalWorkspace,
    ) -> Any:
        """Construct an OpenHands Conversation using the official SDK API."""
        sdk = importlib.import_module("openhands.sdk")
        terminal = importlib.import_module("openhands.tools.terminal")
        file_editor = importlib.import_module("openhands.tools.file_editor")
        task_tracker = importlib.import_module("openhands.tools.task_tracker")

        model = configuration.model or os.getenv("LLM_MODEL") or "anthropic/claude-sonnet-4-5-20250929"
        if configuration.provider and "/" not in model:
            model = f"{configuration.provider}/{model}"
        llm_kwargs: dict[str, Any] = {
            "model": model,
            "api_key": configuration.metadata.get("api_key") or os.getenv("LLM_API_KEY"),
            "base_url": configuration.metadata.get("base_url") or os.getenv("LLM_BASE_URL"),
        }
        if configuration.temperature is not None:
            llm_kwargs["temperature"] = configuration.temperature
        llm = sdk.LLM(**llm_kwargs)

        default_tools = [terminal.TerminalTool.name, file_editor.FileEditorTool.name, task_tracker.TaskTrackerTool.name]
        configured_tools = configuration.tools.get("names")
        tool_names = configured_tools if isinstance(configured_tools, list) else default_tools
        agent = sdk.Agent(
            llm=llm,
            tools=[sdk.Tool(name=str(tool_name)) for tool_name in tool_names],
        )
        persistence_dir = workspace.manifest.log_dir / "openhands"
        persistence_dir.mkdir(parents=True, exist_ok=True)
        return sdk.Conversation(
            agent=agent,
            workspace=workspace.manifest.root,
            persistence_dir=persistence_dir,
            max_iteration_per_run=configuration.max_steps or 500,
            delete_on_close=False,
        )


async def _run_in_session(session: Any, *, prompt: str, timeout_seconds: int | None) -> Any:
    """Send the benchmark prompt and execute a sync or async session."""
    async def execute() -> Any:
        send_message = getattr(session, "send_message", None)
        if send_message is not None:
            sent = send_message(prompt)
            await _resolve(sent)
        run = getattr(session, "run", None)
        if run is None:
            return session
        result = await asyncio.to_thread(run)
        return await _resolve(result)

    if timeout_seconds is None:
        return await execute()
    return await asyncio.wait_for(execute(), timeout=timeout_seconds)


async def _resolve(value: Any) -> Any:
    """Resolve an awaitable while keeping the adapter compatible with sync fakes."""
    return await value if inspect.isawaitable(value) else value


async def _close_session(session: Any) -> None:
    """Close SDK sessions without allowing cleanup to mask the run result."""
    if session is None:
        return
    close = getattr(session, "close", None)
    if close is None:
        return
    try:
        await _resolve(close())
    except Exception:
        return


def _session_events(session: Any) -> list[Any]:
    """Read the persisted OpenHands EventLog after a conversation run."""
    if session is None:
        return []
    state = getattr(session, "state", None)
    events = getattr(state, "events", None) if state is not None else None
    if events is None:
        events = getattr(session, "events", [])
    try:
        return list(events)
    except TypeError:
        return []


def _session_usage(session: Any) -> tuple[TokenUsage | None, EstimatedCost | None]:
    """Translate OpenHands conversation metrics into provider-neutral usage."""
    stats = getattr(session, "conversation_stats", None)
    if stats is None or not hasattr(stats, "get_combined_metrics"):
        return None, None
    metrics = stats.get_combined_metrics()
    payload = (
        metrics.model_dump(mode="json")
        if hasattr(metrics, "model_dump")
        else getattr(metrics, "__dict__", {})
    )
    token_data = payload.get("accumulated_token_usage") or {}
    input_tokens = _first_int(token_data, "input_tokens", "prompt_tokens")
    output_tokens = _first_int(token_data, "output_tokens", "completion_tokens")
    total_tokens = _first_int(token_data, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    usage = (
        TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        if any(value is not None for value in (input_tokens, output_tokens, total_tokens))
        else None
    )
    amount = _numeric_amount(payload.get("accumulated_cost"))
    cost = EstimatedCost(amount=amount, source="openhands") if amount is not None else None
    return usage, cost


def _first_int(payload: Any, *keys: str) -> int | None:
    """Read the first integer-like usage field from an SDK payload."""
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _numeric_amount(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


def _event_payload(event: Any) -> dict[str, Any]:
    """Serialize an SDK event while keeping the core package SDK-agnostic."""
    if isinstance(event, dict):
        payload = dict(event)
    elif hasattr(event, "model_dump"):
        payload = dict(event.model_dump(mode="json"))
    else:
        payload = dict(getattr(event, "__dict__", {}))
    return cast(dict[str, Any], _redact_private_content(payload))


def _redact_private_content(value: Any, *, key: str | None = None) -> Any:
    """Keep observable event metadata without persisting hidden reasoning."""
    sensitive = {
        "analysis",
        "chain_of_thought",
        "llm_message",
        "reasoning",
        "reasoning_content",
        "security_policy",
        "system_prompt",
        "thinking",
        "thinking_blocks",
    }
    if key in sensitive:
        return "[redacted]"
    if isinstance(value, dict):
        return {str(item_key): _redact_private_content(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_redact_private_content(item) for item in value]
    return value


def _event_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(UTC)


def _task_prompt(task: TaskSpecification, initial: EpisodeSnapshot, workspace_root: Path) -> str:
    """Build the explicit, observable task contract sent to OpenHands."""
    task_payload = task.model_dump(mode="json")
    initial_payload = initial.model_dump(mode="json")
    return (
        "You are executing one scientific benchmark task inside a controlled workspace.\n"
        f"Workspace: {workspace_root}\n"
        "Use the available OpenHands tools to inspect data, run analyses, and write requested artifacts.\n"
        "Do not request or expose private chain-of-thought. Report only concise observable actions, results, assumptions, and artifact paths.\n"
        "When the task is complete, leave reproducible outputs in the workspace and give a concise completion message.\n\n"
        f"Task specification:\n{task_payload}\n\n"
        f"Initial environment snapshot:\n{initial_payload}"
    )


__all__ = ["OpenHandsAdapter", "OpenHandsTraceNormalizer"]
