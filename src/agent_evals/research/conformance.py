"""Dataset-independent conformance for the complete SCAIB episode loop.

This fixture is intentionally not a biological benchmark. It is a small contract
oracle used before real datasets are introduced: a URL agent must receive an
observation, return a plan and decisions, cause an evaluator-owned environment
to observe a verified artifact, terminate only after the deliverable exists, and
produce a tamper-evident public bundle.

Passing this suite proves engineering and protocol conformance. It must never be
reported as evidence that a scientific metric or biological task is valid.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents.backends.http_step import HttpStepRuntime
from agent_evals.agents.runtime.manager import AgentRuntimeManager, RuntimeVerdict
from agent_evals.agents.runtime.protocol import AgentContext
from agent_evals.benchmarks.schema import (
    ActionKind,
    ActionSpecification,
    ArtifactKind,
    ArtifactSpecification,
    BenchmarkMetadata,
    BenchmarkSpecification,
    ConstraintSpecification,
    EnvironmentBackend,
    EnvironmentSpecification,
    ObservationSpecification,
    ParameterSpecification,
    TaskSpecification,
    TerminationCondition,
)
from agent_evals.environment.execution import (
    IsolationRequest,
    LocalProcessBackend,
    WorkspaceActionExecutor,
    WorkspaceObservationBuilder,
)
from agent_evals.environment.models import ExecutionStatus, Observation
from agent_evals.environment.ports import CompositeObservationBuilder
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.research.bundle import (
    verify_run_bundle,
    write_event_ledger,
    write_run_bundle_manifest,
)
from agent_evals.scientific.artifacts.validation import ArtifactRuleValidator

CONFORMANCE_BENCHMARK_ID = "synthetic-conformance"
CONFORMANCE_BENCHMARK_VERSION = "1.0.0"


class SyntheticConformanceReport(BaseModel):
    """Machine-readable result of the dataset-independent conformance episode."""

    model_config = ConfigDict(extra="forbid")

    conformance_version: str = "1.0.0"
    passed: bool
    run_id: str
    benchmark_id: str = CONFORMANCE_BENCHMARK_ID
    benchmark_version: str = CONFORMANCE_BENCHMARK_VERSION
    bundle_path: str
    bundle_valid: bool
    endpoint_phases: list[str] = Field(default_factory=list)
    endpoint_request_ids: list[str] = Field(default_factory=list)
    action_count: int = Field(default=0, ge=0)
    artifact_verified: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=lambda: [
            "synthetic fixture only; no biological or dataset validity claim",
            "local synthetic execution cannot certify Linux sandbox isolation",
            "does not certify metric implementation correctness",
        ]
    )


class _SyntheticObservationBuilder:
    """Expose only evaluator-created synthetic state to the agent."""

    async def build(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: Any,
    ) -> list[Observation]:
        del specification
        return [
            Observation(
                observation_id="synthetic-state",
                value={
                    "step": snapshot.state.current_step,
                    "artifact_ids": sorted(snapshot.state.artifacts),
                    "objective": task.objective,
                },
                source="synthetic-environment",
                step=snapshot.state.current_step,
            )
        ]


class _SyntheticResponse:
    """Minimal response object accepted by the injected HTTP client path."""

    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.content = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.text = self.content.decode("utf-8")

    def json(self) -> dict[str, Any]:
        """Decode the response just like an httpx response."""
        return self._payload


class _SyntheticEndpoint:
    """Scripted URL endpoint that records only public exchange metadata."""

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self._observation_count = 0

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> Any:
        del url, headers
        self.posts.append(json)
        phase = str(json["type"])
        if phase == "initialize":
            response: dict[str, Any] = {"state": {"fixture": "ready"}}
        elif phase == "plan":
            response = {
                "plan": {
                    "goal": "Produce the verified synthetic result",
                    "steps": ["inspect", "produce", "finish"],
                    "success_criteria": ["result artifact is verified"],
                }
            }
        elif phase == "observation":
            self._observation_count += 1
            response = {
                "action_type": (
                    "produce" if self._observation_count == 1 else "finish"
                ),
                "parameters": (
                    {
                        "code": (
                            "from pathlib import Path\n"
                            "Path('result.json').write_text('{\\\"value\\\":1}\\n', encoding='utf-8')\n"
                        ),
                        "language": "python",
                        "produces": {"result": "result.json"},
                        "method": "deterministic_json_writer",
                    }
                    if self._observation_count == 1
                    else {}
                ),
            }
        elif phase == "terminate":
            response = {"summary": "synthetic conformance complete"}
        else:  # pragma: no cover - the runtime owns the phase vocabulary
            raise AssertionError(f"unexpected protocol phase: {phase}")
        request_id = json.get("request_id")
        response["protocol_version"] = "1.0"
        if request_id is not None:
            response["request_id"] = request_id
        self.responses.append(
            {
                "type": phase,
                "step": json.get("step"),
                "request_id": request_id,
                "request_sha256": _sha256_json(json),
                "response_keys": sorted(response),
                "response_sha256": _sha256_json(response),
            }
        )
        return _SyntheticResponse(response)


def build_synthetic_conformance_specification() -> BenchmarkSpecification:
    """Build the small no-dataset benchmark used by the conformance runner."""
    return BenchmarkSpecification(
        metadata=BenchmarkMetadata(
            id=CONFORMANCE_BENCHMARK_ID,
            title="SCAIB synthetic protocol conformance",
            description="Dataset-independent evaluator and endpoint contract fixture.",
            version=CONFORMANCE_BENCHMARK_VERSION,
            license="MIT",
            tags=["conformance", "synthetic", "research-readiness"],
        ),
        observations=[
            ObservationSpecification(
                id="synthetic-state",
                name="Synthetic state",
                description="Evaluator-owned toy state summary.",
                type="object",
                source="synthetic-environment",
            ),
            ObservationSpecification(
                id="workspace-tree",
                name="Workspace tree",
                description="Observed files in the agent workspace.",
                type="file_listing",
                source="environment",
            ),
            ObservationSpecification(
                id="execution-output",
                name="Execution output",
                description="Output from the latest workspace execution.",
                type="text",
                source="environment",
            ),
            ObservationSpecification(
                id="pipeline-history",
                name="Pipeline history",
                description="Observed history of workspace actions.",
                type="event_log",
                source="environment",
            ),
        ],
        actions=[
            ActionSpecification(
                id="inspect",
                name="Inspect synthetic state",
                purpose="Read the current toy state before acting.",
            ),
            ActionSpecification(
                id="produce",
                name="Produce verified result",
                purpose="Create the required synthetic result artifact in the agent workspace.",
                kind=ActionKind.FREE_EXECUTION,                    parameters=[
                        ParameterSpecification(
                            name="code",
                            description="Python source to execute.",
                            type="string",
                            required=True,
                        ),
                        ParameterSpecification(
                            name="language",
                            description="Interpreter for the source.",
                            type="string",
                            required=False,
                            default="python",
                            choices=["python"],
                        ),
                        ParameterSpecification(
                            name="produces",
                            description="Workspace-relative artifact paths.",
                            type="object",
                            required=False,
                        ),
                        ParameterSpecification(
                            name="method",
                            description="Observable method label.",
                            type="string",
                            required=False,
                        ),
                    ],
                required_inputs=["synthetic-state"],
                expected_outputs=[],
            ),
            ActionSpecification(
                id="finish",
                name="Finish episode",
                purpose="Declare that the required result is ready.",
            ),
        ],
        artifacts=[
            ArtifactSpecification(
                id="result",
                name="Synthetic result",
                description="Evaluator-verified toy result artifact.",
                kind=ArtifactKind.JSON,
                format="json",
                required=True,
                produced_by=["produce"],
            )
        ],
        tasks=[
            TaskSpecification(
                id="conformance-task",
                name="Protocol conformance task",
                objective="Produce a verified synthetic result through the public agent boundary.",
                description="Exercise the full observation, decision, execution, and termination loop.",
                observations=[
                    "synthetic-state",
                    "workspace-tree",
                    "execution-output",
                    "pipeline-history",
                ],
                allowed_actions=["produce", "finish"],                    artifacts=["result"],
                    environment="synthetic-local",
                    termination=[
                    TerminationCondition(
                        name="verified-result",
                        description="The result artifact has been independently verified.",
                        condition="result artifact validated",
                    )
                ],
            )
        ],
        environments=[
            EnvironmentSpecification(
                id="synthetic-local",
                name="Synthetic local workspace",
                description="Dataset-free Python workspace used only for contract conformance.",
                backend=EnvironmentBackend.LOCAL,
                languages=["python"],
            )
        ],
        constraints=ConstraintSpecification(
            internet_access=False,
            deterministic=True,
            random_seed=0,
            max_runtime_seconds=30,
        ),
    )


async def run_synthetic_conformance(
    output_dir: Path | str,
) -> SyntheticConformanceReport:
    """Run the complete URL-to-bundle contract without loading a dataset."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    specification = build_synthetic_conformance_specification()
    endpoint = _SyntheticEndpoint()
    workspace_root = root / "_synthetic_workspace"
    backend = LocalProcessBackend(
        workspace_root,
        isolation=IsolationRequest(network_access=False),
    )
    await backend.start()
    workspace_executor = WorkspaceActionExecutor(backend)
    environment = ScientificEnvironment(
        specification,
        task_id="conformance-task",
        executor=workspace_executor,
        observation_builder=CompositeObservationBuilder(
            _SyntheticObservationBuilder(),
            WorkspaceObservationBuilder(backend),
        ),
        artifact_validator=ArtifactRuleValidator(),
    )
    runtime = HttpStepRuntime(
        endpoint="https://synthetic-agent.example/step",
        client=endpoint,
        agent_id="synthetic-http-agent",
        strict_protocol=True,
    )
    context = AgentContext(
        benchmark_id=CONFORMANCE_BENCHMARK_ID,
        task_id="conformance-task",
        workspace=str(root),
    )
    try:
        result = await AgentRuntimeManager().run(runtime, environment, context, seed=0)
    finally:
        await backend.close()

    bundle_root = root / result.run_id
    bundle_root.mkdir(parents=True, exist_ok=True)
    source_result = root / "_synthetic_workspace" / "result.json"
    if source_result.is_file():
        (bundle_root / "result.json").write_bytes(source_result.read_bytes())
    trajectory = result.trajectory.model_dump(mode="json")
    (bundle_root / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exchange = {
        "protocol_version": "1.0",
        "endpoint": "https://synthetic-agent.example/step",
        "requests": endpoint.responses,
        "request_count": len(endpoint.posts),
    }
    (bundle_root / "endpoint_exchange.json").write_text(
        json.dumps(exchange, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_verified = (
        bool(result.final_snapshot.state.artifacts)
        and all(
            artifact.validated
            and artifact.validation is not None
            and artifact.validation.is_valid
            for artifact in result.final_snapshot.state.artifacts.values()
        )
    )
    result_artifact = result.final_snapshot.state.artifacts.get("result")
    artifact_bytes_match = bool(
        result_artifact is not None
        and result_artifact.checksum is not None
        and (bundle_root / "result.json").is_file()
        and result_artifact.checksum
        == "sha256:" + hashlib.sha256((bundle_root / "result.json").read_bytes()).hexdigest()
    )
    report_payload = {
        "run_id": result.run_id,
        "termination_status": result.termination_status.value,
        "termination_reason": result.termination_reason,
        "step_count": result.step_count,
        "artifact_verified": artifact_verified,
        "artifact_bytes_match": artifact_bytes_match,
        "endpoint_phases": [item["type"] for item in endpoint.responses],
        "limitations": [
            "synthetic fixture only; no biological or dataset validity claim",
            "local synthetic execution cannot certify Linux sandbox isolation",
            "does not certify metric implementation correctness",
        ],
    }
    (bundle_root / "report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger_events = [
        {
            "source": "agent-boundary",
            "event_type": "endpoint_exchange",
            "payload": exchange,
        },
        *[
            {
                "source": "runtime",
                "event_type": event.event_type.value,
                "payload": event.model_dump(mode="json"),
            }
            for event in result.trajectory.events
        ],
    ]
    ledger_digest = write_event_ledger(bundle_root, ledger_events)
    (bundle_root / "replay.json").write_text(
        json.dumps(
            {
                "replay_version": "1.0.0",
                "run_id": result.run_id,
                "benchmark_id": CONFORMANCE_BENCHMARK_ID,
                "task_id": "conformance-task",
                "seed": 0,
                "deterministic": True,
                "replay_mode": "event_sourced_conformance",
                "event_ledger": "events.ndjson",
                "event_ledger_sha256": ledger_digest,
                "trajectory": "trajectory.json",
                "report": "report.json",
                "endpoint_exchange": "endpoint_exchange.json",
                "limitations": [
                    "descriptor validates replay inputs only; it does not execute agent code",
                    "synthetic fixture only; no biological or dataset validity claim",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_run_bundle_manifest(bundle_root, run_id=result.run_id)
    verification = verify_run_bundle(bundle_root)
    phases = [str(item["type"]) for item in endpoint.responses]
    expected_phases = ["initialize", "plan", "observation", "observation", "terminate"]
    checks = {
        "endpoint_back_and_forth": phases == expected_phases,
        "request_ids_present": bool(endpoint.posts)
        and all(bool(item.get("request_id")) for item in endpoint.posts),
        "protocol_version_present": all(
            item.get("protocol_version") == "1.0" for item in endpoint.posts
        ),
        "exchange_requests_and_responses_hashed": all(
            item.get("request_sha256") and item.get("response_sha256")
            for item in endpoint.responses
        ),
        "controller_completed_after_verified_artifact": (
            result.termination_status is RuntimeVerdict.COMPLETED
            and artifact_verified
            and artifact_bytes_match
        ),
        "agent_workspace_execution_was_observed": any(
            action.intent.action_id == "produce"
            and action.result.execution_status is ExecutionStatus.SUCCESS
            and action.result.observed_state_delta is not None
            and "result.json" in action.result.observed_state_delta.files.added
            for action in result.final_snapshot.state.actions
        ),
        "agent_visible_state_contains_no_scores": bool(result.trajectory.observations)
        and all(
            _agent_visible_state_is_clean(observation.model_dump(mode="json"))
            for observation in result.trajectory.observations
        ),
        "bundle_hash_chain_and_manifest_verify": verification.valid
        and verification.event_chain_valid,
        "replay_descriptor_is_ready": verification.replay_ready,
    }
    findings = [name for name, passed in checks.items() if not passed]
    (bundle_root / "conformance.json").write_text(
        json.dumps(
            {
                "conformance_version": "1.0.0",
                "benchmark_id": CONFORMANCE_BENCHMARK_ID,
                "benchmark_version": CONFORMANCE_BENCHMARK_VERSION,
                "run_id": result.run_id,
                "checks": checks,
                "findings": findings,
                "limitations": [
                    "synthetic fixture only; no biological or dataset validity claim",
                    "local synthetic execution cannot certify Linux sandbox isolation",
                    "does not certify metric implementation correctness",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_run_bundle_manifest(bundle_root, run_id=result.run_id)
    verification = verify_run_bundle(bundle_root)
    return SyntheticConformanceReport(
        passed=not findings and verification.valid,
        run_id=result.run_id,
        bundle_path=str(bundle_root),
        bundle_valid=verification.valid,
        endpoint_phases=phases,
        endpoint_request_ids=[
            str(item.get("request_id")) for item in endpoint.posts if item.get("request_id")
        ],
        action_count=result.step_count,
        artifact_verified=artifact_verified and artifact_bytes_match,
        checks=checks,
        findings=findings,
    )


def run_synthetic_conformance_sync(
    output_dir: Path | str,
) -> SyntheticConformanceReport:
    """Synchronous convenience wrapper for CLI and release checks."""
    return asyncio.run(run_synthetic_conformance(output_dir))


def _agent_visible_state_is_clean(observation: dict[str, Any]) -> bool:
    """Check evaluator-owned score channels are absent without banning field names."""
    state = observation.get("state")
    if not isinstance(state, dict):
        return False
    # ``rewards`` is a legitimate empty structural field in the public state;
    # what must be absent is any evaluator-produced value or reference payload.
    public_state = json.dumps(state, sort_keys=True, default=str).lower()
    return state.get("rewards") == [] and "held_out" not in public_state and "reference_labels" not in public_state


def _sha256_json(value: Any) -> str:
    """Hash a JSON-compatible public exchange record deterministically."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CONFORMANCE_BENCHMARK_ID",
    "CONFORMANCE_BENCHMARK_VERSION",
    "SyntheticConformanceReport",
    "build_synthetic_conformance_specification",
    "run_synthetic_conformance",
    "run_synthetic_conformance_sync",
]
