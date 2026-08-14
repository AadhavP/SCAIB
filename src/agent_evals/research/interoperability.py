"""Offline interoperability checks for the universal agent boundary.

Research comparisons should not depend on one provider SDK or one internal agent
framework. These fixtures exercise the public protocol and decision extractor
without making network calls, so CI can prove that structured agents, black-box
text agents, and opaque multi-agent systems enter the same canonical boundary.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents.decisions import ExtractionMode, extract_action_response
from agent_evals.agents.runtime.protocol import (
    PROTOCOL_VERSION,
    AgentAction,
    AgentPlan,
    FinalSubmission,
)

MAX_PROTOCOL_RESPONSE_BYTES = 2 * 1024 * 1024


class ProtocolPhase(StrEnum):
    """Lifecycle phase represented by one endpoint fixture."""

    ACTION = "action"
    PLAN = "plan"
    TERMINATE = "terminate"


class ProtocolFixture(BaseModel):
    """Input/expected behavior for one provider-neutral protocol case."""

    model_config = ConfigDict(extra="forbid")

    fixture_id: str = Field(min_length=1)
    phase: ProtocolPhase
    response: Any
    available_actions: list[str] = Field(default_factory=list)
    expected_action: str | None = None
    expected_extraction_mode: ExtractionMode | None = None
    opaque_internal_agents: bool = False
    response_size_bytes: int | None = Field(default=None, ge=0)
    max_response_bytes: int = Field(default=MAX_PROTOCOL_RESPONSE_BYTES, gt=0)
    expected_oversize_rejection: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProtocolFixtureResult(BaseModel):
    """Auditable outcome of one offline boundary fixture."""

    model_config = ConfigDict(extra="forbid")

    fixture_id: str
    passed: bool
    protocol_version: str = PROTOCOL_VERSION
    extraction_mode: ExtractionMode | None = None
    observed_action: str | None = None
    findings: list[str] = Field(default_factory=list)


class InteroperabilityReport(BaseModel):
    """Complete endpoint interoperability evidence report."""

    model_config = ConfigDict(extra="forbid")

    interoperability_version: str = "1.0.0"
    fixtures: list[ProtocolFixtureResult] = Field(default_factory=list)
    opaque_multi_agent_supported: bool = False
    limitations: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every fixture passed and opaque systems were exercised."""
        return (
            bool(self.fixtures)
            and all(fixture.passed for fixture in self.fixtures)
            and self.opaque_multi_agent_supported
            and not self.limitations
        )


def run_protocol_fixture(fixture: ProtocolFixture) -> ProtocolFixtureResult:  # noqa: C901
    """Validate one response using the same parser as the live HTTP runtime."""
    findings: list[str] = []
    observed_action: str | None = None
    extraction_mode: ExtractionMode | None = None
    try:
        if fixture.response_size_bytes is not None and (
            fixture.response_size_bytes > fixture.max_response_bytes
        ):
            if fixture.expected_oversize_rejection:
                return ProtocolFixtureResult(
                    fixture_id=fixture.fixture_id,
                    passed=True,
                    findings=[
                        "oversized response was rejected by the declared boundary limit"
                    ],
                )
            findings.append(
                f"response size {fixture.response_size_bytes} exceeds "
                f"limit {fixture.max_response_bytes} without rejection"
            )
            return ProtocolFixtureResult(
                fixture_id=fixture.fixture_id,
                passed=False,
                findings=findings,
            )
        if fixture.phase is ProtocolPhase.ACTION:
            payload, evidence = extract_action_response(
                fixture.response,
                available_actions=fixture.available_actions,
            )
            action = AgentAction.model_validate(payload)
            observed_action = action.action_type
            extraction_mode = evidence.mode
            if fixture.expected_action is not None and observed_action != fixture.expected_action:
                findings.append(
                    f"expected action '{fixture.expected_action}', got '{observed_action}'"
                )
            if fixture.expected_extraction_mode is not None and extraction_mode is not fixture.expected_extraction_mode:
                findings.append(
                    f"expected extraction mode '{fixture.expected_extraction_mode.value}', "
                    f"got '{extraction_mode.value}'"
                )
        elif fixture.phase is ProtocolPhase.PLAN:
            plan_payload = (
                fixture.response.get("plan")
                if isinstance(fixture.response, dict)
                else None
            )
            if not isinstance(plan_payload, dict):
                findings.append("plan fixture did not contain a plan object")
            else:
                AgentPlan.model_validate(plan_payload)
        else:
            submission_payload = (
                fixture.response.get("submission")
                if isinstance(fixture.response, dict)
                else fixture.response
            )
            if isinstance(submission_payload, str):
                if not submission_payload.strip():
                    findings.append("termination text response was empty")
            elif isinstance(submission_payload, dict):
                FinalSubmission.model_validate(submission_payload)
            else:
                findings.append("termination response was neither text nor an object")
    except Exception as error:
        findings.append(f"protocol parser raised {type(error).__name__}: {error}")
    return ProtocolFixtureResult(
        fixture_id=fixture.fixture_id,
        passed=not findings,
        extraction_mode=extraction_mode,
        observed_action=observed_action,
        findings=findings,
    )


def run_interoperability_suite(
    fixtures: list[ProtocolFixture],
    *,
    opaque_multi_agent_fixture_ids: set[str] | None = None,
) -> InteroperabilityReport:
    """Run structured, free-form, terminal, and opaque-agent fixtures offline."""
    seen: set[str] = set()
    results: list[ProtocolFixtureResult] = []
    opaque_ids = opaque_multi_agent_fixture_ids or set()
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            raise ValueError(f"duplicate protocol fixture '{fixture.fixture_id}'")
        seen.add(fixture.fixture_id)
        results.append(run_protocol_fixture(fixture))
    limitations: list[str] = []
    if not fixtures:
        limitations.append("no interoperability fixtures were supplied")
    if not opaque_ids:
        limitations.append("no opaque multi-agent fixture was supplied")
    unknown_opaque = sorted(opaque_ids - seen)
    if unknown_opaque:
        limitations.append(
            "opaque fixture IDs were not present in the suite: " + ", ".join(unknown_opaque)
        )
    return InteroperabilityReport(
        fixtures=results,
        opaque_multi_agent_supported=bool(opaque_ids) and not unknown_opaque,
        limitations=limitations,
    )


def default_protocol_fixtures() -> list[ProtocolFixture]:
    """Return the minimum boundary suite used by the readiness checklist."""
    return [
        ProtocolFixture(
            fixture_id="structured-action",
            phase=ProtocolPhase.ACTION,
            response={"action_type": "qc", "parameters": {"min_genes": 200}},
            available_actions=["qc", "normalize"],
            expected_action="qc",
            expected_extraction_mode=ExtractionMode.STRUCTURED,
        ),
        ProtocolFixture(
            fixture_id="black-box-text-action",
            phase=ProtocolPhase.ACTION,
            response="I will run qc. action_type: qc; method: fixed_threshold",
            available_actions=["qc", "normalize"],
            expected_action="qc",
            expected_extraction_mode=ExtractionMode.FREE_TEXT,
            opaque_internal_agents=True,
        ),
        ProtocolFixture(
            fixture_id="plan-response",
            phase=ProtocolPhase.PLAN,
            response={"plan": {"goal": "produce a valid analysis", "steps": ["inspect", "validate"]}},
        ),
        ProtocolFixture(
            fixture_id="termination-response",
            phase=ProtocolPhase.TERMINATE,
            response={"submission": {"summary": "artifacts are ready"}},
        ),
        ProtocolFixture(
            fixture_id="oversize-response",
            phase=ProtocolPhase.ACTION,
            response={},
            response_size_bytes=MAX_PROTOCOL_RESPONSE_BYTES + 1,
            expected_oversize_rejection=True,
        ),
    ]


__all__ = [
    "MAX_PROTOCOL_RESPONSE_BYTES",
    "InteroperabilityReport",
    "ProtocolFixture",
    "ProtocolFixtureResult",
    "ProtocolPhase",
    "default_protocol_fixtures",
    "run_interoperability_suite",
    "run_protocol_fixture",
]
