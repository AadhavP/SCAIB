"""Research-grade qualification of one benchmark result.

A scientific score can be numerically computable while still being unsuitable for
comparison: a local workspace may not be isolated, a primary metric may be
unmeasured, or the public archive may have been modified. Qualification keeps
those claims separate from the score and gives consumers a machine-readable
answer to the question "can this result be treated as a certified benchmark
measurement?".
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.metrics.results import MetricResult, MetricStatus


class QualificationStatus(StrEnum):
    """Interpretation of a completed run's evidence."""

    CERTIFIED = "certified"
    EXPLORATORY = "exploratory"
    UNMEASURED = "unmeasured"
    INVALID = "invalid"


class RunQualification(BaseModel):
    """Auditable status of a run independently from its numeric score."""

    model_config = ConfigDict(extra="forbid")

    qualification_version: str = "1.0.0"
    status: QualificationStatus
    score_comparable: bool = False
    reasons: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: dict[str, bool | None] = Field(default_factory=dict)


def qualify_run(
    *,
    agent_termination_status: str,
    environment: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    evaluation: Any | None,
    archive_valid: bool | None,
    artifacts: Sequence[Any] | None = None,
    state_evidence: Mapping[str, Any] | None = None,
    cutoff_evidence: Mapping[str, Any] | None = None,
) -> RunQualification:
    """Classify a result without changing any scientific score.

    ``CERTIFIED`` is intentionally conservative. A run can still be useful and
    receive diagnostic scores when it is ``EXPLORATORY`` or ``UNMEASURED``; those
    scores simply must not be presented as comparable benchmark measurements.
    """
    checks: dict[str, bool | None] = {}
    invalid: list[str] = []
    unmeasured: list[str] = []
    warnings: list[str] = []

    _check_termination(agent_termination_status, checks, invalid)
    _check_archive(archive_valid, checks, invalid, warnings)
    _check_environment(environment, checks, warnings)
    _check_provenance(provenance, checks, invalid, unmeasured, warnings)
    _check_evaluation(evaluation, checks, invalid, unmeasured, warnings)
    _check_artifact_evidence(artifacts, checks, invalid, unmeasured, warnings)
    _check_state_evidence(state_evidence, checks, unmeasured, warnings)
    _check_cutoff_evidence(cutoff_evidence, checks, unmeasured)

    blocking = [*invalid, *unmeasured]
    if invalid:
        status = QualificationStatus.INVALID
    elif unmeasured:
        status = QualificationStatus.UNMEASURED
    elif warnings:
        status = QualificationStatus.EXPLORATORY
    else:
        status = QualificationStatus.CERTIFIED

    return RunQualification(
        status=status,
        score_comparable=status is QualificationStatus.CERTIFIED,
        reasons=_unique([*blocking, *warnings]),
        blocking_reasons=_unique(blocking),
        warnings=_unique(warnings),
        checks=checks,
    )


def _check_termination(
    status: str,
    checks: dict[str, bool | None],
    invalid: list[str],
) -> None:
    """Require the controller and agent to report a completed episode."""
    completed = status == "completed"
    checks["agent_completed"] = completed
    if not completed:
        invalid.append(f"agent termination status was '{status}', not 'completed'")


def _check_archive(
    archive_valid: bool | None,
    checks: dict[str, bool | None],
    invalid: list[str],
    warnings: list[str],
) -> None:
    """Require an independently verified public archive when available."""
    checks["archive_integrity"] = archive_valid
    if archive_valid is False:
        invalid.append("the public run archive failed independent integrity verification")
    elif archive_valid is None:
        warnings.append("the public run archive was not independently verified")


def _check_environment(
    environment: Mapping[str, Any] | None,
    checks: dict[str, bool | None],
    warnings: list[str],
) -> None:
    """Record execution-tier guarantees without calling local execution certified."""
    record = environment or {}
    backend = str(record.get("backend") or "typed")
    checks["execution_backend_research_grade"] = backend in {"typed", "container"}
    if backend == "local":
        warnings.append(
            "local free execution is not filesystem- or network-confined; "
            "the result is exploratory rather than comparable"
        )
    elif backend == "container":
        _check_container_environment(record, checks, warnings)
    elif backend not in {"typed", "container"}:
        warnings.append(f"execution backend '{backend}' is not a certified tier")

    reference_scope = record.get("reference_store_scope")
    reference_boundary = reference_scope in {
        "outside_run_root",
        "evaluator-process memory",
    }
    # The typed tier has no materialized reference path by definition. Preserve
    # compatibility with its compact programmatic record while requiring an
    # explicit scope for free/container workspaces, where a path boundary is a
    # claim that must be disclosed and checked.
    if reference_scope is None and backend == "typed":
        reference_boundary = True
    checks["reference_boundary"] = reference_boundary
    if reference_scope is None and backend != "typed":
        warnings.append("the evaluator-only reference boundary was not recorded")
    elif not reference_boundary:
        warnings.append(
            "the evaluator-only reference store scope was not independently established"
        )


def _check_container_environment(
    environment: Mapping[str, Any],
    checks: dict[str, bool | None],
    warnings: list[str],
) -> None:
    """Require the container evidence needed for a comparable run."""
    image_digest = environment.get("image_digest")
    immutable_digest = (
        isinstance(image_digest, str)
        and re.fullmatch(r"sha256:[0-9a-fA-F]{64}", image_digest) is not None
    )
    checks["immutable_environment_image"] = immutable_digest
    if not immutable_digest:
        warnings.append(
            "the container image was not pinned by a complete immutable sha256 digest"
        )
    isolation = environment.get("isolation") or {}
    controls = isolation.get("controls", []) if isinstance(isolation, Mapping) else []
    control_outcomes = {
        str(control.get("control")): str(control.get("outcome"))
        for control in controls
        if isinstance(control, Mapping)
    }
    unenforced = [
        name
        for name, outcome in control_outcomes.items()
        if outcome in {"unenforceable", "failed"}
    ]
    checks["requested_isolation_controls"] = not unenforced
    if unenforced:
        warnings.append(
            "requested isolation controls were not enforced: "
            + ", ".join(sorted(unenforced))
        )
    required_hardening = {
        "capabilities",
        "privilege_escalation",
        "root_filesystem",
        "temporary_filesystem",
        "non_root",
    }
    missing_hardening = sorted(
        name
        for name in required_hardening
        if control_outcomes.get(name) != "enforced"
    )
    checks["container_hardening"] = not missing_hardening
    if missing_hardening:
        warnings.append(
            "container hardening evidence was incomplete: "
            + ", ".join(missing_hardening)
        )


def _check_provenance(
    provenance: Mapping[str, Any],
    checks: dict[str, bool | None],
    invalid: list[str],
    unmeasured: list[str],
    warnings: list[str],
) -> None:
    """Check dataset identity and reproducibility metadata."""
    source_checksum = provenance.get("source_dataset_sha256")
    checksum_verified = provenance.get("source_dataset_checksum_verified")
    checks["dataset_identity"] = checksum_verified is True
    if checksum_verified is False:
        invalid.append("the loaded dataset did not match its declared checksum")
    elif checksum_verified is not True:
        if source_checksum is None:
            unmeasured.append("the source dataset checksum was unavailable")
        else:
            unmeasured.append(
                "the source dataset checksum was recorded but not verified against "
                "a benchmark declaration"
            )

    if provenance.get("requested_max_cells") is not None:
        warnings.append(
            "the run used a reduced max_cells subset and is not comparable to a full-data run"
        )
    if "dataset_shape_verified" in provenance:
        shape_verified = provenance.get("dataset_shape_verified")
        checks["dataset_shape"] = shape_verified is True
        if shape_verified is False:
            unmeasured.append(
                "the loaded dataset shape does not match the benchmark declaration"
            )
        elif shape_verified is not True:
            unmeasured.append(
                "the benchmark did not provide enough declared dataset dimensions "
                "to verify the loaded shape"
            )
    if provenance.get("dependency_lock_sha256") is None:
        warnings.append("no dependency lock digest was recorded")
    if provenance.get("source_revision") is None:
        warnings.append("no source revision was recorded")


def _check_evaluation(
    evaluation: Any | None,
    checks: dict[str, bool | None],
    invalid: list[str],
    unmeasured: list[str],
    warnings: list[str],
) -> None:
    """Check aggregate scores, primary metrics, and reference-leakage findings."""
    if evaluation is None:
        checks["evaluation_present"] = False
        unmeasured.append("no scientific evaluation was produced")
        return

    checks["evaluation_present"] = True
    _check_score_dimensions(evaluation, unmeasured)
    _check_metric_results(evaluation.metric_results, checks, invalid, unmeasured)
    _check_domain_evidence(evaluation, unmeasured)
    _check_resource_telemetry(evaluation, checks, warnings)
    _check_outcome_limitations(evaluation.outcome_limitations, checks, invalid, warnings)


def _check_score_dimensions(evaluation: Any, unmeasured: list[str]) -> None:
    """Do not certify a run with a missing *weighted* score dimension."""
    score_detail = getattr(evaluation, "score_detail", None)
    weights = getattr(score_detail, "weights", None)
    dimensions = (
        ("combined benchmark score", evaluation.global_agent_score, True),
        ("scientific outcome score", evaluation.scientific_outcome_score, getattr(weights, "outcome", 1) > 0),
        ("decision quality", evaluation.decision_quality_score, getattr(weights, "decision", 1) > 0),
        ("trajectory quality", evaluation.trajectory_score, getattr(weights, "trajectory", 1) > 0),
    )
    for label, value, required in dimensions:
        if required and value is None:
            unmeasured.append(f"the {label} was unmeasured")


def _check_metric_results(
    results: list[MetricResult],
    checks: dict[str, bool | None],
    invalid: list[str],
    unmeasured: list[str],
) -> None:
    """Separate evaluator failures from absent scientific measurements."""
    primary_unmeasured: list[str] = []
    evaluator_errors: list[str] = []
    for result in results:
        role = getattr(result.role, "value", str(result.role))
        if result.status is MetricStatus.EVALUATOR_ERROR:
            evaluator_errors.append(result.metric_id)
        profile_required = result.metadata.get("profile_required")
        is_required = (
            bool(profile_required)
            if isinstance(profile_required, bool)
            else role in {"primary", "gate"}
        )
        if is_required and result.status is not MetricStatus.SCORED:
            primary_unmeasured.append(f"{result.metric_id} ({result.status.value})")
    checks["primary_metrics_measured"] = not primary_unmeasured
    if primary_unmeasured:
        unmeasured.append(
            "primary scientific metrics were not scored: "
            + ", ".join(primary_unmeasured)
        )
    if evaluator_errors:
        invalid.append(
            "the evaluator raised while computing metric(s): "
            + ", ".join(evaluator_errors)
        )


def _check_domain_evidence(evaluation: Any, unmeasured: list[str]) -> None:
    """Block certification when a required scientific domain was dropped."""
    domains = getattr(evaluation, "domain_scores", None)
    if not isinstance(domains, list):
        return
    blocking_domains = [
        str(domain.domain)
        for domain in domains
        if domain.value is None and domain.blocking_metrics
    ]
    if blocking_domains:
        unmeasured.append(
            "required scientific domain(s) were unmeasured: "
            + ", ".join(blocking_domains)
        )


def _check_resource_telemetry(
    evaluation: Any,
    checks: dict[str, bool | None],
    warnings: list[str],
) -> None:
    """Require at least measured wall-time telemetry when the report carries it."""
    trajectory = getattr(evaluation, "trajectory", None)
    if trajectory is None:
        # Compatibility for compact in-memory evaluation doubles and reports
        # written before telemetry became part of the evaluation model.
        return
    resource_usage = getattr(trajectory, "resource_usage", None)
    measured = isinstance(resource_usage, Mapping) and "wall_time_seconds" in resource_usage
    checks["resource_telemetry"] = measured
    if not measured:
        warnings.append(
            "execution resource telemetry did not include measured wall time"
        )


def _check_artifact_evidence(  # noqa: C901
    artifacts: Sequence[Any] | None,
    checks: dict[str, bool | None],
    invalid: list[str],
    unmeasured: list[str],
    warnings: list[str],
) -> None:
    """Require produced artifacts to carry evaluator-owned validation evidence."""
    if artifacts is None:
        return
    missing_evidence: list[str] = []
    checksum_failures: list[str] = []
    uncheckable: list[str] = []
    for artifact in artifacts:
        artifact_id = str(getattr(artifact, "artifact_id", "unknown"))
        validation = getattr(artifact, "validation", None)
        if validation is None:
            missing_evidence.append(artifact_id)
            continue
        if getattr(validation, "checksum_verified", None) is False:
            checksum_failures.append(artifact_id)
        if not getattr(validation, "exists", False):
            uncheckable.append(f"{artifact_id} (artifact was not readable)")
        for rule in getattr(validation, "rules", []):
            outcome = getattr(getattr(rule, "outcome", None), "value", str(getattr(rule, "outcome", "")))
            if outcome == "uncheckable":
                uncheckable.append(f"{artifact_id} ({getattr(rule, 'name', 'rule')})")
        if getattr(validation, "checksum_verified", None) is None:
            warnings.append(f"artifact '{artifact_id}' has no independently comparable checksum")
    checks["artifact_validation_evidence"] = not missing_evidence and not uncheckable and not checksum_failures
    if missing_evidence:
        unmeasured.append(
            "produced artifacts had no evaluator validation record: "
            + ", ".join(missing_evidence)
        )
    if uncheckable:
        unmeasured.append(
            "artifact validation was uncheckable for: " + ", ".join(uncheckable)
        )
    if checksum_failures:
        invalid.append(
            "artifact checksum verification failed for: " + ", ".join(checksum_failures)
        )


def _check_state_evidence(
    state_evidence: Mapping[str, Any] | None,
    checks: dict[str, bool | None],
    unmeasured: list[str],
    warnings: list[str],
) -> None:
    """Require free-execution actions to have an independently observed delta."""
    if state_evidence is None:
        return
    action_count = int(state_evidence.get("action_count", 0) or 0)
    observed_count = int(state_evidence.get("observed_action_count", 0) or 0)
    backend = str(state_evidence.get("backend") or "typed")
    measured = action_count == 0 or observed_count == action_count
    checks["state_transition_evidence"] = measured
    if backend in {"local", "container"} and action_count and observed_count == 0:
        unmeasured.append(
            "free-execution actions produced no independently observed state delta"
        )
    elif action_count and observed_count < action_count:
        warnings.append(
            f"only {observed_count} of {action_count} action(s) had an observed state delta"
        )
    limitations = [str(item) for item in state_evidence.get("limitations", [])]
    warnings.extend(limitations)


def _check_cutoff_evidence(
    cutoff_evidence: Mapping[str, Any] | None,
    checks: dict[str, bool | None],
    unmeasured: list[str],
) -> None:
    """Ensure declared token/cost budgets were actually observable.

    Step and wall-clock limits are controller-owned. Token and monetary limits
    depend on provider telemetry, so a declared budget without a measurement is
    not evidence that the run stayed within it. Older compact qualification calls
    may omit this record; new universal-runtime runs pass it explicitly.
    """
    if cutoff_evidence is None:
        return
    enforcement = cutoff_evidence.get("enforcement")
    if not isinstance(enforcement, Mapping):
        checks["cutoff_enforcement"] = False
        unmeasured.append("cutoff enforcement evidence was not recorded")
        return
    unobservable = sorted(
        str(reason)
        for reason, status in enforcement.items()
        if str(status) == "unobservable"
    )
    checks["cutoff_enforcement"] = not unobservable
    if unobservable:
        unmeasured.append(
            "declared resource cutoff(s) were unobservable: "
            + ", ".join(unobservable)
        )


def _check_outcome_limitations(
    limitations: list[str],
    checks: dict[str, bool | None],
    invalid: list[str],
    warnings: list[str],
) -> None:
    """Promote confirmed leakage to an invalid result while preserving other gaps."""
    outcome_limitations = [str(item) for item in limitations]
    confirmed_leakage = [
        item
        for item in outcome_limitations
        if "confirmed" in item.lower() and "leak" in item.lower()
    ]
    checks["reference_leakage"] = not confirmed_leakage
    invalid.extend(confirmed_leakage)
    warnings.extend(outcome_limitations)


def _unique(values: list[str]) -> list[str]:
    """Preserve diagnostic order while removing duplicate explanations."""
    return list(dict.fromkeys(values))


__all__ = ["QualificationStatus", "RunQualification", "qualify_run"]
