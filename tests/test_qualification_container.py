"""Qualification tests for the research-grade container execution tier."""

from __future__ import annotations

from types import SimpleNamespace

from agent_evals.evaluation.qualification import QualificationStatus, qualify_run
from agent_evals.metrics.models import MetricDirection, MetricRole
from agent_evals.metrics.results import MetricResult, MetricStatus


def _evaluation() -> SimpleNamespace:
    score = 0.8
    return SimpleNamespace(
        global_agent_score=score,
        scientific_outcome_score=score,
        decision_quality_score=score,
        trajectory_score=score,
        outcome_limitations=[],
        metric_results=[
            MetricResult(
                metric_id="toy.metric",
                version="1.0",
                metric_name="Toy",
                role=MetricRole.PRIMARY,
                direction=MetricDirection.HIGHER_IS_BETTER,
                raw_value=score,
                normalized_value=score,
                eligible=True,
                status=MetricStatus.SCORED,
                eligibility_reason="fixture",
            )
        ],
    )


def _provenance() -> dict[str, object]:
    return {
        "source_dataset_sha256": "dataset",
        "source_dataset_checksum_verified": True,
        "dependency_lock_sha256": "lock",
        "source_revision": "revision",
        "requested_max_cells": None,
    }


def test_container_without_hardening_is_not_comparable() -> None:
    result = qualify_run(
        agent_termination_status="completed",
        environment={
            "backend": "container",
            "image_digest": "sha256:old",
            "isolation": {"controls": []},
        },
        provenance=_provenance(),
        evaluation=_evaluation(),
        archive_valid=True,
    )

    assert result.status is QualificationStatus.EXPLORATORY
    assert result.checks["container_hardening"] is False
    assert any("container hardening" in warning for warning in result.warnings)


def test_hardened_container_with_pinned_image_can_be_comparable() -> None:
    controls = [
        {"control": name, "outcome": "enforced"}
        for name in (
            "capabilities",
            "privilege_escalation",
            "root_filesystem",
            "temporary_filesystem",
            "non_root",
        )
    ]
    result = qualify_run(
        agent_termination_status="completed",
        environment={
            "backend": "container",
            "image_digest": "sha256:" + "a" * 64,
            "user": "65532:65532",
            "reference_store_scope": "outside_run_root",
            "isolation": {"controls": controls},
        },
        provenance=_provenance(),
        evaluation=_evaluation(),
        archive_valid=True,
    )

    assert result.status is QualificationStatus.CERTIFIED
    assert result.checks["container_hardening"] is True
    assert result.checks["immutable_environment_image"] is True


def test_declared_unobservable_cutoff_is_not_comparable() -> None:
    result = qualify_run(
        agent_termination_status="completed",
        environment={"backend": "typed", "reference_store_scope": "evaluator-process memory"},
        provenance=_provenance(),
        evaluation=_evaluation(),
        archive_valid=True,
        cutoff_evidence={
            "enforcement": {
                "max_steps": "enforced",
                "cost": "unobservable",
            }
        },
    )

    assert result.status is QualificationStatus.UNMEASURED
    assert result.checks["cutoff_enforcement"] is False
    assert any("unobservable" in reason for reason in result.blocking_reasons)


def test_malformed_container_digest_is_not_comparable() -> None:
    result = qualify_run(
        agent_termination_status="completed",
        environment={
            "backend": "container",
            "image_digest": "sha256:not-a-digest",
            "reference_store_scope": "outside_run_root",
            "isolation": {
                "controls": [
                    {"control": name, "outcome": "enforced"}
                    for name in (
                        "capabilities",
                        "privilege_escalation",
                        "root_filesystem",
                        "temporary_filesystem",
                        "non_root",
                    )
                ]
            },
        },
        provenance=_provenance(),
        evaluation=_evaluation(),
        archive_valid=True,
    )

    assert result.status is QualificationStatus.EXPLORATORY
    assert result.checks["immutable_environment_image"] is False
