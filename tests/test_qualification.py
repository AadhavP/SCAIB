"""Run qualification must prevent incomplete evidence from looking comparable."""

from types import SimpleNamespace

from agent_evals.evaluation.qualification import QualificationStatus, qualify_run
from agent_evals.metrics.models import MetricDirection, MetricRole
from agent_evals.metrics.results import MetricResult, MetricStatus


def _evaluation(*, score: float | None = 0.8, limitations: list[str] | None = None):
    return SimpleNamespace(
        global_agent_score=score,
        scientific_outcome_score=score,
        decision_quality_score=score,
        trajectory_score=score,
        outcome_limitations=limitations or [],
        metric_results=[
            MetricResult(
                metric_id="cell_annotation.macro_f1",
                version="1.0",
                metric_name="Macro F1",
                role=MetricRole.PRIMARY,
                direction=MetricDirection.HIGHER_IS_BETTER,
                raw_value=score,
                normalized_value=score,
                eligible=True,
                status=MetricStatus.SCORED,
                eligibility_reason="candidate and reference were available",
            )
        ],
    )


def _provenance(**overrides: object) -> dict[str, object]:
    return {
        "source_dataset_sha256": "source-digest",
        "source_dataset_checksum_verified": True,
        "dependency_lock_sha256": "lock-digest",
        "source_revision": "revision",
        "requested_max_cells": None,
        **overrides,
    }


def test_a_complete_typed_run_is_certified() -> None:
    result = qualify_run(
        agent_termination_status="completed",
        environment={"backend": "typed"},
        provenance=_provenance(),
        evaluation=_evaluation(),
        archive_valid=True,
    )

    assert result.status is QualificationStatus.CERTIFIED
    assert result.score_comparable is True
    assert result.blocking_reasons == []


def test_local_free_execution_is_exploratory_even_with_a_numeric_score() -> None:
    result = qualify_run(
        agent_termination_status="completed",
        environment={"backend": "local"},
        provenance=_provenance(),
        evaluation=_evaluation(),
        archive_valid=True,
    )

    assert result.status is QualificationStatus.EXPLORATORY
    assert result.score_comparable is False
    assert any("not filesystem- or network-confined" in item for item in result.warnings)


def test_unmeasured_primary_evidence_is_not_a_certified_zero() -> None:
    result = qualify_run(
        agent_termination_status="completed",
        environment={"backend": "typed"},
        provenance=_provenance(),
        evaluation=_evaluation(score=None, limitations=["robustness was unmeasured"]),
        archive_valid=True,
    )

    assert result.status is QualificationStatus.UNMEASURED
    assert result.score_comparable is False
    assert any("combined benchmark score was unmeasured" in item for item in result.blocking_reasons)


def test_evaluator_error_invalidates_the_measurement() -> None:
    evaluation = _evaluation()
    evaluation.metric_results = [
        evaluation.metric_results[0].model_copy(
            update={"status": MetricStatus.EVALUATOR_ERROR}
        )
    ]

    result = qualify_run(
        agent_termination_status="completed",
        environment={"backend": "typed"},
        provenance=_provenance(),
        evaluation=evaluation,
        archive_valid=True,
    )

    assert result.status is QualificationStatus.INVALID
    assert any("evaluator raised" in item for item in result.blocking_reasons)


def test_archive_tampering_invalidates_an_otherwise_complete_run() -> None:
    result = qualify_run(
        agent_termination_status="completed",
        environment={"backend": "typed"},
        provenance=_provenance(),
        evaluation=_evaluation(),
        archive_valid=False,
    )

    assert result.status is QualificationStatus.INVALID
    assert result.score_comparable is False


def test_a_declared_shape_mismatch_blocks_comparability() -> None:
    result = qualify_run(
        agent_termination_status="completed",
        environment={"backend": "typed"},
        provenance=_provenance(
            declared_cells=68579,
            declared_genes=32738,
            dataset_shape_verified=False,
        ),
        evaluation=_evaluation(),
        archive_valid=True,
    )

    assert result.status is QualificationStatus.UNMEASURED
    assert result.checks["dataset_shape"] is False
    assert any("shape does not match" in item for item in result.blocking_reasons)


def test_a_verified_declared_shape_can_be_certified() -> None:
    result = qualify_run(
        agent_termination_status="completed",
        environment={"backend": "typed"},
        provenance=_provenance(
            declared_cells=120,
            declared_genes=10,
            dataset_shape_verified=True,
        ),
        evaluation=_evaluation(),
        archive_valid=True,
    )

    assert result.status is QualificationStatus.CERTIFIED
    assert result.checks["dataset_shape"] is True


def test_missing_resource_telemetry_is_disclosed_as_exploratory() -> None:
    evaluation = _evaluation()
    evaluation.trajectory = SimpleNamespace(resource_usage={})

    result = qualify_run(
        agent_termination_status="completed",
        environment={"backend": "typed"},
        provenance=_provenance(),
        evaluation=evaluation,
        archive_valid=True,
    )

    assert result.status is QualificationStatus.EXPLORATORY
    assert result.checks["resource_telemetry"] is False
    assert any("measured wall time" in item for item in result.warnings)
