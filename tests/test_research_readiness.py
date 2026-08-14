"""Machine-checkable research-readiness evidence tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.research import (
    EvidenceItem,
    GateStatus,
    GoldenMetricCase,
    InvariantRelation,
    MetricInvariantCase,
    ReplicateScore,
    ReplicateStatus,
    ResearchGate,
    ResearchReadinessManifest,
    ReviewerAttestation,
    StudyArm,
    StudyArmKind,
    StudyPlan,
    benjamini_hochberg,
    build_benchmark_freeze_gate,
    build_starter_manifest,
    build_statistics_report,
    build_study_report,
    compare_paired,
    default_protocol_fixtures,
    dump_readiness_manifest,
    evaluate_research_readiness,
    load_readiness_manifest,
    run_golden_metric_suite,
    run_interoperability_suite,
    verify_evidence_item,
    verify_research_certification,
)
from agent_evals.research.certification import CertificationStatus


def _complete_manifest() -> ResearchReadinessManifest:
    manifest = build_starter_manifest(
        benchmark_id="benchmark",
        benchmark_version="1.0.0",
        manifest_id="manifest-1",
    )
    manifest.gates = [
        gate.model_copy(
            update={
                "checks": dict.fromkeys(gate.checks, True),
                "evidence": [
                    EvidenceItem(
                        evidence_id=f"{gate.gate.value}-evidence",
                        kind="test_report",
                        description=f"Evidence for {gate.gate.value}",
                        uri=f"evidence://{gate.gate.value}/report.json",
                        sha256="a" * 64,
                        externally_verified=True,
                    )
                ],
                "reviewers": [
                    ReviewerAttestation(
                        reviewer_id="reviewer-1",
                        role="independent reviewer",
                        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
                        decision="accept",
                        evidence_ids=[f"{gate.gate.value}-evidence"],
                        attestation_uri=f"evidence://{gate.gate.value}/review.json",
                        attestation_sha256="b" * 64,
                    )
                ],
                "externally_verified": True,
            }
        )
        for gate in manifest.gates
    ]
    return manifest


def test_benchmark_declaration_gets_a_digest_without_claiming_empirical_freeze() -> None:
    specification = load_benchmark("examples/benchmarks/pbmc-cell-annotation.yaml")

    gate = build_benchmark_freeze_gate(specification)

    assert gate.checks["benchmark_specification_digest_recorded"] is True
    assert gate.checks["dataset_digest_verified"] is None
    assert gate.evidence[0].sha256 is not None
    evaluation = gate.evaluate()
    assert evaluation.status is GateStatus.MISSING
    assert evaluation.evidence_verifications[0].status == "pending_external"


def test_starter_manifest_is_blocked_and_names_every_required_gate() -> None:
    manifest = build_starter_manifest(
        benchmark_id="benchmark",
        benchmark_version="1.0.0",
    )

    certification = evaluate_research_readiness(manifest)

    assert certification.status is CertificationStatus.BLOCKED
    assert certification.research_grade is False
    assert certification.readiness_fraction == 0
    assert {gate.gate for gate in certification.gates} == set(ResearchGate)
    assert all(gate.status is GateStatus.MISSING for gate in certification.gates)


def test_all_gates_require_evidence_and_external_verification() -> None:
    certification = evaluate_research_readiness(_complete_manifest())

    assert certification.status is CertificationStatus.CERTIFIED
    assert certification.research_grade is True
    assert certification.readiness_fraction == 1
    assert len(certification.manifest_sha256) == 64


def test_gate_field_cannot_claim_external_review_without_reviewed_evidence() -> None:
    manifest = build_starter_manifest(benchmark_id="benchmark", benchmark_version="1.0.0")
    gate = manifest.gates[0].model_copy(
        update={
            "checks": dict.fromkeys(manifest.gates[0].checks, True),
            "externally_verified": True,
            "evidence": [
                EvidenceItem(
                    evidence_id="unreviewed",
                    kind="report",
                    description="Not reviewed",
                    uri="evidence://unreviewed/report.json",
                    sha256="a" * 64,
                    externally_verified=False,
                )
            ],
        }
    )
    manifest.gates[0] = gate

    certification = evaluate_research_readiness(manifest)

    assert certification.gates[0].status is GateStatus.PENDING_EXTERNAL


def test_failed_gate_is_invalid_even_when_other_gates_pass() -> None:
    manifest = _complete_manifest()
    failed = manifest.gates[0].model_copy(
        update={"checks": {**manifest.gates[0].checks, "dataset_digest_verified": False}}
    )
    manifest.gates[0] = failed

    certification = evaluate_research_readiness(manifest)

    assert certification.status is CertificationStatus.INVALID
    assert certification.gates[0].status is GateStatus.FAIL
    assert any("benchmark_freeze" in reason for reason in certification.blocking_reasons)


def test_local_evidence_digest_is_verified_and_tampering_is_visible(tmp_path: Path) -> None:
    evidence_path = tmp_path / "report.json"
    evidence_path.write_text('{"status":"pass"}\n', encoding="utf-8")
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    item = EvidenceItem(
        evidence_id="local-report",
        kind="report",
        description="Local report",
        uri="report.json",
        sha256=digest,
    )

    valid = verify_evidence_item(item, root=tmp_path)
    assert valid.status == "valid"
    evidence_path.write_text('{"status":"changed"}\n', encoding="utf-8")
    invalid = verify_evidence_item(item, root=tmp_path)
    assert invalid.status == "mismatch"
    assert invalid.observed_sha256 != invalid.expected_sha256


def test_certificate_integrity_detects_claim_and_manifest_tampering() -> None:
    manifest = _complete_manifest()
    certificate = evaluate_research_readiness(manifest)

    verified = verify_research_certification(certificate, manifest=manifest)
    assert verified.valid is True
    assert certificate.verify_integrity() is True

    tampered_certificate = certificate.model_copy(
        update={"warnings": ["unreviewed claim"]}
    )
    assert verify_research_certification(
        tampered_certificate, manifest=manifest
    ).valid is False

    tampered_manifest = manifest.model_copy(update={"metadata": {"changed": True}})
    assert verify_research_certification(
        certificate, manifest=tampered_manifest
    ).manifest_digest_matches is False


def test_readiness_manifest_round_trips_without_changing_its_digest(tmp_path: Path) -> None:
    manifest = _complete_manifest()
    path = tmp_path / "readiness.yaml"

    dump_readiness_manifest(manifest, path)
    restored = load_readiness_manifest(path)

    assert restored.canonical_digest() == manifest.canonical_digest()


def test_golden_suite_checks_expected_values_determinism_and_invariants() -> None:
    report = run_golden_metric_suite(
        "toy.metric",
        [
            GoldenMetricCase(case_id="low", inputs={"value": 0.2}, expected_normalized=0.2),
            GoldenMetricCase(case_id="high", inputs={"value": 0.8}, expected_normalized=0.8),
        ],
        lambda inputs: float(inputs["value"]),
        invariants=[
            MetricInvariantCase(
                invariant_id="higher-is-better",
                left_case_id="low",
                right_case_id="high",
                relation=InvariantRelation.NON_DECREASING,
            )
        ],
    )

    assert report.passed is True
    assert all(case.deterministic for case in report.cases)
    assert report.invariants[0].passed is True


def test_empty_golden_suite_is_not_research_evidence() -> None:
    report = run_golden_metric_suite("toy.metric", [], lambda inputs: 1.0)

    assert report.passed is False


def test_golden_suite_without_invariants_is_not_complete_evidence() -> None:
    report = run_golden_metric_suite(
        "toy.metric",
        [GoldenMetricCase(case_id="identity", inputs={"value": 0.5}, expected_normalized=0.5)],
        lambda inputs: float(inputs["value"]),
    )

    assert report.passed is False
    assert "no metric invariants" in report.limitations[0]


def test_statistics_rejects_an_empty_dimension_protocol() -> None:
    with pytest.raises(ValueError, match="at least one measured dimension"):
        build_statistics_report("study-1", {}, dimensions=())


def test_golden_suite_fails_a_broken_invariant() -> None:
    report = run_golden_metric_suite(
        "toy.metric",
        [
            GoldenMetricCase(case_id="left", inputs={"value": 0.8}, expected_normalized=0.8),
            GoldenMetricCase(case_id="right", inputs={"value": 0.2}, expected_normalized=0.2),
        ],
        lambda inputs: float(inputs["value"]),
        invariants=[
            MetricInvariantCase(
                invariant_id="broken-order",
                left_case_id="left",
                right_case_id="right",
                relation=InvariantRelation.NON_DECREASING,
            )
        ],
    )

    assert report.passed is False
    assert report.invariants[0].passed is False


def _replicate(arm: str, replicate_id: str, score: float) -> ReplicateScore:
    return ReplicateScore(
        replicate_id=replicate_id,
        run_id=f"{arm}-{replicate_id}",
        seed=int(replicate_id),
        score=score,
        dimensions={"outcome": score},
    )


def test_study_report_pairs_by_replicate_id_and_records_replicate_gaps() -> None:
    plan = StudyPlan(
        study_id="study-1",
        benchmark_id="benchmark",
        benchmark_version="1.0.0",
        required_replicates=2,
        seed_schedule=[1, 2],
        arm_ids=["agent", "baseline"],
    )
    report = build_study_report(
        plan,
        [
            StudyArm(
                arm_id="agent",
                label="Agent",
                kind=StudyArmKind.AGENT,
                replicates=[_replicate("agent", "2", 0.9), _replicate("agent", "1", 0.8)],
            ),
            StudyArm(
                arm_id="baseline",
                label="Baseline",
                kind=StudyArmKind.DETERMINISTIC_BASELINE,
                implementation_digest="a" * 64,
                replicates=[_replicate("baseline", "1", 0.7)],
            ),
        ],
    )

    assert report.research_ready is False
    assert any("baseline" in limitation for limitation in report.limitations)
    comparison = report.statistics.comparisons[0]
    assert comparison.n_pairs == 1
    assert comparison.missing_replicates
    assert comparison.permutation_method == "exact_sign_flip"
    assert comparison.permutation_iterations == 2
    assert comparison.permutation_requested_iterations == 5000


def test_study_cannot_replace_a_missing_primary_seed_with_an_extra_seed() -> None:
    plan = StudyPlan(
        study_id="scheduled-study",
        benchmark_id="benchmark",
        benchmark_version="1.0.0",
        required_replicates=2,
        seed_schedule=[1, 2, 3],
        arm_ids=["agent", "baseline"],
        required_arm_kinds=[StudyArmKind.DETERMINISTIC_BASELINE],
    )
    digests = {
        "implementation_digest": "a" * 64,
        "environment_digest": "b" * 64,
        "benchmark_digest": "c" * 64,
        "dataset_digest": "d" * 64,
        "configuration_digest": "e" * 64,
    }
    report = build_study_report(
        plan,
        [
            StudyArm(
                arm_id="agent",
                label="Agent",
                kind=StudyArmKind.AGENT,
                replicates=[_replicate("agent", "1", 0.8), _replicate("agent", "3", 0.9)],
                **digests,
            ),
            StudyArm(
                arm_id="baseline",
                label="Baseline",
                kind=StudyArmKind.DETERMINISTIC_BASELINE,
                replicates=[
                    _replicate("baseline", "1", 0.7),
                    _replicate("baseline", "3", 0.75),
                ],
                **digests,
            ),
        ],
    )

    assert report.research_ready is False
    assert any("missing required frozen seed(s)" in item for item in report.limitations)


def test_failed_replicates_cannot_be_smuggled_into_statistics_as_zero() -> None:
    with pytest.raises(ValidationError, match="must omit score"):
        ReplicateScore(
            replicate_id="failed",
            seed=1,
            score=0.0,
            status=ReplicateStatus.FAILED,
        )

    failed = ReplicateScore(
        replicate_id="1",
        seed=1,
        status=ReplicateStatus.FAILED,
        metadata={"reason": "timeout"},
    )
    completed = _replicate("baseline", "1", 0.4)
    comparison = compare_paired("agent", [failed], "baseline", [completed])

    assert comparison.n_pairs == 0
    assert "agent:1:failed" in comparison.missing_replicates


def test_benjamini_hochberg_rejects_invalid_p_values() -> None:
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        benjamini_hochberg([0.1, 1.2])



def test_pairing_rejects_same_id_with_different_frozen_seed() -> None:
    candidate = _replicate("agent", "1", 0.8).model_copy(update={"seed": 11})
    baseline = _replicate("baseline", "1", 0.7).model_copy(update={"seed": 22})

    comparison = compare_paired("agent", [candidate], "baseline", [baseline])

    assert comparison.n_pairs == 0
    assert any("seed_mismatch" in item for item in comparison.missing_replicates)


def test_protocol_suite_covers_structured_text_and_opaque_agents() -> None:
    report = run_interoperability_suite(
        default_protocol_fixtures(),
        opaque_multi_agent_fixture_ids={"black-box-text-action"},
    )

    assert report.passed is True
    assert {fixture.fixture_id for fixture in report.fixtures} == {
        "structured-action",
        "black-box-text-action",
        "plan-response",
        "termination-response",
        "oversize-response",
    }
