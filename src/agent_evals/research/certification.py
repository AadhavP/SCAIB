"""Machine-checkable evidence gates for research-grade benchmark claims.

A passing unit suite is not the same thing as a validated scientific benchmark.
This module makes that distinction executable: every claim is attached to a named
gate, every gate records its checks and evidence, and an overall certificate is
issued only when all required gates have both passed checks and external
verification. Missing evidence is never silently interpreted as a failed metric
or as a successful benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RESEARCH_CERTIFICATION_VERSION = "1.0.0"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_LOCAL_URI_SCHEMES = frozenset({"", "file"})
_REMOTE_URI_SCHEMES = frozenset({"http", "https", "doi", "s3", "evidence"})


def _validate_sha256(value: str | None, field_name: str = "sha256") -> str | None:
    """Validate and normalize an optional SHA-256 digest."""
    if value is not None and _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character hexadecimal digest")
    return value.lower() if value is not None else None


class ResearchGate(StrEnum):
    """Evidence domains that must be closed before publishing a benchmark claim."""

    BENCHMARK_FREEZE = "benchmark_freeze"
    ISOLATION = "isolation"
    METRICS = "metrics"
    CALIBRATION = "calibration"
    BASELINES = "baselines"
    STATISTICS = "statistics"
    INTEROPERABILITY = "interoperability"
    REPRODUCIBILITY = "reproducibility"


class GateStatus(StrEnum):
    """Status of one research-readiness gate."""

    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"
    PENDING_EXTERNAL = "pending_external"


class CertificationStatus(StrEnum):
    """Overall claim status, deliberately stricter than engineering readiness."""

    CERTIFIED = "certified"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    INVALID = "invalid"


class EvidenceItem(BaseModel):
    """Immutable reference to evidence supporting one gate."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    description: str = Field(min_length=1)
    uri: str | None = None
    sha256: str | None = None
    externally_verified: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        """Reject malformed digests before they can support a certificate."""
        return _validate_sha256(value)

    @model_validator(mode="after")
    def validate_external_reference(self) -> EvidenceItem:
        """Require an address and digest before evidence can be externally attested."""
        if self.externally_verified and (not self.uri or self.sha256 is None):
            raise ValueError(
                "externally verified evidence requires both uri and sha256"
            )
        return self


class ReviewerAttestation(BaseModel):
    """Provenance for an independent review of a gate's evidence.

    This is an attestation record, not a digital signature. A deployment that
    needs non-repudiation should store the signed review at ``attestation_uri``
    and provide its digest; the certificate only claims what this record says.
    """

    model_config = ConfigDict(extra="forbid")

    reviewer_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    reviewed_at: datetime
    decision: Literal["accept", "reject"]
    evidence_ids: list[str] = Field(min_length=1)
    independent: bool = True
    attestation_uri: str | None = None
    attestation_sha256: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("attestation_sha256")
    @classmethod
    def validate_attestation_digest(cls, value: str | None) -> str | None:
        """Validate the digest of the review record itself."""
        return _validate_sha256(value, "attestation_sha256")

    @model_validator(mode="after")
    def validate_attestation_reference(self) -> ReviewerAttestation:
        """Make accepted external reviews independently addressable."""
        if self.decision == "accept" and (
            not self.attestation_uri or self.attestation_sha256 is None
        ):
            raise ValueError(
                "an accepted reviewer attestation requires attestation_uri and "
                "attestation_sha256"
            )
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("reviewer attestation evidence_ids must be unique")
        return self


class EvidenceVerification(BaseModel):
    """Result of checking an evidence reference against its declared digest."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    status: Literal["valid", "pending_external", "missing", "mismatch", "unsupported"]
    uri: str | None = None
    expected_sha256: str | None = None
    observed_sha256: str | None = None
    detail: str | None = None

    @property
    def valid(self) -> bool:
        """Whether the evidence bytes matched, or await an external review."""
        return self.status in {"valid", "pending_external"}


class GateEvidence(BaseModel):
    """Checks and citations for one required research gate."""

    model_config = ConfigDict(extra="forbid")

    gate: ResearchGate
    required: bool = True
    external_verification_required: bool = True
    externally_verified: bool = False
    checks: dict[str, bool | None] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    reviewers: list[ReviewerAttestation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("checks")
    @classmethod
    def validate_check_names(cls, value: dict[str, bool | None]) -> dict[str, bool | None]:
        """Keep checklist keys readable and stable in reports."""
        for name in value:
            if not name.strip():
                raise ValueError("research gate check names cannot be empty")
        return value

    def evaluate(self, *, evidence_root: Path | None = None) -> GateEvaluation:  # noqa: C901
        """Evaluate this gate without making assumptions about absent evidence."""
        failed = sorted(name for name, value in self.checks.items() if value is False)
        missing = sorted(name for name, value in self.checks.items() if value is None)
        reasons: list[str] = []
        verifications = [
            verify_evidence_item(item, root=evidence_root) for item in self.evidence
        ]
        reviewer_verifications: list[EvidenceVerification] = []
        for reviewer in self.reviewers:
            if reviewer.decision != "accept":
                continue
            reviewer_verifications.append(
                verify_evidence_item(
                    EvidenceItem(
                        evidence_id=f"reviewer:{reviewer.reviewer_id}",
                        kind="reviewer_attestation",
                        description="Digest-addressed reviewer attestation",
                        uri=reviewer.attestation_uri,
                        sha256=reviewer.attestation_sha256,
                    ),
                    root=evidence_root,
                )
            )
        for item, verification in zip(self.evidence, verifications, strict=True):
            if verification.status == "mismatch":
                failed.append(f"evidence:{item.evidence_id}:sha256")
                reasons.append(
                    f"evidence '{item.evidence_id}' does not match its declared sha256"
                )
            elif verification.status in {"missing", "unsupported"}:
                missing.append(f"evidence:{item.evidence_id}:verifiable_reference")
                reasons.append(
                    f"evidence '{item.evidence_id}' could not be verified: "
                    f"{verification.detail or verification.status}"
                )
        if not self.checks:
            missing.append("gate checklist is empty")
        if not self.evidence:
            missing.append("no evidence item was attached")
        evidence_ids = {item.evidence_id for item in self.evidence}
        reviewer_ids: list[str] = []
        accepted_reviewed_ids: set[str] = set()
        for reviewer in self.reviewers:
            if reviewer.reviewer_id in reviewer_ids:
                failed.append(f"reviewer:{reviewer.reviewer_id}:duplicate")
                continue
            reviewer_ids.append(reviewer.reviewer_id)
            if reviewer.decision == "reject":
                failed.append(f"reviewer:{reviewer.reviewer_id}:rejected")
            elif reviewer.independent:
                accepted_reviewed_ids.update(reviewer.evidence_ids)
        reviewers_cover_evidence = (
            bool(self.reviewers)
            and accepted_reviewed_ids == evidence_ids
            and all(reviewer.independent for reviewer in self.reviewers)
        )
        for verification in reviewer_verifications:
            if verification.status == "mismatch":
                failed.append(f"{verification.evidence_id}:sha256")
                reasons.append(
                    f"reviewer attestation '{verification.evidence_id}' does not "
                    "match its declared sha256"
                )
            elif verification.status in {"missing", "unsupported"}:
                missing.append(f"{verification.evidence_id}:verifiable_reference")
        if failed:
            status = GateStatus.FAIL
            reasons.append("one or more required checks or evidence validations failed")
        elif missing:
            status = GateStatus.MISSING
            reasons.append("one or more required checks or evidence items are missing")
        elif self.external_verification_required and not (
            self.externally_verified
            and all(item.externally_verified for item in self.evidence)
            and reviewers_cover_evidence
        ):
            status = GateStatus.PENDING_EXTERNAL
            reasons.append(
                "external verification requires a gate attestation, per-item "
                "attestations, and an independent accepted reviewer covering all "
                "evidence items"
            )
        else:
            status = GateStatus.PASS
        if self.notes:
            reasons.extend(self.notes)
        return GateEvaluation(
            gate=self.gate,
            required=self.required,
            status=status,
            passed_checks=sorted(name for name, value in self.checks.items() if value is True),
            failed_checks=sorted(set(failed)),
            missing_checks=sorted(set(missing)),
            evidence_ids=[item.evidence_id for item in self.evidence],
            reviewer_ids=reviewer_ids,
            evidence_verifications=[*verifications, *reviewer_verifications],
            reasons=_unique(reasons),
        )


class GateEvaluation(BaseModel):
    """Persisted result of evaluating one gate's evidence."""

    model_config = ConfigDict(extra="forbid")

    gate: ResearchGate
    required: bool
    status: GateStatus
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    missing_checks: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    reviewer_ids: list[str] = Field(default_factory=list)
    evidence_verifications: list[EvidenceVerification] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether this gate is closed and publishable."""
        return self.status is GateStatus.PASS


class ResearchReadinessManifest(BaseModel):
    """Versioned evidence manifest supplied to the certification command."""

    model_config = ConfigDict(extra="forbid")

    certification_version: str = RESEARCH_CERTIFICATION_VERSION
    manifest_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    gates: list[GateEvidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_gates(self) -> ResearchReadinessManifest:
        """Reject duplicate gate declarations and duplicate evidence IDs."""
        gate_ids = [gate.gate for gate in self.gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("research readiness manifest contains duplicate gates")
        evidence_ids = [item.evidence_id for gate in self.gates for item in gate.evidence]
        duplicates = sorted({item for item in evidence_ids if evidence_ids.count(item) > 1})
        if duplicates:
            raise ValueError(
                "research readiness manifest contains duplicate evidence IDs: "
                + ", ".join(duplicates)
            )
        if self.certification_version != RESEARCH_CERTIFICATION_VERSION:
            raise ValueError(
                f"unsupported certification_version '{self.certification_version}'; "
                f"supported version is '{RESEARCH_CERTIFICATION_VERSION}'"
            )
        return self

    def canonical_payload(self) -> dict[str, Any]:
        """Return timestamp-independent JSON data used for reproducibility."""
        return self.model_dump(mode="json", exclude={"created_at"})

    def canonical_digest(self) -> str:
        """Hash the exact evidence declaration, excluding only its creation time."""
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_json(self) -> str:
        """Serialize the manifest as canonical, human-readable JSON."""
        return self.model_dump_json(indent=2)


class ResearchCertification(BaseModel):
    """Conservative, reproducible interpretation of a readiness manifest."""

    model_config = ConfigDict(extra="forbid")

    certification_version: str = RESEARCH_CERTIFICATION_VERSION
    manifest_id: str
    benchmark_id: str
    benchmark_version: str
    manifest_sha256: str
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: CertificationStatus
    claim_level: str
    readiness_fraction: float = Field(ge=0, le=1)
    gates: list[GateEvaluation] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    certificate_sha256: str | None = None

    @field_validator("manifest_sha256", "certificate_sha256")
    @classmethod
    def validate_certificate_digest(cls, value: str | None) -> str | None:
        """Reject malformed integrity claims in persisted certificates."""
        return _validate_sha256(value, "certificate digest")

    @property
    def research_grade(self) -> bool:
        """Whether the result may be described as research-grade."""
        return (
            self.status is CertificationStatus.CERTIFIED
            and self.certificate_sha256 is not None
        )

    def canonical_payload(self) -> dict[str, Any]:
        """Return timestamp- and self-digest-independent certificate data."""
        return self.model_dump(
            mode="json",
            exclude={"evaluated_at", "certificate_sha256"},
        )

    def canonical_digest(self) -> str:
        """Hash the certificate's claims, excluding volatile and recursive fields."""
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def verify_integrity(self) -> bool:
        """Return whether the persisted certificate digest matches its claims."""
        return self.certificate_sha256 == self.canonical_digest()

    def to_json(self) -> str:
        """Serialize the certification result."""
        return self.model_dump_json(indent=2)


def evaluate_research_readiness(
    manifest: ResearchReadinessManifest,
    *,
    evidence_root: Path | str | None = None,
) -> ResearchCertification:
    """Evaluate every required gate, adding explicit entries for omitted gates.

    ``evidence_root`` is optional because remote evidence cannot be downloaded by
    a certification command safely. When supplied, local file references are
    hashed before a gate can pass; remote references remain pending external
    review and must be covered by an accepted reviewer attestation.
    """
    root = Path(evidence_root).resolve() if evidence_root is not None else None
    declared = {gate.gate: gate.evaluate(evidence_root=root) for gate in manifest.gates}
    evaluations: list[GateEvaluation] = []
    for gate in ResearchGate:
        evaluations.append(
            declared.get(
                gate,
                GateEvaluation(
                    gate=gate,
                    required=True,
                    status=GateStatus.MISSING,
                    missing_checks=["gate was not declared in the manifest"],
                    reasons=["required research gate was omitted"],
                ),
            )
        )
    required = [gate for gate in evaluations if gate.required]
    passed = sum(gate.status is GateStatus.PASS for gate in required)
    readiness = passed / len(required) if required else 1.0
    failed = [gate for gate in required if gate.status is GateStatus.FAIL]
    incomplete = [
        gate
        for gate in required
        if gate.status in {GateStatus.MISSING, GateStatus.PENDING_EXTERNAL}
    ]
    if failed:
        status = CertificationStatus.INVALID
    elif not incomplete and passed == len(required):
        status = CertificationStatus.CERTIFIED
    elif passed:
        status = CertificationStatus.PARTIAL
    else:
        status = CertificationStatus.BLOCKED
    reasons = [
        f"{gate.gate.value}: {reason}"
        for gate in [*failed, *incomplete]
        for reason in gate.reasons
    ]
    warnings = [
        f"{gate.gate.value}: {reason}"
        for gate in evaluations
        if not gate.required
        for reason in gate.reasons
    ]
    certificate = ResearchCertification(
        manifest_id=manifest.manifest_id,
        benchmark_id=manifest.benchmark_id,
        benchmark_version=manifest.benchmark_version,
        manifest_sha256=manifest.canonical_digest(),
        status=status,
        claim_level="research_grade" if status is CertificationStatus.CERTIFIED else "prototype",
        readiness_fraction=readiness,
        gates=evaluations,
        blocking_reasons=_unique(reasons),
        warnings=_unique(warnings),
    )
    return certificate.model_copy(update={"certificate_sha256": certificate.canonical_digest()})


def verify_evidence_item(  # noqa: C901
    item: EvidenceItem,
    *,
    root: Path | None = None,
) -> EvidenceVerification:
    """Verify a local evidence file or classify a remote reference honestly.

    The checker never downloads URLs during certification. Network retrieval would
    make a certificate depend on mutable external state and would give a manifest
    an unexpected SSRF capability. Remote evidence therefore requires a declared
    digest plus an independent reviewer attestation; local files can be checked
    directly when ``root`` is supplied.
    """
    if not item.uri or item.sha256 is None:
        return EvidenceVerification(
            evidence_id=item.evidence_id,
            status="missing",
            uri=item.uri,
            expected_sha256=item.sha256,
            detail="evidence requires uri and sha256 for independent verification",
        )
    parsed = urlparse(item.uri)
    scheme = parsed.scheme.lower()
    if scheme in _REMOTE_URI_SCHEMES:
        return EvidenceVerification(
            evidence_id=item.evidence_id,
            status="pending_external",
            uri=item.uri,
            expected_sha256=item.sha256,
            detail="remote evidence was not downloaded; reviewer attestation is required",
        )
    if scheme not in _LOCAL_URI_SCHEMES:
        return EvidenceVerification(
            evidence_id=item.evidence_id,
            status="unsupported",
            uri=item.uri,
            expected_sha256=item.sha256,
            detail=f"unsupported evidence URI scheme '{scheme}'",
        )
    relative_to_root = scheme == "" and not Path(item.uri).is_absolute()
    if scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            return EvidenceVerification(
                evidence_id=item.evidence_id,
                status="unsupported",
                uri=item.uri,
                expected_sha256=item.sha256,
                detail="file evidence URI must refer to the local host",
            )
        path = Path(unquote(parsed.path))
    else:
        path = Path(item.uri)
    if root is not None and relative_to_root:
        path = root / path
    try:
        resolved = path.resolve()
    except OSError as error:
        return EvidenceVerification(
            evidence_id=item.evidence_id,
            status="missing",
            uri=item.uri,
            expected_sha256=item.sha256,
            detail=f"evidence path could not be resolved: {error}",
        )
    if root is not None and relative_to_root and not resolved.is_relative_to(root):
        return EvidenceVerification(
            evidence_id=item.evidence_id,
            status="unsupported",
            uri=item.uri,
            expected_sha256=item.sha256,
            detail="relative evidence path escapes the evidence root",
        )
    if not resolved.is_file():
        return EvidenceVerification(
            evidence_id=item.evidence_id,
            status="missing",
            uri=item.uri,
            expected_sha256=item.sha256,
            detail=f"evidence file does not exist: {resolved}",
        )
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        return EvidenceVerification(
            evidence_id=item.evidence_id,
            status="missing",
            uri=item.uri,
            expected_sha256=item.sha256,
            detail=f"evidence file could not be read: {error}",
        )
    observed = digest.hexdigest()
    if observed != item.sha256:
        return EvidenceVerification(
            evidence_id=item.evidence_id,
            status="mismatch",
            uri=item.uri,
            expected_sha256=item.sha256,
            observed_sha256=observed,
            detail="observed bytes do not match the declared digest",
        )
    return EvidenceVerification(
        evidence_id=item.evidence_id,
        status="valid",
        uri=item.uri,
        expected_sha256=item.sha256,
        observed_sha256=observed,
    )


class CertificationIntegrityReport(BaseModel):
    """Independent check that a certificate and its manifest still agree."""

    model_config = ConfigDict(extra="forbid")

    certificate_digest_matches: bool
    manifest_digest_matches: bool | None = None
    claims_match_manifest: bool | None = None
    evidence: list[EvidenceVerification] = Field(default_factory=list)
    reviewer_attestations: list[EvidenceVerification] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether the certificate is internally intact and not locally tampered."""
        return (
            self.certificate_digest_matches
            and self.manifest_digest_matches is not False
            and self.claims_match_manifest is not False
            and not any(
                item.status in {"missing", "mismatch", "unsupported"}
                for item in [*self.evidence, *self.reviewer_attestations]
            )
        )


def verify_research_certification(
    certification: ResearchCertification,
    *,
    manifest: ResearchReadinessManifest | None = None,
    evidence_root: Path | str | None = None,
) -> CertificationIntegrityReport:
    """Verify certificate bytes, manifest identity, and local evidence references."""
    limitations: list[str] = []
    manifest_matches: bool | None = None
    claims_match: bool | None = None
    evidence: list[EvidenceVerification] = []
    if manifest is not None:
        manifest_matches = certification.manifest_sha256 == manifest.canonical_digest()
        if not manifest_matches:
            limitations.append("certificate manifest_sha256 does not match the supplied manifest")
        expected = evaluate_research_readiness(manifest, evidence_root=evidence_root)
        claims_match = certification.canonical_payload() == expected.canonical_payload()
        if not claims_match:
            limitations.append("certificate claims do not match a fresh manifest evaluation")
        root = Path(evidence_root).resolve() if evidence_root is not None else None
        evidence = [
            verify_evidence_item(item, root=root)
            for gate in manifest.gates
            for item in gate.evidence
        ]
        reviewer_attestations = [
            verify_evidence_item(
                EvidenceItem(
                    evidence_id=f"reviewer:{reviewer.reviewer_id}",
                    kind="reviewer_attestation",
                    description="Digest-addressed reviewer attestation",
                    uri=reviewer.attestation_uri,
                    sha256=reviewer.attestation_sha256,
                ),
                root=root,
            )
            for gate in manifest.gates
            for reviewer in gate.reviewers
            if reviewer.decision == "accept"
        ]
    else:
        reviewer_attestations = []
    return CertificationIntegrityReport(
        certificate_digest_matches=certification.verify_integrity(),
        manifest_digest_matches=manifest_matches,
        claims_match_manifest=claims_match,
        evidence=evidence,
        reviewer_attestations=reviewer_attestations,
        limitations=limitations,
    )


def build_benchmark_freeze_gate(
    specification: Any,
    *,
    dataset_checksums_verified: bool | None = None,
    hidden_reference_boundary_tested: bool | None = None,
    reference_pipeline_digest: str | None = None,
    reference_pipeline_uri: str | None = None,
    specification_uri: str | None = None,
    held_out_case_count: int | None = None,
) -> GateEvidence:
    """Convert a validated benchmark declaration into freeze-gate evidence.

    Structural declaration checks can be automated, while dataset loading,
    hidden-reference boundary tests, reference-pipeline execution, and held-out
    case construction remain explicit inputs. The function therefore records the
    digest and declaration facts without pretending that a YAML file proves the
    empirical claims.
    """
    if not hasattr(specification, "model_dump_serializable"):
        raise TypeError("specification must expose model_dump_serializable()")
    payload = specification.model_dump_serializable()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    specification_digest = hashlib.sha256(encoded).hexdigest()
    metadata = getattr(specification, "metadata", None)
    datasets = list(getattr(specification, "datasets", []))
    version = str(getattr(metadata, "version", ""))
    declarations_complete = bool(
        datasets
        and all(
            bool(getattr(dataset, "source", None))
            and bool(getattr(dataset, "license", None))
            and bool(getattr(dataset, "citation", []))
            for dataset in datasets
        )
    )
    evidence = [
        EvidenceItem(
            evidence_id="benchmark-specification",
            kind="benchmark_specification",
            description="Canonical digest of the validated benchmark specification",
            # A digest without an address is not independently verifiable. The
            # default evidence URI is deliberately remote/pending rather than a
            # fake local path; callers with a materialized package can provide its
            # relative path through ``specification_uri``.
            uri=(
                specification_uri
                or f"evidence://{specification.metadata.id}/benchmark-specification.json"
            ),
            sha256=specification_digest,
        )
    ]
    if reference_pipeline_digest is not None:
        evidence.append(
            EvidenceItem(
                evidence_id="reference-pipeline",
                kind="reference_pipeline",
                description="Pinned reference pipeline artifact",
                uri=(
                    reference_pipeline_uri
                    or f"evidence://{specification.metadata.id}/reference-pipeline"
                ),
                sha256=reference_pipeline_digest,
            )
        )
    return GateEvidence(
        gate=ResearchGate.BENCHMARK_FREEZE,
        checks={
            "benchmark_version_pinned": bool(version),
            "benchmark_specification_digest_recorded": True,
            "dataset_digest_verified": dataset_checksums_verified,
            "dataset_license_and_provenance_recorded": declarations_complete,
            "hidden_reference_boundary_tested": hidden_reference_boundary_tested,
            "reference_pipeline_frozen": (
                True if reference_pipeline_digest is not None else None
            ),
            "held_out_or_adversarial_cases_defined": (
                held_out_case_count is not None and held_out_case_count > 0
            )
            if held_out_case_count is not None
            else None,
        },
        evidence=evidence,
    )


def build_starter_manifest(
    *,
    benchmark_id: str,
    benchmark_version: str,
    manifest_id: str = "research-readiness",
) -> ResearchReadinessManifest:
    """Create an explicit all-missing checklist for a benchmark author.

    This is intentionally not a certificate. It is the executable starting point
    that prevents a paper or CI job from forgetting a gate entirely.
    """
    checks = {
        ResearchGate.BENCHMARK_FREEZE: [
            "benchmark_version_pinned",
            "benchmark_specification_digest_recorded",
            "dataset_digest_verified",
            "dataset_license_and_provenance_recorded",
            "hidden_reference_boundary_tested",
            "reference_pipeline_frozen",
            "held_out_or_adversarial_cases_defined",
        ],
        ResearchGate.ISOLATION: [
            "immutable_environment_image",
            "filesystem_boundary_adversarial_test",
            "network_policy_adversarial_test",
            "resource_limits_adversarial_test",
            "hidden_reference_unreadable",
            "runtime_identity_recorded",
        ],
        ResearchGate.METRICS: [
            "implementations_pinned",
            "golden_fixtures_pass",
            "invariants_pass",
            "missing_and_ineligible_semantics_tested",
            "parameter_sensitivity_reported",
            "metric_correlation_reviewed",
        ],
        ResearchGate.CALIBRATION: [
            "decision_quality_rubric_frozen",
            "independent_expert_ratings_collected",
            "inter_rater_agreement_reported",
            "decision_score_calibrated_to_experts",
            "adjudication_protocol_recorded",
        ],
        ResearchGate.BASELINES: [
            "deterministic_reference_baseline",
            "weak_or_random_baseline",
            "oracle_or_upper_bound_baseline",
            "baseline_replicates_completed",
            "ablation_plan_completed",
        ],
        ResearchGate.STATISTICS: [
            "replicate_count_meets_protocol",
            "paired_comparisons_used",
            "confidence_intervals_reported",
            "seed_policy_frozen",
            "multiple_comparison_policy_recorded",
        ],
        ResearchGate.INTEROPERABILITY: [
            "structured_endpoint_fixture_passes",
            "black_box_text_fixture_passes",
            "timeout_and_oversize_response_tests_pass",
            "opaque_multi_agent_fixture_passes",
            "endpoint_identity_and_protocol_version_recorded",
        ],
        ResearchGate.REPRODUCIBILITY: [
            "archive_manifest_verifies",
            "public_run_bundle_verifies",
            "replay_descriptor_verifies",
            "source_revision_recorded",
            "dependency_lock_digest_recorded",
            "configuration_and_seed_recorded",
            "report_recomputed_from_archive",
            "independent_reviewer_reproduction_completed",
        ],
    }
    return ResearchReadinessManifest(
        manifest_id=manifest_id,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        gates=[
            GateEvidence(
                gate=gate,
                checks=dict.fromkeys(names),
            )
            for gate, names in checks.items()
        ],
    )


def load_readiness_manifest(path: Path | str) -> ResearchReadinessManifest:
    """Load a JSON or YAML evidence manifest and validate it strictly."""
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        payload = (
            json.load(handle)
            if manifest_path.suffix.lower() == ".json"
            else yaml.safe_load(handle)
        )
    if not isinstance(payload, dict):
        raise ValueError(f"research manifest '{manifest_path}' must contain a mapping")
    return ResearchReadinessManifest.model_validate(payload)


def dump_readiness_manifest(
    manifest: ResearchReadinessManifest,
    path: Path | str,
) -> None:
    """Write a validated research-readiness manifest as JSON or YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        output.write_text(
            yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )


def _unique(values: list[str]) -> list[str]:
    """Preserve diagnostic order while removing repeated causes."""
    return list(dict.fromkeys(values))


__all__ = [
    "CertificationIntegrityReport",
    "CertificationStatus",
    "EvidenceItem",
    "EvidenceVerification",
    "GateEvaluation",
    "GateEvidence",
    "GateStatus",
    "ResearchCertification",
    "ResearchGate",
    "ResearchReadinessManifest",
    "ReviewerAttestation",
    "build_benchmark_freeze_gate",
    "build_starter_manifest",
    "dump_readiness_manifest",
    "evaluate_research_readiness",
    "load_readiness_manifest",
    "verify_evidence_item",
    "verify_research_certification",
]
