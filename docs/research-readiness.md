# Research readiness and certification

`agent-evals` now has an executable research-readiness protocol. It is deliberately
stricter than a passing test suite or a numerically populated run report:

```text
benchmark evidence
        + isolation evidence
        + metric validation
        + expert calibration
        + baselines and ablations
        + replicated statistics
        + endpoint interoperability
        + reproducibility archive
        = research-grade claim
```

The certificate is a claim boundary, not a score. A run can be useful while
remaining `exploratory`, but it cannot be presented as comparable research data
until the evidence manifest is complete.

## Create and evaluate the checklist

Create a versioned checklist for a benchmark:

```bash
uv run agent-evals research init \
  --benchmark-id pbmc-cell-annotation \
  --benchmark-version 1.0.0 \
  --output research/pbmc-cell-annotation-readiness.yaml
```

Populate each gate with explicit boolean checks and attach evidence records. An
evidence item should point to a report, fixture result, CI artifact, or reviewer
record and include a SHA-256 digest when it is a file. Local file references
are re-hashed by the verifier; remote references are never downloaded by the
CLI and must be covered by a reviewer attestation. A validated manifest can also
be attached to a scientific run with `--research-manifest`; the resulting
certificate is persisted beside the normal episode qualification and does not
change `O`, `D`, or `T`. Set `externally_verified: true` only after the named
evidence has actually been reviewed or reproduced; it is not a convenience flag
for making CI green.

An accepted gate also needs an independent `ReviewerAttestation` covering every
evidence ID. The attestation records reviewer identity, role, timestamp,
decision, and a digest-addressed review artifact. This is provenance, not a
cryptographic signature; organizations that require non-repudiation should store
signed review records at the attestation URI.

Evaluate without changing any scientific score:

```bash
uv run agent-evals research certify \
  --manifest research/pbmc-cell-annotation-readiness.yaml \
  --output research/pbmc-cell-annotation-certificate.json

# Use in release CI when a missing gate must fail the job.
uv run agent-evals research verify \
  --manifest research/pbmc-cell-annotation-readiness.yaml \
  --certificate research/pbmc-cell-annotation-certificate.json \
  --strict
```

`certify` emits `CERTIFIED`, `PARTIAL`, `BLOCKED`, or `INVALID` and attaches a
self-digest to the certificate. `verify` recomputes the manifest digest,
certificate claims, and local evidence bytes without network access. Omitted or
`null` checks are missing evidence; `false` is a failed claim. A certificate is
`CERTIFIED` only when all required gates pass and have external verification.

## The eight required gates

### Benchmark freeze

Record the benchmark schema/version and canonical digest, dataset checksum and
license, hidden-reference boundary tests, a frozen reference pipeline, and
held-out or adversarial cases. The agent-visible workspace must not contain
reference labels, reference outputs, or evaluator credentials.

### Isolation

Run adversarial Linux/container tests for filesystem scope, network policy,
process/memory/CPU/file limits, read-only inputs, environment scrubbing, dropped
Linux capabilities, `no-new-privileges`, read-only root filesystem, controlled
`/tmp`, non-root execution, and hidden-reference unreadability. Record the
immutable image digest, runtime, mounts, capabilities, kernel, and actual per-
control outcomes. A local process is an exploratory execution tier, not a
certified sandbox. For a comparable container run, the image reference must be
`image@sha256:<64 hexadecimal characters>`; an incomplete or tag-based digest is
reported as exploratory rather than immutable.


### Metrics

Pin implementation versions and backend parameters. Attach golden fixtures and
invariant results, test missing/malformed/ineligible semantics, quantify parameter
sensitivity across multiple scenarios, measure rank stability, and review metric
correlation/double counting. A metric unavailable
in a deployment must remain `UNIMPLEMENTED` rather than becoming an agent score of
zero.

The code-level fixture API is:

```python
from agent_evals.research import (
    GoldenMetricCase,
    MetricInvariantCase,
    InvariantRelation,
    run_golden_metric_suite,
)

report = run_golden_metric_suite(
    "my.metric",
    [GoldenMetricCase(case_id="identity", inputs={"x": 1}, expected_normalized=1)],
    lambda inputs: float(inputs["x"]),
    invariants=[MetricInvariantCase(
        invariant_id="monotone",
        left_case_id="identity",
        right_case_id="identity",
        relation=InvariantRelation.EQUAL,
    )],
)
assert report.passed
```

### Calibration

Freeze the decision-quality rubric. Collect independent expert ratings for a
representative set of observable decisions, report tolerance agreement and
ICC(2,1) absolute-agreement reliability, record adjudication, and calibrate the
decision score against those ratings. Do not treat an LLM judge as ground truth
without human calibration.

### Baselines

Run at least a deterministic reference baseline, a weak/random baseline, and an
evaluator-only oracle or upper bound. Baseline implementations, environments,
and seeds must be hashed. Compare agents on the same replicate IDs; an agent that
was run on different seeds or a different dataset is not a paired comparison.

### Statistics

Use repeated seeds, a frozen replicate schedule, paired candidate-minus-baseline
comparisons, bootstrap confidence intervals, and an explicit multiple-comparison
policy. The standard-library implementation is deterministic:

```python
from agent_evals.research import build_statistics_report

report = build_statistics_report("study-1", {"agent": agent_runs, "baseline": baseline_runs}, seed=7)
```Each report records `n`, mean/median, sample standard deviation, percentile
bootstrap interval, paired deltas, wins/ties/losses, a sign-flip p-value, the
requested and actual permutation counts/method, a standardized paired effect,
missing/failed replicate IDs, frozen seeds, and Benjamini-Hochberg adjusted
p-values.
 Failed or ineligible replicates cannot
carry a score and are excluded rather than silently converted to zero. Pairing
also requires identical replicate seeds. A confidence interval is not a
substitute for independent replication.

### Interoperability

Exercise the same boundary with structured actions, nested responses, free-form
text, plans, termination responses, oversize-response rejection, and opaque
multi-agent internals. The offline protocol suite is a deterministic CI smoke
test:

```bash
uv run agent-evals research protocol-check --strict
```

This verifies the parser contract; a real endpoint still needs a deployment test
with its URL, protocol version, model identity, timeout policy, and response
limits recorded. Strict endpoint mode requires a JSON-object response with the
negotiated protocol version and matching request ID for every lifecycle phase;
Level-0 black-box mode may use bounded text responses instead.

### Reproducibility

Persist source revision, dependency-lock digest, benchmark/profile digests, exact
configuration and seeds, package/platform details, environment image digest,
trajectory/events/artifact hashes, archive manifest, result digest, termination
reason, and the report itself. Recompute the report from the archive in an
independent environment and attach that review as evidence.

Every materialized scientific run also contains a replay-oriented public bundle:

```text
run/
├── events.ndjson
├── bundle_manifest.json
├── replay.json
├── artifacts/
├── workspace/
└── report.json
```

`events.ndjson` is canonical JSONL containing normalized observable agent and
environment events. `bundle_manifest.json` content-addresses every public file
and carries a self-digest. Verify it without importing the agent runtime:

```bash
uv run agent-evals verify-bundle results/<run-id>
```

This is an integrity and replay-readiness check, not proof that the scientific
result is correct. `verify-bundle --strict-replay` additionally requires the
bounded `replay.json` descriptor, a chained event ledger, and the public trajectory
and report files it references. The descriptor describes replay inputs; it does
not execute arbitrary agent code. A clean-room reproduction must still rerun
metric computation from the archived benchmark, dependency, dataset, and
reference packages.
The older `verify-run` archive verifier remains available for compatibility; new
release evidence should record both verifiers.

## Controlled study input

A study plan is intentionally separate from a run. It declares arm IDs, required
replicates, unique seeds, required deterministic/weak/oracle arm kinds, and
ablations. Study arms carry their replicate records plus benchmark, dataset,
configuration, environment, and implementation digests:

```yaml
study_id: pbmc-agent-v1
benchmark_id: pbmc-cell-annotation
benchmark_version: 1.0.0
required_replicates: 5
seed_schedule: [11, 22, 33, 44, 55]
arm_ids: [agent, scanpy]
ablations:
  - ablation_id: no-adaptation
    full_arm_id: agent
    ablated_arm_id: scanpy
    removed_components: ["adaptive replanning"]
    hypothesis: "Adaptation improves recovery after a QC regression."
    expected_direction: higher
```

Generate a report with `agent-evals research stats --plan plan.yaml --arms
arms.yaml`. Insufficient replicates remain visible as limitations and keep the
study out of `research_ready`; the tool does not manufacture certainty.

## What code can and cannot certify

The implementation can now enforce and report:

- strict, versioned evidence manifests and conservative gate semantics;
- deterministic bootstrap/paired statistics and multiple-comparison correction;
- golden metric expected values, status semantics, bounds, determinism, and
  pairwise invariants;
- baseline/ablation identity, seed, pairing, and replicate requirements;
- provider-neutral structured/text/plan/termination/oversize protocol fixtures;
  - bounded, secret-free endpoint exchange hashes and replay descriptors;
  - reviewer-attested, digest-verifiable evidence references and certificate
    integrity checks;
  - explicit separation between episode qualification and benchmark-wide research
    certification.


The following still require actual scientific or deployment evidence and cannot be
honestly generated by code alone:

- a frozen licensed dataset and hidden-reference package;
- Linux/container adversarial isolation execution;
- independent metric fixture values and sensitivity/correlation studies;
- expert decision ratings and inter-rater calibration;
- real baseline, ablation, and cross-dataset replicate runs;
- independent archive reproduction and publication review.

A blocked certificate is therefore the correct result on a fresh checkout. It is a
machine-readable to-do list, not a failure of the benchmark architecture.
