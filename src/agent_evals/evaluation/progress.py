"""Per-stage scientific state ``S_t`` and the progress signal ``dS_t``.

Scientific quality used to be computed once, at the end, which left the
benchmark's trajectory claim unsupported: with no ``S_t`` there is no ``dS_t``,
so a trajectory could only be scored on step counts, wall clock, and a copy of
the final outcome. This module supplies the missing per-step number.

Two things here are easy to get wrong and are the reason the module exists.

**A naive ``S_t - S_{t-1}`` is invalid.** ``S`` jumps discontinuously the moment
a new metric family becomes eligible -- computing PCA makes clustering metrics
answerable, so the next step would register a large "improvement" that nobody
earned. Every delta is therefore re-aggregated over the metrics both steps could
answer, and reported as ``None`` when that intersection is empty. An unmeasurable
delta is not a zero delta.

**Deltas must not be summed.** Consecutive deltas are computed on *different*
comparable subsets, so they do not telescope; adding them would treat
incomparable quantities as a distance travelled. The run-level statistic is
therefore built from per-step verdicts, which is a valid operation on a sequence
of independent comparisons.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from agent_evals.core.reference_columns import (
    AGENT_CLUSTER_COLUMNS,
    AGENT_PREDICTION_COLUMNS,
)
from agent_evals.environment.models import StateDelta
from agent_evals.evaluation.profiles.base import (
    BenchmarkMetricProfile,
    MetricGroupProfile,
)
from agent_evals.evaluation.scoring.aggregation import (
    MetricScoreInput,
    WeightedGeometricAggregator,
)
from agent_evals.evaluation.scoring.domains import aggregate_domains

#: Change in ``S`` below which a step is called flat rather than moving. Floating
#: point re-aggregation of the same values is not bit-identical, so without a
#: threshold an unchanged run would report a stream of microscopic regressions.
PROGRESS_EPSILON = 1e-3

#: Neutral point of the reported progress scalar. A run whose measurable state
#: never moved is neither rewarded nor punished, because the benchmark cannot
#: tell a deliberately stable step from an ineffective one.
NEUTRAL_PROGRESS = 0.5


class PipelineStage(StrEnum):
    """Position in the single-cell workflow that an observed step reached."""

    QC = "qc"
    NORMALIZATION = "normalization"
    FEATURE_SELECTION = "feature_selection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    INTEGRATION = "integration"
    CLUSTERING = "clustering"
    ANNOTATION = "annotation"
    DIFFERENTIAL_EXPRESSION = "differential_expression"


# Observable markers, named for what the typed operations actually write rather
# than for what a stage is conventionally called. `test_stage_progress` asserts
# the operations' real output keys are recognized here, which is what keeps a
# rename in an operation from silently making its stage uninferrable.
_QC_OBS_COLUMNS = frozenset(
    {
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
        "n_counts",
        "n_genes",
        "percent_mito",
    }
)
_FEATURE_VAR_COLUMNS = frozenset(
    {"highly_variable", "highly_variable_rank", "dispersions", "dispersions_norm"}
)
_REDUCTION_OBSM_KEYS = frozenset({"X_pca", "X_umap", "X_tsne", "X_diffmap"})
_INTEGRATION_OBSM_KEYS = frozenset(
    {"X_integrated", "X_pca_harmony", "X_harmony", "X_scanorama", "X_scvi", "X_bbknn"}
)
_DE_FILE_MARKERS = ("de_", "differential_expression", "marker_genes", "rank_genes")

#: Artifact-id substrings per stage, in the same precedence order as the dataset
#: checks below.  This exists because two real stages leave *no* trace in any
#: namespace a dataset fingerprint covers.  Differential expression writes
#: ``uns["rank_genes_groups"]`` and a table file, and ``uns`` is deliberately
#: outside ``DATASET_NAMESPACES`` while ``files`` is unobservable on the typed
#: tier.  Normalization writes ``X`` -- except when the input arrives pre-scaled,
#: in which case the operation correctly skips the transform and the step leaves
#: the data untouched.  Both were observed producing a completely empty delta on a
#: real run, and both were therefore unattributable to any stage.
#:
#: Artifact ids are benchmark-authored rather than universal, so a benchmark that
#: names its outputs unrecognizably still degrades to ``None``.  That is the same
#: honest failure as before, and the alternative -- inferring a stage from the
#: agent's own account of what it ran -- is the thing this function exists to
#: avoid.
_ARTIFACT_MARKERS: tuple[tuple[PipelineStage, tuple[str, ...]], ...] = (
    (PipelineStage.ANNOTATION, ("annotat",)),
    (PipelineStage.CLUSTERING, ("cluster", "leiden", "louvain")),
    (PipelineStage.DIFFERENTIAL_EXPRESSION, ("de_", "de-", "differential", "marker", "rank_genes")),
    (PipelineStage.INTEGRATION, ("integrat", "harmony", "scanorama", "scvi", "bbknn")),
    (PipelineStage.DIMENSIONALITY_REDUCTION, ("pca", "umap", "tsne", "diffmap", "embedding")),
    (PipelineStage.FEATURE_SELECTION, ("hvg", "highly_variable", "highly-variable", "feature")),
    (PipelineStage.QC, ("qc", "quality")),
    (PipelineStage.NORMALIZATION, ("normali",)),
)


def _cells_removed(delta: StateDelta) -> bool:
    """Whether the observation confirms cells were dropped."""
    before, after = delta.n_obs_before, delta.n_obs_after
    return before is not None and after is not None and after < before


def _genes_removed(delta: StateDelta) -> bool:
    """Whether the observation confirms genes were dropped."""
    before, after = delta.n_vars_before, delta.n_vars_after
    return before is not None and after is not None and after < before


def _mentions(names: Iterable[str], markers: Sequence[str]) -> bool:
    """Whether any name contains any marker substring, case-insensitively."""
    lowered = [str(name).lower() for name in names]
    return any(marker in name for name in lowered for marker in markers)


def _stage_from_artifacts(artifact_ids: Iterable[str]) -> PipelineStage | None:
    """Name a stage from the ids of the artifacts a step produced."""
    names = [str(name).lower() for name in artifact_ids]
    if not names:
        return None
    for stage, markers in _ARTIFACT_MARKERS:
        if any(marker in name for name in names for marker in markers):
            return stage
    return None


def infer_stage(
    delta: StateDelta,
    artifact_ids: Iterable[str] = (),
) -> PipelineStage | None:
    """Name the workflow stage an observed step reached, if any.

    Inferred from what changed rather than from what the agent said it did, so
    the answer survives free execution where the harness does not run the
    science. Returns ``None`` when nothing observed identifies a stage, which
    keeps an unrecognized step unattributed instead of attributed wrongly.

    Checks run most-specific first and the first match wins. A step that does
    two stages at once is genuinely ambiguous, and picking its most distinctive
    evidence is more useful than refusing to label it.

    ``artifact_ids`` is a deliberately *weaker* second tier, consulted only once
    the dataset delta has said nothing. A changed namespace is evidence the data
    moved; a produced artifact is only evidence a file appeared. Both are the
    harness's own observations -- artifact records are what ``_normalize_result``
    enforces the declared contract against and what the Stage 3 validator
    re-checksums -- so neither is an agent claim.
    """
    obs_written = frozenset(delta.obs.added)
    if obs_written & frozenset(AGENT_PREDICTION_COLUMNS):
        return PipelineStage.ANNOTATION
    if obs_written & frozenset(AGENT_CLUSTER_COLUMNS):
        return PipelineStage.CLUSTERING
    if _mentions(delta.files.added, _DE_FILE_MARKERS):
        return PipelineStage.DIFFERENTIAL_EXPRESSION
    obsm_written = frozenset(delta.obsm.added) | frozenset(delta.obsm.changed)
    if obsm_written & _INTEGRATION_OBSM_KEYS:
        return PipelineStage.INTEGRATION
    if obsm_written & _REDUCTION_OBSM_KEYS:
        return PipelineStage.DIMENSIONALITY_REDUCTION
    if frozenset(delta.var.added) & _FEATURE_VAR_COLUMNS or _genes_removed(delta):
        return PipelineStage.FEATURE_SELECTION
    if obs_written & _QC_OBS_COLUMNS or _cells_removed(delta):
        return PipelineStage.QC
    # Before the artifact fallback, because normalization's only signature in the
    # data is "the matrix moved and the shape did not", which every earlier stage
    # also does incidentally.
    if delta.layers.added or delta.layers.changed or delta.matrix_changed:
        return PipelineStage.NORMALIZATION
    return _stage_from_artifacts(artifact_ids)


@dataclass(frozen=True)
class ProgressSignal:
    """One step's scientific state and its comparable change from the last.

    ``scientific_state`` is the state score over everything answerable *now*;
    ``delta`` is the change over everything answerable *both now and then*. They
    are deliberately different quantities, and only the second is comparable
    across steps.
    """

    step: int
    stage: PipelineStage | None
    scientific_state: float | None
    delta: float | None
    comparable_metrics: tuple[str, ...] = ()
    previous_state_on_comparable: float | None = None
    current_state_on_comparable: float | None = None
    scored_metrics: tuple[str, ...] = ()
    limitations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_measured(self) -> bool:
        """Whether this step produced a usable progress comparison."""
        return self.delta is not None


class ScientificProgressTracker:
    """Turn per-step metric values into ``S_t`` and a comparable ``dS_t``.

    Holds only the previous step's values, so it is safe to drive from the
    environment's step loop. The profile supplies the weights, which means the
    per-step number and the final outcome are aggregated by the same rules --
    the point of unifying the metric spine before this stage existed.
    """

    def __init__(self, profile: BenchmarkMetricProfile) -> None:
        self._profile = profile
        self._aggregator = WeightedGeometricAggregator()
        self._scored_names = frozenset(
            name for group in profile.metric_groups.values() for name in group.metrics
        )
        self._previous: Mapping[str, float] | None = None

    def reset(self) -> None:
        """Forget the baseline so a new episode starts without carry-over."""
        self._previous = None

    def record(
        self,
        *,
        step: int,
        stage: PipelineStage | None,
        metric_values: Mapping[str, float],
        limitations: Sequence[str] = (),
    ) -> ProgressSignal:
        """Score one step and return its comparable progress signal.

        ``metric_values`` must contain only metrics that actually computed. A
        metric that failed for want of an input its stage has not produced yet
        carries a failure score, and admitting those here would make every early
        step look catastrophic and every later one look like a recovery.
        """
        notes = list(limitations)
        retained = {
            name: value
            for name, value in metric_values.items()
            if name in self._scored_names
        }
        ignored = sorted(set(metric_values) - set(retained))
        if ignored:
            notes.append(
                "metric(s) computed but not weighted by this benchmark's profile, "
                f"so they do not affect progress: {', '.join(ignored)}"
            )
        state = self._aggregate(retained, retained)
        previous = self._previous
        self._previous = retained

        if previous is None:
            notes.append("no earlier step to compare against, so no delta exists")
            return ProgressSignal(
                step=step,
                stage=stage,
                scientific_state=state,
                delta=None,
                scored_metrics=tuple(sorted(retained)),
                limitations=tuple(notes),
            )

        comparable = sorted(set(previous) & set(retained))
        if not comparable:
            notes.append(
                "no metric was answerable at both this step and the last, so the "
                "change in scientific state is unmeasurable rather than zero"
            )
            return ProgressSignal(
                step=step,
                stage=stage,
                scientific_state=state,
                delta=None,
                scored_metrics=tuple(sorted(retained)),
                limitations=tuple(notes),
            )

        current_on = self._aggregate(retained, comparable)
        previous_on = self._aggregate(previous, comparable)
        delta = (
            None
            if current_on is None or previous_on is None
            else current_on - previous_on
        )
        if delta is None:
            notes.append(
                "the shared metrics belong to no scoreable domain, so the change "
                "in scientific state could not be aggregated"
            )
        return ProgressSignal(
            step=step,
            stage=stage,
            scientific_state=state,
            delta=delta,
            comparable_metrics=tuple(comparable),
            previous_state_on_comparable=previous_on,
            current_state_on_comparable=current_on,
            scored_metrics=tuple(sorted(retained)),
            limitations=tuple(notes),
        )

    def _aggregate(
        self,
        values: Mapping[str, float],
        names: Iterable[str],
    ) -> float | None:
        """Aggregate a subset of metrics using the profile's frozen weights.

        Restricting a domain drops the absent metrics from the profile rather
        than marking them excluded, because the aggregator voids a domain whose
        *required* metric is excluded -- correct when scoring a finished run, and
        wrong here, where a metric is absent because its stage has not run yet.

        ``external_score`` is dropped for the same reason: robustness needs
        replicate runs and cannot be answered mid-episode.
        """
        selected = {name: values[name] for name in names if name in values}
        if not selected:
            return None
        domains = []
        for domain, group in self._profile.metric_groups.items():
            entries = {
                name: entry
                for name, entry in group.metrics.items()
                if name in selected
            }
            if not entries:
                continue
            domains.append(
                self._aggregator.aggregate(
                    domain,
                    MetricGroupProfile(weight=group.weight, metrics=entries),
                    [
                        MetricScoreInput(name=name, value=selected[name])
                        for name in entries
                    ],
                )
            )
        if not domains:
            return None
        return aggregate_domains(domains).value


@dataclass(frozen=True)
class ScientificProgressReport:
    """Run-level view of a sequence of per-step progress signals."""

    #: ``None`` when no step produced a comparable delta. Callers must exclude
    #: this dimension rather than substituting a number: a benchmark that could
    #: not observe progress has not observed an absence of progress.
    value: float | None
    measured_steps: int
    regressions: int
    recoveries: int
    progress_per_action: float | None
    progress_per_cost: float | None
    total_gain: float
    total_loss: float
    signals: tuple[ProgressSignal, ...] = ()

    @property
    def formula(self) -> str:
        """Human-readable description of how ``value`` was produced."""
        return f"{NEUTRAL_PROGRESS:g} + mean(comparable_delta_S) / 2"

    def to_step_rows(self) -> list[dict[str, object]]:
        """Serialize per-step scientific state evidence for reports.

        These rows are evaluator-side evidence only. They explain how the
        trajectory score used scientific progress without exposing hidden
        reference-derived metrics to the agent during the run.
        """
        return [
            {
                "step": signal.step,
                "stage": None if signal.stage is None else signal.stage.value,
                "scientific_state": signal.scientific_state,
                "delta": signal.delta,
                "comparable_metrics": list(signal.comparable_metrics),
                "previous_state_on_comparable": signal.previous_state_on_comparable,
                "current_state_on_comparable": signal.current_state_on_comparable,
                "scored_metrics": list(signal.scored_metrics),
                "limitations": list(signal.limitations),
            }
            for signal in self.signals
        ]


def summarize_progress(
    signals: Sequence[ProgressSignal],
    *,
    action_count: int,
    token_usage: int | None = None,
    runtime_seconds: float | None = None,
) -> ScientificProgressReport:
    """Reduce per-step signals to one bounded progress score plus its evidence.

    The score is a bounded transform of the *mean* comparable delta, not of
    their sum. Summing would imply the deltas compose into a distance travelled,
    which is false: each is measured over whichever metrics its two steps had in
    common, so they are independent comparisons rather than segments of a path.

    ``NEUTRAL_PROGRESS`` is the fixed point. A run whose measurable state never
    moved scores exactly neutral, so this term cannot be raised by doing nothing
    and cannot be lowered by the harness failing to look.
    """
    deltas = [signal.delta for signal in signals if signal.delta is not None]
    gains = [value for value in deltas if value > PROGRESS_EPSILON]
    losses = [-value for value in deltas if value < -PROGRESS_EPSILON]
    total_gain = sum(gains)
    total_loss = sum(losses)
    regressions = len(losses)
    recoveries = _count_recoveries(deltas)

    value: float | None = None
    if deltas:
        mean_delta = sum(deltas) / len(deltas)
        value = max(0.0, min(1.0, NEUTRAL_PROGRESS + mean_delta / 2))

    per_action = total_gain / action_count if action_count > 0 else None
    per_cost = _per_cost(total_gain, token_usage, runtime_seconds)
    return ScientificProgressReport(
        value=value,
        measured_steps=len(deltas),
        regressions=regressions,
        recoveries=recoveries,
        progress_per_action=None if per_action is None else min(1.0, per_action),
        progress_per_cost=per_cost,
        total_gain=total_gain,
        total_loss=total_loss,
        signals=tuple(signals),
    )


def _count_recoveries(deltas: Sequence[float]) -> int:
    """Count gains that directly follow a loss.

    Distinguishing recovery from steady improvement is the point: an agent that
    noticed a bad step and undid it demonstrates something an agent that never
    erred cannot, and both would otherwise land in the same mean.
    """
    recoveries = 0
    regressed = False
    for value in deltas:
        if value < -PROGRESS_EPSILON:
            regressed = True
        elif value > PROGRESS_EPSILON and regressed:
            recoveries += 1
            regressed = False
    return recoveries


def _per_cost(
    total_gain: float,
    token_usage: int | None,
    runtime_seconds: float | None,
) -> float | None:
    """Express gain against whichever cost the run actually recorded.

    Tokens are preferred because they are what a paid run is billed for; wall
    clock is the fallback. Both absent yields ``None`` rather than a ratio
    against an assumed cost.
    """
    if token_usage is not None and token_usage > 0:
        return total_gain / (token_usage / 1000)
    if runtime_seconds is not None and runtime_seconds > 0:
        return total_gain / runtime_seconds
    return None


__all__ = [
    "NEUTRAL_PROGRESS",
    "PROGRESS_EPSILON",
    "PipelineStage",
    "ProgressSignal",
    "ScientificProgressReport",
    "ScientificProgressTracker",
    "infer_stage",
    "summarize_progress",
]
