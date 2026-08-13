"""Frozen profile for PBMC differential-expression evaluations."""

from agent_evals.evaluation.profiles.base import (
    BenchmarkMetricProfile,
    MetricGroupProfile,
    MetricProfileEntry,
)


def pbmc_de_profile() -> BenchmarkMetricProfile:
    """Score a DE run on marker recovery, ranking quality, and effect sizes.

    This profile is the *only* declaration of the DE benchmark's scoring rule.
    ``pbmc-differential-expression.yaml`` declares no ``metric_groups`` at all,
    and the ids in its ``metrics:`` block (``marker-recovery-precision``,
    ``precision_at_k``, ``auroc``, ...) are benchmark-local descriptions that
    resolve to no registered metric -- so nothing in that file can be compared
    against the three entries below. The three chosen here answer the three
    scientific quantities the YAML describes in prose (marker-recovery precision,
    ranking quality, effect-size calibration), and that correspondence is a
    judgement no test can check, which is why it is written down here.

    This profile previously required ``differential_expression.pseudobulk_recall``,
    which is registered nowhere in the repo, and nothing resolved this profile so
    nothing noticed. Registering a metric under that name was considered and
    rejected: the only thing available to compute would be marker recall, and
    naming it after pseudobulk would add the very defect Stage 8 exists to remove
    (cf. ``embedding.continuity``, which returns ``knn_overlap`` verbatim).
    Whether the agent aggregated to pseudobulk before testing is a *decision*,
    scored under D through ``method_choice``; scoring it inside O would
    double-count it against D.

    The required/optional split follows the evidence each metric needs, which is
    not the same evidence for all three. Precision@K and AUROC read a reference
    marker set; effect-size correlation reads per-gene reference effect sizes. A
    benchmark supplying markers alone therefore scores the first two and excludes
    the third, instead of being charged a failure score for evidence the
    evaluator never supplied.
    """
    return BenchmarkMetricProfile(
        benchmark="pbmc_de",
        metric_groups={
            "biology": MetricGroupProfile(
                weight=1.0,
                metrics={
                    "differential_expression.precision_at_k": MetricProfileEntry(
                        weight=0.4
                    ),
                    "differential_expression.auroc": MetricProfileEntry(weight=0.4),
                    "differential_expression.effect_size_correlation": (
                        MetricProfileEntry(weight=0.2, required=False)
                    ),
                },
            )
        },
    )


__all__ = ["pbmc_de_profile"]
