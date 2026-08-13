"""Cross-domain scientific score aggregation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.evaluation.scoring.aggregation import DomainScore

#: Said of a metric the profile scores but no metric result mentions, which is
#: what an ``external_score`` looks like when its evaluator produced nothing.
#: Deliberately weaker than a metric's own ``eligibility_reason``: all that is
#: known here is that nobody recorded a verdict, and inventing a cause would be
#: the unchecked claim this field exists to prevent.
UNRECORDED_METRIC_REASON = "no result recorded"


class ScientificScore(BaseModel):
    """Scientific score and domain component breakdown."""

    model_config = ConfigDict(extra="forbid")

    value: float | None = Field(default=None, ge=0, le=1)
    domains: list[DomainScore]
    formula: str


def aggregate_domains(domains: list[DomainScore]) -> ScientificScore:
    """Use a weighted geometric mean over available scientific domains."""
    included = [(domain.weight, domain.value) for domain in domains if domain.value is not None]
    value = None
    if included:
        denominator = sum(weight for weight, _ in included)
        if any(score == 0 for _, score in included):
            value = 0.0
        else:
            value = math.exp(
                sum(weight * math.log(max(score or 0.0, 1e-12)) for weight, score in included)
                / denominator
            )
    combined = [domain.domain for domain in domains if domain.value is not None]
    dropped = [domain.domain for domain in domains if domain.value is None]
    # Named the full domain list regardless of which ones were combined, so a run
    # whose robustness domain went unmeasured published a formula claiming it had
    # been weighed. The weights are renormalized over ``included`` above, so the
    # string was describing a computation that did not happen.
    formula = "weighted_geometric_mean(" + ", ".join(combined) + ")"
    if dropped:
        formula += " excluding_unmeasured(" + ", ".join(dropped) + ")"
    return ScientificScore(value=value, domains=domains, formula=formula)


def describe_unmeasured_domains(
    domains: Iterable[DomainScore],
    reasons: Mapping[str, str],
) -> list[str]:
    """Say why each domain :func:`aggregate_domains` dropped went unmeasured.

    The companion to the formula string: that names *which* domains were excluded,
    this names *why* each one was excludable. Without both, a run whose harness
    could not see the agent's work publishes the same ``None`` as a run whose
    metrics legitimately did not apply -- the absent-versus-unobserved ambiguity
    this project removes everywhere else.

    ``reasons`` maps a metric id to its ``eligibility_reason``. A metric with no
    entry is reported as :data:`UNRECORDED_METRIC_REASON` rather than skipped,
    because dropping it would shorten the explanation in exactly the case where
    least is known.

    Blocking metrics are preferred over the full exclusion list where there are
    any: a required metric voided the domain, and listing the optional metrics
    that merely did not apply beside it would present three equal causes for one
    real one. Domains that scored are not described at all, so this stays empty on
    a fully measured run instead of narrating it.
    """
    described: list[str] = []
    for domain in domains:
        if domain.value is not None:
            continue
        causes = domain.blocking_metrics or domain.excluded_metrics
        if causes:
            detail = "; ".join(
                f"{name} ({reasons.get(name, UNRECORDED_METRIC_REASON)})" for name in causes
            )
        else:
            detail = "no metric in this domain produced a value"
        described.append(f"domain '{domain.domain}' unmeasured: {detail}")
    return described


__all__ = [
    "UNRECORDED_METRIC_REASON",
    "ScientificScore",
    "aggregate_domains",
    "describe_unmeasured_domains",
]
