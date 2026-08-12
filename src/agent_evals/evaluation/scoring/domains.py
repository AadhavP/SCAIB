"""Cross-domain scientific score aggregation."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.evaluation.scoring.aggregation import DomainScore


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


__all__ = ["ScientificScore", "aggregate_domains"]
