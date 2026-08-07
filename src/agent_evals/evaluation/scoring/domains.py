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
    return ScientificScore(
        value=value,
        domains=domains,
        formula="weighted_geometric_mean(" + ", ".join(domain.domain for domain in domains) + ")",
    )


__all__ = ["ScientificScore", "aggregate_domains"]
