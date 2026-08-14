"""Evaluator-only oracle baseline boundary."""

from collections.abc import Callable

from agent_evals.baselines.base import BaselineResult, BaselineRunner


class OracleAgentBaseline(BaselineRunner):
    """Use a supplied completed-result scorer without affecting execution."""

    baseline_id = "oracle_agent"

    def __init__(self, scorer: Callable[[dict[str, object]], float] | None = None) -> None:
        self.scorer = scorer

    def run(self, context: dict[str, object] | None = None, seed: int = 0) -> BaselineResult:
        """Score evaluator-owned context only after a run has completed."""
        context = context or {}
        score = self.scorer(context) if self.scorer is not None else None
        return BaselineResult(
            baseline_id=self.baseline_id,
            status="evaluated" if score is not None else "unavailable",
            seed=seed,
            score=score,
            metadata={"seed": seed, "evaluator_only": True},
        )


__all__ = ["OracleAgentBaseline"]
