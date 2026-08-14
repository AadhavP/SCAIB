"""Seeded random baseline plan."""

from random import Random

from agent_evals.baselines.base import BaselineResult, BaselineRunner


class RandomAgentBaseline(BaselineRunner):
    """Choose declared actions randomly with a reproducible seed."""

    baseline_id = "random_agent"

    def run(self, context: dict[str, object] | None = None, seed: int = 0) -> BaselineResult:
        """Generate a finite random action plan from the supplied action list."""
        context = context or {}
        raw_actions = context.get("allowed_actions", [])
        actions = [str(item) for item in raw_actions] if isinstance(raw_actions, list) else []
        rng = Random(seed)
        chosen = [rng.choice(actions)] if actions else []
        return BaselineResult(
            baseline_id=self.baseline_id,
            status="planned",
            seed=seed,
            actions=chosen,
            metadata={"seed": seed},
        )


__all__ = ["RandomAgentBaseline"]
