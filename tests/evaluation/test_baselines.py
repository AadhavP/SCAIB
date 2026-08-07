"""Baseline framework tests."""

from agent_evals.baselines import (
    OracleAgentBaseline,
    RandomAgentBaseline,
    ScanpyDefaultBaseline,
    SeuratReferenceBaseline,
)


def test_baselines_are_reproducible_and_explicit_about_availability() -> None:
    random = RandomAgentBaseline().run({"allowed_actions": ["qc", "normalize"]}, seed=7)
    assert random == RandomAgentBaseline().run({"allowed_actions": ["qc", "normalize"]}, seed=7)
    assert ScanpyDefaultBaseline().run().actions[0] == "qc"
    assert SeuratReferenceBaseline().run().status == "unavailable"
    assert OracleAgentBaseline(lambda context: float(context["score"])).run({"score": 0.8}).score == 0.8
