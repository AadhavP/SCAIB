"""Tests for multiplicative global agent scoring."""

from agent_evals.evaluation.global_score import compute_global_agent_score


def test_global_score_is_multiplicative() -> None:
    result = compute_global_agent_score(0.82, 0.75, 0.9)

    assert result is not None
    assert result.value == 0.82 * 0.75 * 0.9


def test_global_score_preserves_unavailable_outcome() -> None:
    assert compute_global_agent_score(None, 1.0, 1.0) is None
