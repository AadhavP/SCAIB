"""Tests for the weighted geometric global agent score and its confidence."""

import pytest
from pydantic import ValidationError

from agent_evals.evaluation.global_score import (
    SCORE_VERSION,
    ScoreWeights,
    compute_global_agent_score,
    compute_score_confidence,
)


def test_score_is_the_weighted_geometric_mean_of_the_three_dimensions() -> None:
    weights = ScoreWeights(outcome=0.5, decision=0.3, trajectory=0.2)
    result = compute_global_agent_score(0.82, 0.75, 0.9, weights=weights)

    assert result is not None
    assert result.value == pytest.approx(0.82**0.5 * 0.75**0.3 * 0.9**0.2)
    assert result.score_version == SCORE_VERSION
    assert result.weights == weights


def test_a_mean_stays_on_the_scale_of_its_inputs() -> None:
    """The reason for the version bump: a product of three 0.8s reads as 0.512."""
    result = compute_global_agent_score(0.8, 0.8, 0.8)

    assert result is not None
    assert result.value == pytest.approx(0.8)


def test_a_worthless_dimension_still_annihilates_the_score() -> None:
    """Deliberately preserved from the product: good process cannot redeem a
    worthless artifact, and a good artifact cannot excuse the route to it."""
    assert compute_global_agent_score(0.0, 0.9, 0.9) is not None
    result = compute_global_agent_score(0.0, 0.9, 0.9)
    assert result is not None
    assert result.value == 0.0


def test_weights_that_are_not_a_mean_are_rejected() -> None:
    """Exponents summing to 1.5 depress every score; to 0.5, inflate every one."""
    with pytest.raises(ValidationError):
        ScoreWeights(outcome=0.5, decision=0.5, trajectory=0.5)


def test_the_neutral_default_sums_to_one_in_floating_point() -> None:
    """Three exact thirds are not representable, so the default must be built
    to survive its own validator rather than written as ``1/3`` three times."""
    weights = ScoreWeights.neutral()

    assert weights.outcome == pytest.approx(1 / 3)
    result = compute_global_agent_score(0.5, 0.5, 0.5)
    assert result is not None
    assert result.value == pytest.approx(0.5)


def test_a_zero_weighted_dimension_is_excluded_not_raised_to_the_zeroth_power() -> None:
    """``0.0 ** 0.0`` is 1.0, which would let an ignored dimension score perfectly."""
    weights = ScoreWeights(outcome=0.5, decision=0.5, trajectory=0.0)
    result = compute_global_agent_score(0.64, 0.64, 0.0, weights=weights)

    assert result is not None
    assert result.value == pytest.approx(0.64)


def test_a_zero_weighted_dimension_need_not_be_measured_at_all() -> None:
    weights = ScoreWeights(outcome=0.5, decision=0.5, trajectory=0.0)
    result = compute_global_agent_score(0.64, 0.64, None, weights=weights)

    assert result is not None
    assert result.value == pytest.approx(0.64)


def test_global_score_preserves_unavailable_outcome() -> None:
    assert compute_global_agent_score(None, 1.0, 1.0) is None


def test_an_unmeasured_decision_dimension_yields_no_score() -> None:
    assert compute_global_agent_score(0.8, None, 0.9) is None


# --------------------------------------------------------------------------
# Confidence qualifies a score and can never raise it
# --------------------------------------------------------------------------


def test_confidence_is_one_when_every_dimension_was_measurable() -> None:
    confidence = compute_score_confidence(
        ineligible_fraction_decision=0.0,
        ineligible_fraction_trajectory=0.0,
    )

    assert confidence.value == pytest.approx(1.0)


def test_confidence_falls_as_evidence_goes_missing() -> None:
    confidence = compute_score_confidence(
        ineligible_fraction_decision=1.0,
        ineligible_fraction_trajectory=0.5,
        decision_penalty=0.5,
        trajectory_penalty=0.5,
    )

    assert confidence.value == pytest.approx(0.25)


def test_confidence_can_never_exceed_one() -> None:
    """The one property that makes it safe to report beside a score."""
    for decision in (0.0, 0.25, 1.0):
        for trajectory in (0.0, 0.5, 1.0):
            confidence = compute_score_confidence(
                ineligible_fraction_decision=decision,
                ineligible_fraction_trajectory=trajectory,
            )
            assert confidence.value <= 1.0


def test_confidence_does_not_change_the_score_it_qualifies() -> None:
    """Folding it in would penalize an agent for the harness's blindness, and
    would pay an agent that suppressed measurable evidence."""
    thin = compute_score_confidence(
        ineligible_fraction_decision=1.0,
        ineligible_fraction_trajectory=1.0,
    )
    with_confidence = compute_global_agent_score(0.8, 0.8, 0.8, confidence=thin)
    without = compute_global_agent_score(0.8, 0.8, 0.8)

    assert with_confidence is not None
    assert without is not None
    assert with_confidence.value == without.value
    assert with_confidence.confidence is not None
    assert with_confidence.confidence.value < 1.0
