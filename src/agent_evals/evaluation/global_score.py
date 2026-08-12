"""The single benchmark score, and the confidence that qualifies it.

The score is a *weighted geometric mean* of the three dimensions the benchmark
measures -- the scientific outcome ``O``, the methodological decisions ``D``, and
the trajectory ``T`` -- with ``wO + wD + wT = 1`` and the weights declared by the
benchmark rather than assumed here.

Three properties this shape has and a plain product does not.

**It is a mean, so it stays on the scale of its inputs.** A plain ``O*D*T`` of
three respectable 0.8s is 0.512, which reads as a mediocre run rather than a good
one, and the distortion compounds with every dimension added. The weighted
geometric mean of the same three is 0.8.

**The weights become a declaration rather than an accident.** Under a plain
product every dimension carries exponent 1, which is not "equal weighting" so
much as no weighting at all, and it cannot express a benchmark that cares more
about the result than the route, or the reverse. Because the exponents must sum
to 1, the trade-off has to be stated.

**Annihilation is preserved deliberately.** Any dimension at zero takes the whole
score to zero, exactly as the product did. That is the benchmark's central claim:
a worthless artifact is not redeemed by good process, and a good artifact reached
by unjustifiable decisions is not a success either.

Because this changes every previously published number it carries an explicit
:data:`SCORE_VERSION`, and the resolved weights are recorded in the result so an
archived score stays recomputable.

Confidence is reported *beside* the score and never folded into it. Its purpose is
to say how much of the evidence was measurable at all, and a quantity that
multiplied the score would turn "the harness could not look" into a penalty on the
agent -- or reward an agent for suppressing evidence, since fewer measurable
components would then mean a smaller product.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Bumped whenever the combination rule changes in a way that moves numbers.
#: Version 1 was the unweighted product ``O * D * T``.
SCORE_VERSION = 2

#: Tolerance for the weights-sum-to-one check. Three exact thirds are not
#: representable in binary floating point, so an exact comparison would reject
#: the neutral default.
WEIGHT_SUM_TOLERANCE = 1e-6

#: Default penalty coefficients ``kD`` and ``kT``. At 0.5, a dimension whose
#: evidence was entirely unmeasurable costs half of confidence rather than all of
#: it: the score is still a number computed from what *could* be seen, and
#: confidence's job is to report how thin that basis was, not to void it.
DEFAULT_DECISION_PENALTY = 0.5
DEFAULT_TRAJECTORY_PENALTY = 0.5


class ScoreWeights(BaseModel):
    """Exponents for the three scored dimensions, summing to one."""

    model_config = ConfigDict(extra="forbid")

    outcome: float = Field(ge=0, le=1)
    decision: float = Field(ge=0, le=1)
    trajectory: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ScoreWeights:
        """Reject a weighting that would not be a mean.

        Exponents summing to anything but one make scores incomparable across
        benchmarks: at 1.5 every result is systematically depressed, at 0.5
        systematically inflated, and in neither case is the number a weighted
        mean of anything.
        """
        total = self.outcome + self.decision + self.trajectory
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                "score weights must sum to 1.0 so the result is a weighted mean; "
                f"got {total!r}"
            )
        return self

    @classmethod
    def neutral(cls) -> ScoreWeights:
        """Weight the three dimensions equally.

        Deliberately neutral rather than opinionated. Any other split asserts
        something about what a benchmark values, and that assertion belongs to
        the benchmark making it rather than to the scoring code.
        """
        third = 1.0 / 3.0
        return cls(outcome=third, decision=third, trajectory=1.0 - 2 * third)


class ScoreConfidence(BaseModel):
    """How much of the decision and trajectory evidence was measurable.

    Qualifies a score without changing it. Bounded above by 1 by construction,
    since both fractions and both penalties are non-negative.
    """

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0, le=1)
    ineligible_fraction_decision: float = Field(ge=0, le=1)
    ineligible_fraction_trajectory: float = Field(ge=0, le=1)
    decision_penalty: float = Field(ge=0, le=1)
    trajectory_penalty: float = Field(ge=0, le=1)
    formula: str


class GlobalAgentScore(BaseModel):
    """The benchmark score together with everything needed to recompute it."""

    model_config = ConfigDict(extra="forbid")

    score_version: int
    scientific_outcome: float = Field(ge=0, le=1)
    decision_quality: float = Field(ge=0, le=1)
    trajectory_quality: float = Field(ge=0, le=1)
    weights: ScoreWeights
    value: float = Field(ge=0, le=1)
    formula: str
    confidence: ScoreConfidence | None = None


def compute_score_confidence(
    *,
    ineligible_fraction_decision: float,
    ineligible_fraction_trajectory: float,
    decision_penalty: float = DEFAULT_DECISION_PENALTY,
    trajectory_penalty: float = DEFAULT_TRAJECTORY_PENALTY,
) -> ScoreConfidence:
    """Report how thin the evidence under the D and T dimensions was."""
    decision_fraction = max(0.0, min(1.0, ineligible_fraction_decision))
    trajectory_fraction = max(0.0, min(1.0, ineligible_fraction_trajectory))
    raw = (
        1.0
        - decision_penalty * decision_fraction
        - trajectory_penalty * trajectory_fraction
    )
    return ScoreConfidence(
        value=max(0.0, min(1.0, raw)),
        ineligible_fraction_decision=decision_fraction,
        ineligible_fraction_trajectory=trajectory_fraction,
        decision_penalty=decision_penalty,
        trajectory_penalty=trajectory_penalty,
        formula=(
            f"1 - {decision_penalty:g}*ineligible_fraction_D"
            f" - {trajectory_penalty:g}*ineligible_fraction_T"
        ),
    )


def describe_score(
    weights: ScoreWeights,
    unmeasured: list[str] | None = None,
) -> str:
    """Describe the combination rule, naming any dimension that was unmeasured.

    Persisted into result JSON and read by people comparing runs, so a run with
    no score has to say *which* dimension is missing -- otherwise the absent
    number reads as a crash rather than an honest gap. The decision dimension's
    composition is spelled out because it is the one factor built from two
    separately-reported halves.
    """
    combination = " * ".join(
        f"{name}^{weight:.2f}"
        for name, weight in (
            ("scientific_outcome", weights.outcome),
            ("decision_quality", weights.decision),
            ("trajectory_quality", weights.trajectory),
        )
        if weight > 0
    )
    formula = (
        f"{combination} "
        "(decision_quality = decision_score * method_selection_score)"
    )
    if not unmeasured:
        return formula
    return f"{formula} (not computed: {', '.join(unmeasured)} unmeasured)"


def compute_global_agent_score(
    scientific_outcome: float | None,
    decision_quality: float | None,
    trajectory_quality: float | None,
    *,
    weights: ScoreWeights | None = None,
    confidence: ScoreConfidence | None = None,
) -> GlobalAgentScore | None:
    """Compute the global score, or nothing when a weighted dimension is unmeasured.

    A missing dimension yields no score rather than a substituted one. Filling
    ``decision_quality`` with a neutral 1.0 made an agent that recorded no
    decisions score *higher* than one whose decisions were scored and found
    merely good, which inverts what the benchmark exists to measure.

    A dimension the benchmark weights at zero is *excluded* rather than raised to
    the zeroth power. ``0.0 ** 0.0`` is 1.0 in IEEE arithmetic, so exponentiating
    would silently let a dimension the benchmark asked to ignore contribute a
    perfect factor, and would demand a value for a dimension that -- being
    unweighted -- may legitimately have none.
    """
    resolved = weights if weights is not None else ScoreWeights.neutral()
    terms = (
        (scientific_outcome, resolved.outcome),
        (decision_quality, resolved.decision),
        (trajectory_quality, resolved.trajectory),
    )
    if any(value is None for value, weight in terms if weight > 0):
        return None

    value = 1.0
    for raw, weight in terms:
        # ``raw is None`` here only for a zero-weighted dimension; the check
        # above has already refused every weighted absence.
        if weight <= 0 or raw is None:
            continue
        value *= max(0.0, min(1.0, raw)) ** weight
    return GlobalAgentScore(
        score_version=SCORE_VERSION,
        scientific_outcome=_clamp(scientific_outcome),
        decision_quality=_clamp(decision_quality),
        trajectory_quality=_clamp(trajectory_quality),
        weights=resolved,
        value=max(0.0, min(1.0, value)),
        formula=describe_score(resolved),
        confidence=confidence,
    )


def _clamp(value: float | None) -> float:
    """Clamp a reported dimension, recording an unweighted absence as zero.

    Only reachable for a dimension the benchmark weighted at zero, which by
    definition contributes nothing to ``value``; the field is filled so the
    persisted record stays complete.
    """
    return 0.0 if value is None else max(0.0, min(1.0, value))


__all__ = [
    "DEFAULT_DECISION_PENALTY",
    "DEFAULT_TRAJECTORY_PENALTY",
    "SCORE_VERSION",
    "WEIGHT_SUM_TOLERANCE",
    "GlobalAgentScore",
    "ScoreConfidence",
    "ScoreWeights",
    "compute_global_agent_score",
    "compute_score_confidence",
    "describe_score",
]
