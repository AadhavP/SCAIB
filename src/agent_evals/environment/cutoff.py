"""Run cutoffs owned by the controller rather than by the agent.

Until now the only thing that could end a run was the step budget in the
runtime loop's ``while`` condition, plus the agent's own terminal action. Three
consequences.

**A run that had stopped making progress ran to the end of its budget.** There
was no notion of progress at the controller level, so an agent looping on the
same decision, or plateaued, spent the whole budget doing it. That is expensive
on paid models and it pollutes the trajectory record with steps that carry no
information.

**A run that had exhausted its resource budget kept stepping.**
``ConstraintMonitor`` turns a resource violation into a *failed step*
(``runtime.py``), not into a stopped run, so once ``max_runtime_seconds`` is
gone every remaining step fails in the same way and the agent burns its step
budget on guaranteed failures. The consecutive-failure cutoff below is what
ends that.

**Wall time was never bounded at all.** ``constraints.max_runtime_seconds`` is
checked against ``ResourceUsage.wall_time_seconds``, which is time inside the
*executor*. An agent that deliberates for twenty minutes between two cheap
actions reports almost no executor time, so the declared runtime budget did not
constrain the thing that actually takes the time.

Two rules the whole module is built around.

**Unmeasured progress is not absent progress.** ``dS_t`` is ``None`` whenever
two consecutive steps had no metric in common, which is normal and frequent --
a real annotation run measures five deltas across six steps. If ``None`` counted
as "no improvement", the controller would kill runs for the harness's own
blindness. :class:`StagnationDetector` therefore looks only at *measured*
deltas and reports :attr:`StagnationVerdict.UNDETERMINED` until it has enough of
them, and the controller never stops on an undetermined verdict.

**The agent may see its hard budgets and never the progress state.** Steps and
seconds remaining are facts about the harness, and an agent that cannot see them
cannot plan; ``dS_t`` is derived from the held-out reference, so exposing the
stagnation state would hand the agent a channel into exactly the information the
benchmark withholds -- and would let it defeat the detector by manufacturing an
improvement on whatever it inferred was being measured.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.benchmarks.schema import ConstraintSpecification, CutoffSpecification

#: Deltas at or below this count as no improvement. Small but non-zero: metric
#: recomputation on an unchanged dataset is not bit-identical, so an exact
#: ``> 0`` test would read floating-point noise as scientific progress and the
#: detector would never fire.
DEFAULT_STAGNATION_EPSILON = 0.01

#: Measured deltas the detector needs before it will call a run stagnant.
DEFAULT_STAGNATION_WINDOW = 3

#: Consecutive stagnant verdicts tolerated before stopping. A grace period, so
#: one flat window inside a longer productive run does not end it. Each verdict
#: costs a *measured* delta, so with the default window this is five measured flat
#: steps rather than five steps.
DEFAULT_PATIENCE_STEPS = 2

#: Times the same decision may succeed before it is a loop rather than a choice.
DEFAULT_MAX_REPEATED_DECISIONS = 3

#: Consecutive failed steps tolerated. Retrying a failure is legitimate
#: adaptation, so this is a budget rather than a ban.
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


class CutoffReason(StrEnum):
    """Which mechanism ended a run.

    Recorded rather than collapsed into a single "stopped" flag: a run that ran
    out of steps, one that looped, and one that plateaued are three different
    findings about the agent, and a scorecard that spells them the same way
    cannot say which happened.
    """

    #: The step budget was exhausted.
    MAX_STEPS = "max_steps"
    #: Wall-clock time for the whole run, including agent deliberation.
    WALL_TIME = "wall_time"
    #: Estimated monetary cost.
    COST = "cost"
    #: Cumulative model tokens.
    TOKENS = "tokens"
    #: No measured improvement above epsilon, for longer than patience allows.
    STAGNATION = "stagnation"
    #: The same decision succeeded more times than a choice plausibly needs.
    REPETITION = "repetition"
    #: Steps kept failing, which is what an exhausted resource budget looks
    #: like from inside the loop.
    CONSECUTIVE_FAILURES = "consecutive_failures"


class CutoffEnforcement(StrEnum):
    """Whether a declared cutoff could actually be checked.

    The same honesty requirement as :class:`~agent_evals.environment.execution.isolation.IsolationReport`:
    a budget that silently never fires is worse than one that was never
    declared, because the run record implies a limit that was not in force.
    """

    #: Declared, and something supplies the measurement.
    ENFORCED = "enforced"
    #: No budget declared, so nothing to enforce.
    UNDECLARED = "undeclared"
    #: Declared, but no measurement reached the controller. Cost and tokens are
    #: the realistic cases -- only some agent backends report usage at all.
    UNOBSERVABLE = "unobservable"


class StagnationVerdict(StrEnum):
    """What the measured progress window supports saying."""

    #: Enough measured deltas, and none of them cleared epsilon.
    STAGNANT = "stagnant"
    #: Enough measured deltas, and at least one cleared epsilon.
    PROGRESSING = "progressing"
    #: Too few measured deltas to say either way. Never a reason to stop.
    UNDETERMINED = "undetermined"


class CutoffBudget(BaseModel):
    """The limits a controller enforces for one run.

    Separate from the benchmark schema on purpose: the controller takes plain
    numbers so it can be exercised without constructing a benchmark, and so a
    caller-supplied ``max_steps`` can be intersected with the declared one
    before anything is enforced.
    """

    model_config = ConfigDict(extra="forbid")

    max_steps: int | None = Field(default=None, gt=0)
    max_wall_time_seconds: float | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_total_tokens: int | None = Field(default=None, gt=0)
    #: ``None`` means this run declares no stagnation cutoff, and the detector is
    #: not constructed at all. Opt-in rather than defaulted: ``dS`` is derived
    #: from the metric profile, and a profile whose domain is pinned by an
    #: unrelated defect makes a *correct* run look flat. Stopping runs on that
    #: would be the same error as stopping them on an unmeasured delta.
    stagnation_window: int | None = Field(default=None, gt=0)
    stagnation_epsilon: float = Field(default=DEFAULT_STAGNATION_EPSILON, ge=0)
    patience_steps: int = Field(default=DEFAULT_PATIENCE_STEPS, ge=0)
    #: ``None`` by default because a signature is action-plus-parameters and does
    #: not include the input state, so the same call on a changed dataset is
    #: legitimately different work. It earns its keep on the free tier, where the
    #: signature is the whole script and a repeat really is a loop.
    max_repeated_decisions: int | None = Field(default=None, gt=0)
    #: Defaulted on, unlike the two above: it cannot fire on a run whose steps
    #: succeed, and it is what stops an exhausted resource budget from spending
    #: the remaining step budget on steps that are guaranteed to fail.
    max_consecutive_failures: int | None = Field(
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES, gt=0
    )


class StepObservation(BaseModel):
    """What the controller learns from one completed step.

    Everything here is a harness observation. ``progress_delta`` is the
    evaluator-side ``dS_t`` and is the reason this object never travels back to
    the agent.
    """

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    succeeded: bool
    #: Canonical action-plus-parameters fingerprint, or ``None`` when the step
    #: had nothing stable to key on. Absent signatures are never counted as
    #: repeats: an unidentifiable step is unidentifiable, not identical.
    signature: str | None = None
    #: ``None`` when the two steps shared no comparable metric.
    progress_delta: float | None = None
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class StagnationTrace(BaseModel):
    """Why the detector reached the verdict it did."""

    model_config = ConfigDict(extra="forbid")

    verdict: StagnationVerdict
    window: int
    epsilon: float
    measured_deltas: list[float]
    considered: list[float]
    stagnant_streak: int


class CutoffDecision(BaseModel):
    """Whether to take another step, and why not."""

    model_config = ConfigDict(extra="forbid")

    stop: bool
    reason: CutoffReason | None = None
    #: Names the observed value and the budget it crossed, so a run record does
    #: not just assert that a limit was hit.
    detail: str | None = None


class CutoffReport(BaseModel):
    """Everything about the cutoff worth archiving with the run."""

    model_config = ConfigDict(extra="forbid")

    stopped: bool = False
    reason: CutoffReason | None = None
    detail: str | None = None
    budget: CutoffBudget
    steps_used: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    most_repeated_decisions: int = Field(default=0, ge=0)
    enforcement: dict[CutoffReason, CutoffEnforcement] = Field(default_factory=dict)
    stagnation: StagnationTrace | None = None


class StagnationDetector:
    """Decide whether measured scientific progress has flattened."""

    def __init__(
        self,
        *,
        window: int = DEFAULT_STAGNATION_WINDOW,
        epsilon: float = DEFAULT_STAGNATION_EPSILON,
    ) -> None:
        self.window = window
        self.epsilon = epsilon

    def evaluate(self, deltas: list[float | None]) -> StagnationTrace:
        """Judge the most recent measured deltas, ignoring unmeasured steps.

        Unmeasured steps are skipped rather than treated as zeros. Two steps
        with no metric in common produce no comparison at all, and counting that
        as a flat delta would make a run look stagnant precisely when the
        harness could see least.
        """
        measured = [delta for delta in deltas if delta is not None]
        considered = measured[-self.window :]
        if len(considered) < self.window:
            verdict = StagnationVerdict.UNDETERMINED
        elif max(considered) > self.epsilon:
            verdict = StagnationVerdict.PROGRESSING
        else:
            verdict = StagnationVerdict.STAGNANT
        return StagnationTrace(
            verdict=verdict,
            window=self.window,
            epsilon=self.epsilon,
            measured_deltas=measured,
            considered=considered,
            stagnant_streak=0,
        )


class CutoffController:
    """Decide when a run stops, from harness observations only.

    Held by the runtime manager. The agent contributes a completion *claim*,
    which is verified elsewhere against the declared artifact contract; nothing
    the agent says reaches this object.
    """

    def __init__(self, budget: CutoffBudget | None = None) -> None:
        self.budget = budget or CutoffBudget()
        # Not constructed when no window is declared, so an undeclared stagnation
        # cutoff has no detector to accidentally consult rather than a detector
        # whose verdict is ignored somewhere further down.
        self._detector = (
            StagnationDetector(
                window=self.budget.stagnation_window,
                epsilon=self.budget.stagnation_epsilon,
            )
            if self.budget.stagnation_window is not None
            else None
        )
        self._deltas: list[float | None] = []
        self._signatures: Counter[str] = Counter()
        self._steps = 0
        self._consecutive_failures = 0
        self._stagnant_streak = 0
        self._total_tokens: int | None = None
        self._cost_usd: float | None = None
        self._elapsed_seconds = 0.0
        self._stop: CutoffDecision | None = None
        self._trace: StagnationTrace | None = None

    @property
    def steps_used(self) -> int:
        """Steps the controller has been told about."""
        return self._steps

    def observe(self, observation: StepObservation) -> None:
        """Record one completed step. Never raises and never decides."""
        self._steps = max(self._steps, observation.step)
        self._deltas.append(observation.progress_delta)
        if observation.succeeded:
            self._consecutive_failures = 0
            if observation.signature is not None:
                self._signatures[observation.signature] += 1
        else:
            self._consecutive_failures += 1
        self.observe_usage(
            total_tokens=observation.total_tokens,
            cost_usd=observation.cost_usd,
        )
        if self._detector is None:
            return
        trace = self._detector.evaluate(self._deltas)
        if observation.progress_delta is not None:
            # Only a *measured* step can move the streak. The detector reads
            # measured deltas only, so a blank leaves the window untouched and
            # re-returns the previous verdict -- billing the streak for it would
            # repeat a stagnant reading once per unmeasured step and drain
            # patience on steps the harness could not measure. A real run measures
            # five deltas across six steps, so that would end correct runs for the
            # harness's own blindness, just past where the all-blank case stops
            # catching it. Within this branch, the verdict is earned: stagnant
            # extends the streak, progressing clears it, and undetermined does
            # neither, because the run may well be stagnating and the harness
            # simply cannot tell yet.
            if trace.verdict is StagnationVerdict.STAGNANT:
                self._stagnant_streak += 1
            elif trace.verdict is StagnationVerdict.PROGRESSING:
                self._stagnant_streak = 0
        # The trace is published either way, so that an armed-but-never-measurable
        # detector reports an undetermined verdict over zero measured deltas rather
        # than nothing at all. A missing trace would be indistinguishable from an
        # undeclared window, which is the one distinction this report exists to
        # make.
        self._trace = trace.model_copy(
            update={"stagnant_streak": self._stagnant_streak}
        )

    def observe_usage(
        self,
        *,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Record cumulative provider usage without inventing a scientific step.

        Planning and termination calls consume provider resources too, but they
        are not environment actions and must not advance the step/stagnation
        state. Values are monotonic because a late or cumulative report must not
        resurrect a budget that already fired.
        """
        if total_tokens is not None:
            self._total_tokens = max(self._total_tokens or 0, total_tokens)
        if cost_usd is not None:
            self._cost_usd = max(self._cost_usd or 0.0, cost_usd)

    def decide(self, *, elapsed_seconds: float) -> CutoffDecision:
        """Say whether another step may be taken.

        ``elapsed_seconds`` is passed in rather than read from a clock here so
        the controller stays pure and deterministic under test, and so the run's
        single notion of elapsed time lives in the loop that owns it.
        """
        self._elapsed_seconds = max(self._elapsed_seconds, elapsed_seconds)
        decision = self._first_breach()
        if decision.stop and self._stop is None:
            self._stop = decision
        return decision

    def _first_breach(self) -> CutoffDecision:
        """Return the first budget crossed, checking hard limits first.

        Order is deliberate. A hard budget is a fact about the run; stagnation
        and repetition are inferences from what the harness could measure. When
        both apply, the fact is the better explanation to archive.
        """
        for check in (
            self._check_steps,
            self._check_wall_time,
            self._check_tokens,
            self._check_cost,
            self._check_failures,
            self._check_repetition,
            self._check_stagnation,
        ):
            decision = check()
            if decision is not None:
                return decision
        return CutoffDecision(stop=False)

    def _check_steps(self) -> CutoffDecision | None:
        limit = self.budget.max_steps
        if limit is None or self._steps < limit:
            return None
        return CutoffDecision(
            stop=True,
            reason=CutoffReason.MAX_STEPS,
            detail=f"{self._steps} steps taken of a {limit}-step budget",
        )

    def _check_wall_time(self) -> CutoffDecision | None:
        limit = self.budget.max_wall_time_seconds
        if limit is None or self._elapsed_seconds < limit:
            return None
        return CutoffDecision(
            stop=True,
            reason=CutoffReason.WALL_TIME,
            detail=(
                f"{self._elapsed_seconds:.1f}s elapsed of a {limit:.1f}s wall-clock "
                "budget, measured across the whole run rather than inside the "
                "executor"
            ),
        )

    def _check_tokens(self) -> CutoffDecision | None:
        limit = self.budget.max_total_tokens
        if limit is None or self._total_tokens is None or self._total_tokens < limit:
            return None
        return CutoffDecision(
            stop=True,
            reason=CutoffReason.TOKENS,
            detail=f"{self._total_tokens} tokens used of a {limit}-token budget",
        )

    def _check_cost(self) -> CutoffDecision | None:
        limit = self.budget.max_cost_usd
        if limit is None or self._cost_usd is None or self._cost_usd < limit:
            return None
        return CutoffDecision(
            stop=True,
            reason=CutoffReason.COST,
            detail=f"{self._cost_usd:.4f} USD spent of a {limit:.4f} USD budget",
        )

    def _check_failures(self) -> CutoffDecision | None:
        limit = self.budget.max_consecutive_failures
        if limit is None or self._consecutive_failures < limit:
            return None
        return CutoffDecision(
            stop=True,
            reason=CutoffReason.CONSECUTIVE_FAILURES,
            detail=(
                f"{self._consecutive_failures} consecutive steps failed, which is "
                "what an exhausted resource budget looks like from inside the loop"
            ),
        )

    def _check_repetition(self) -> CutoffDecision | None:
        limit = self.budget.max_repeated_decisions
        if limit is None or not self._signatures:
            return None
        signature, count = self._signatures.most_common(1)[0]
        if count < limit:
            return None
        return CutoffDecision(
            stop=True,
            reason=CutoffReason.REPETITION,
            detail=(
                f"the same decision succeeded {count} times ({signature}), against "
                f"a limit of {limit}"
            ),
        )

    def _check_stagnation(self) -> CutoffDecision | None:
        trace = self._trace
        if trace is None or trace.verdict is not StagnationVerdict.STAGNANT:
            return None
        if self._stagnant_streak <= self.budget.patience_steps:
            return None
        return CutoffDecision(
            stop=True,
            reason=CutoffReason.STAGNATION,
            detail=(
                f"no measured scientific progress above {trace.epsilon} across the "
                f"last {trace.window} measured steps, for {self._stagnant_streak} "
                f"consecutive checks against a patience of "
                f"{self.budget.patience_steps}"
            ),
        )

    def agent_visible_budget(
        self, *, elapsed_seconds: float
    ) -> dict[str, float | int | None]:
        """The part of the budget an agent is allowed to plan against.

        Hard limits only. Progress, deltas, and the stagnation verdict are
        derived from the held-out reference and are deliberately absent: an
        agent that could read them could both infer what is being measured and
        defeat the detector by manufacturing an improvement against it.

        ``elapsed_seconds`` is a parameter for the same reason it is one on
        :meth:`decide`, plus one specific to this method: an agent plans against
        the remaining wall clock, and a reading left over from the previous
        step's decision would tell it that it has a step's worth more time than
        it does.
        """
        return {
            "steps_used": self._steps,
            "steps_remaining": _remaining(self.budget.max_steps, self._steps),
            "seconds_remaining": _remaining(
                self.budget.max_wall_time_seconds,
                max(self._elapsed_seconds, elapsed_seconds),
            ),
            "tokens_remaining": _remaining(
                self.budget.max_total_tokens, self._total_tokens
            ),
        }

    def report(self) -> CutoffReport:
        """Archive the budget, what was measured against it, and what fired."""
        stop = self._stop
        return CutoffReport(
            stopped=stop is not None,
            reason=stop.reason if stop is not None else None,
            detail=stop.detail if stop is not None else None,
            budget=self.budget,
            steps_used=self._steps,
            elapsed_seconds=self._elapsed_seconds,
            total_tokens=self._total_tokens,
            cost_usd=self._cost_usd,
            consecutive_failures=self._consecutive_failures,
            most_repeated_decisions=(
                self._signatures.most_common(1)[0][1] if self._signatures else 0
            ),
            enforcement=self._enforcement(),
            stagnation=self._trace,
        )

    def _enforcement(self) -> dict[CutoffReason, CutoffEnforcement]:
        """Say which declared budgets were actually in force.

        Tokens and cost are the ones that matter here: only some agent backends
        report usage, so a declared token budget on a backend that reports
        nothing is ``UNOBSERVABLE``. Recording that is the difference between a
        run that stayed under budget and one whose budget was never checked.
        """
        return {
            CutoffReason.MAX_STEPS: _declared(self.budget.max_steps),
            CutoffReason.WALL_TIME: _declared(self.budget.max_wall_time_seconds),
            CutoffReason.TOKENS: _observed(
                self.budget.max_total_tokens, self._total_tokens
            ),
            CutoffReason.COST: _observed(self.budget.max_cost_usd, self._cost_usd),
            CutoffReason.CONSECUTIVE_FAILURES: _declared(
                self.budget.max_consecutive_failures
            ),
            CutoffReason.REPETITION: _declared(self.budget.max_repeated_decisions),
            CutoffReason.STAGNATION: self._stagnation_enforcement(),
        }

    def _stagnation_enforcement(self) -> CutoffEnforcement:
        """Distinguish an undeclared stagnation cutoff from an unobservable one.

        Progress arrives only when two consecutive steps share a comparable
        metric, so a declared window on a run that never produced one was never
        actually checked -- which is a different fact from never having declared
        it, and the one a reader would otherwise get wrong.
        """
        if self.budget.stagnation_window is None:
            return CutoffEnforcement.UNDECLARED
        if any(delta is not None for delta in self._deltas):
            return CutoffEnforcement.ENFORCED
        return CutoffEnforcement.UNOBSERVABLE


def budget_from_specification(
    cutoff: CutoffSpecification,
    constraints: ConstraintSpecification | None = None,
    *,
    caller_max_steps: int | None = None,
) -> CutoffBudget:
    """Resolve a declared cutoff plus a caller's own step limit into one budget.

    Both step limits are *ceilings*, so the effective budget is the smaller of
    the two and neither can raise the other. A caller passing ``--max-steps 50``
    to a benchmark that declares 10 gets 10; a benchmark declaring 100 run with
    ``--max-steps 5`` gets 5. Taking the maximum, or letting the caller win,
    would let a run exceed a limit its own benchmark declared.

    Wall time falls back to ``constraints.max_runtime_seconds`` when the cutoff
    block does not declare its own. The two measure different things -- see
    :class:`~agent_evals.benchmarks.schema.CutoffSpecification` -- but the
    constraint is a ceiling the benchmark author already accepted, and the
    alternative is an unbounded clock.
    """
    wall_time = cutoff.max_wall_time_seconds
    if wall_time is None and constraints is not None:
        runtime = constraints.max_runtime_seconds
        wall_time = None if runtime is None else float(runtime)
    return CutoffBudget(
        max_steps=_stricter(cutoff.max_steps, caller_max_steps),
        max_wall_time_seconds=wall_time,
        max_cost_usd=cutoff.max_cost_usd,
        max_total_tokens=cutoff.max_total_tokens,
        stagnation_window=cutoff.stagnation_window,
        stagnation_epsilon=cutoff.stagnation_epsilon,
        patience_steps=cutoff.patience_steps,
        max_repeated_decisions=cutoff.max_repeated_decisions,
        max_consecutive_failures=cutoff.max_consecutive_failures,
    )


def _stricter(declared: int | None, requested: int | None) -> int | None:
    """Return the tighter of two ceilings, or whichever one exists."""
    if declared is None:
        return requested
    if requested is None:
        return declared
    return min(declared, requested)


def _declared(limit: float | int | None) -> CutoffEnforcement:
    """A limit needing no external measurement is in force once declared."""
    if limit is None:
        return CutoffEnforcement.UNDECLARED
    return CutoffEnforcement.ENFORCED


def _observed(
    limit: float | int | None, measurement: float | int | None
) -> CutoffEnforcement:
    """A limit is only in force when something reports the quantity it bounds."""
    if limit is None:
        return CutoffEnforcement.UNDECLARED
    if measurement is None:
        return CutoffEnforcement.UNOBSERVABLE
    return CutoffEnforcement.ENFORCED


def _remaining(
    limit: float | int | None, used: float | int | None
) -> float | int | None:
    """Headroom left, or ``None`` when there is no limit or no measurement."""
    if limit is None or used is None:
        return None
    return max(type(limit)(0), limit - used)


__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_FAILURES",
    "DEFAULT_MAX_REPEATED_DECISIONS",
    "DEFAULT_PATIENCE_STEPS",
    "DEFAULT_STAGNATION_EPSILON",
    "DEFAULT_STAGNATION_WINDOW",
    "CutoffBudget",
    "CutoffController",
    "CutoffDecision",
    "CutoffEnforcement",
    "CutoffReason",
    "CutoffReport",
    "StagnationDetector",
    "StagnationTrace",
    "StagnationVerdict",
    "StepObservation",
    "budget_from_specification",
]
