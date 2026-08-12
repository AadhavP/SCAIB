"""The keys under which scientific progress travels on a reward record.

One module because three subsystems have to agree on these strings exactly and a
disagreement fails *silently* in both directions. The reward evaluator writes
them; the cutoff controller reads ``dS`` to decide whether a run has stopped
making progress; and the leakage tests assert the prefix never appears in an
agent-visible payload. A rename on the writing side would leave the controller
seeing no deltas at all -- so a stagnation cutoff that never fires -- and would
leave the leakage assertion passing because it was looking for a prefix nobody
emits any more.

Same reasoning as :mod:`agent_evals.core.reference_columns`,
:mod:`agent_evals.core.intent_parameters`, and
:mod:`agent_evals.core.decision_components`. It lives in ``core`` rather than
beside the evaluator so the controller can read one string without importing the
whole metric engine.
"""

from __future__ import annotations

#: Namespace marking a key as evaluator-side evidence rather than an observable
#: the delegate scored the step on. Asserted absent from agent-visible payloads.
PROGRESS_PREFIX = "progress."

#: ``S_t``, the scientific state score for the step. Absent when unmeasured.
PROGRESS_STATE_KEY = f"{PROGRESS_PREFIX}scientific_state"

#: ``dS_t``. Absent -- not zero -- when the two steps shared no comparable
#: metric, which is what keeps the cutoff controller from reading the harness's
#: blindness as a lack of progress.
PROGRESS_DELTA_KEY = f"{PROGRESS_PREFIX}delta"

#: The inferred pipeline stage, or ``None`` when nothing observed could name it.
PROGRESS_STAGE_KEY = f"{PROGRESS_PREFIX}stage"

#: Which metrics both steps had in common, i.e. what ``dS_t`` was computed over.
PROGRESS_COMPARABLE_KEY = f"{PROGRESS_PREFIX}comparable_metrics"

#: Why a progress number is missing or partial, when something can say.
PROGRESS_LIMITATIONS_KEY = f"{PROGRESS_PREFIX}limitations"


__all__ = [
    "PROGRESS_COMPARABLE_KEY",
    "PROGRESS_DELTA_KEY",
    "PROGRESS_LIMITATIONS_KEY",
    "PROGRESS_PREFIX",
    "PROGRESS_STAGE_KEY",
    "PROGRESS_STATE_KEY",
]
