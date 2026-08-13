"""The intent parameters that carry execution mechanics rather than method.

Two subsystems must agree on these names exactly and neither can detect
disagreement. The executor *reads* them to find the source to run, the language
to run it in, and the artifacts the agent says it will produce. Decision
extraction must *skip* them, because a parameter decision is a record of a
methodological choice and an entire script is not one -- left in, every
free-execution step emits a parameter decision whose selected value is the
agent's whole program, which silently pollutes the decision score with noise no
reader would recognise as noise.

They live in ``core`` for the same reason the reference-column vocabulary does:
it is the only leaf package, and the alternative homes are cyclic. Keeping the
executor as the definition point would make decision extraction import the
container and subprocess machinery it has no business knowing about.
"""

from __future__ import annotations

#: Intent parameter carrying the source the agent wants executed.
CODE_PARAMETER = "code"

#: Intent parameter naming the language the source is written in.
LANGUAGE_PARAMETER = "language"

#: Intent parameter declaring the artifacts the source will produce.
PRODUCES_PARAMETER = "produces"

#: Parameters that are execution mechanics, not methodological choices.
EXECUTION_PARAMETERS = frozenset(
    {CODE_PARAMETER, LANGUAGE_PARAMETER, PRODUCES_PARAMETER}
)

__all__ = [
    "CODE_PARAMETER",
    "EXECUTION_PARAMETERS",
    "LANGUAGE_PARAMETER",
    "PRODUCES_PARAMETER",
]
