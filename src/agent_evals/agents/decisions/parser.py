"""Normalize an agent's stated reasoning into canonical decision metadata.

Every runtime turn carries a free-form ``reasoning_metadata`` mapping, and
``decision_cascade_from_episode`` reads a specific vocabulary out of it. Nothing
sat between the two, which produced three failures that all look like the
benchmark working:

**Silent loss.** An agent that names its methodological choice in a key the
cascade does not read has its reasoning dropped, and scores as though it had
explained nothing. The score then measures protocol compliance rather than
scientific reasoning.

**Silent corruption.** ``evidence_used: "the elbow plot"`` iterates a string,
so the cascade recorded fifteen pieces of evidence named ``t``, ``h``, ``e``.
That is worse than no evidence: it is fabricated evidence, and it aggregates
into a decision score no reader would know to distrust.

**Crashes blamed on the agent.** ``expected_effect: "large improvement"`` reaches
``.items()`` and raises ``AttributeError``, which surfaces as a harness failure
attributed to the run rather than as a malformed response by the agent.

So this layer coerces what it can, records what it could not, and never
discards anything. A response whose reasoning cannot be parsed is recorded as
``MALFORMED`` rather than dropped, because "the agent explained itself badly"
and "the agent did not explain itself" are different findings, and both are
different from "the agent reasoned well".

One rule is adversarial rather than tidy: the extraction verdict is computed
here and an incoming key that tries to state it is discarded. An agent that
could set its own ``decision_extraction_quality`` would be grading its own
paper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Metadata key holding this layer's verdict on the agent's stated reasoning.
DECISION_QUALITY_KEY = "decision_extraction_quality"

#: Metadata key holding the human-readable reasons behind that verdict.
DECISION_FINDINGS_KEY = "decision_extraction_findings"

#: Metadata key preserving prose that arrived where a decision object belongs.
DECISION_TEXT_KEY = "decision_text"

#: Keys this layer owns. An agent-supplied value for any of them is discarded,
#: since they are the record of how well the agent was understood.
_RESERVED_KEYS = frozenset({DECISION_QUALITY_KEY, DECISION_FINDINGS_KEY})

#: Nested block name used by the structured decision response protocol.
_DECISION_BLOCK = "decision"

_TEXT_KEYS = (
    "decision_type",
    "decision_category",
    "intent",
    "hypothesis",
    "plan_reference",
    "method",
    "method_id",
    "implementation",
    "parent_decision_id",
    "rationale",
)

_TEXT_LIST_KEYS = (
    "evidence_used",
    "input_artifacts",
    "alternatives_considered",
    "predecessor_decision_ids",
    "dependency_decision_ids",
    "source_event_ids",
)

_FLOAT_MAP_KEYS = ("expected_effect",)

_MAP_KEYS = ("downstream_dependency", "state_claim")

_UNIT_FLOAT_KEYS = ("confidence",)

#: Free-form; the cascade stores it verbatim as the decision's selected value.
_PASSTHROUGH_KEYS = ("selected_value",)

#: The full vocabulary this layer understands, for tests and documentation.
CANONICAL_DECISION_KEYS = frozenset(
    (
        *_TEXT_KEYS,
        *_TEXT_LIST_KEYS,
        *_FLOAT_MAP_KEYS,
        *_MAP_KEYS,
        *_UNIT_FLOAT_KEYS,
        *_PASSTHROUGH_KEYS,
    )
)


class DecisionQuality(StrEnum):
    """How completely the agent's stated reasoning could be understood.

    This grades the *response*, not the science. ``STRUCTURED`` says the agent
    answered in the protocol, which is a precondition for scoring its reasoning
    and not itself evidence that the reasoning was good.
    """

    #: Nothing the agent supplied was lost. A field may still have needed
    #: coercion, which is recorded as a finding -- the verdict tracks loss, the
    #: findings track everything worth knowing.
    STRUCTURED = "structured"
    #: Some fields were kept and others could not be read at all.
    PARTIAL = "partial"
    #: Reasoning was offered but none of it could be read.
    MALFORMED = "malformed"
    #: The agent stated no reasoning at all. Not an error, but not evidence.
    ABSENT = "absent"


class ExtractedDecision(BaseModel):
    """Canonical decision metadata plus the record of what was understood."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: dict[str, Any] = Field(default_factory=dict)
    quality: DecisionQuality = DecisionQuality.ABSENT
    findings: list[str] = Field(default_factory=list)

    @property
    def observable(self) -> bool:
        """Return whether any of the agent's reasoning survived extraction."""
        return self.quality in (DecisionQuality.STRUCTURED, DecisionQuality.PARTIAL)


def extract_decision(reasoning: Mapping[str, Any] | None) -> ExtractedDecision:
    """Coerce stated reasoning to canonical metadata, recording what failed."""
    if not reasoning:
        return ExtractedDecision(quality=DecisionQuality.ABSENT)

    findings: list[str] = []
    raw, block_findings, prose = _merge_decision_block(reasoning)
    findings.extend(block_findings)

    metadata: dict[str, Any] = {}
    claimed = 0
    parsed = 0
    for key, value in raw.items():
        if key in _RESERVED_KEYS:
            findings.append(
                f"discarded agent-supplied '{key}': the extraction verdict is "
                "recorded by the harness, not claimed by the agent"
            )
            continue
        if key not in CANONICAL_DECISION_KEYS:
            # Kept verbatim. An unrecognized key may be a protocol the harness
            # has not learned yet, and dropping it would erase the evidence
            # that the agent tried to say something.
            metadata[key] = value
            continue
        claimed += 1
        coerced, note = _coerce(key, value)
        if note is not None:
            findings.append(note)
        if coerced is _REJECTED:
            continue
        metadata[key] = coerced
        parsed += 1

    if prose is not None:
        metadata[DECISION_TEXT_KEY] = prose

    quality = _grade(claimed=claimed, parsed=parsed, prose=prose is not None)
    metadata[DECISION_QUALITY_KEY] = quality.value
    if findings:
        metadata[DECISION_FINDINGS_KEY] = list(findings)
    return ExtractedDecision(metadata=metadata, quality=quality, findings=findings)


def _grade(*, claimed: int, parsed: int, prose: bool) -> DecisionQuality:
    """Decide the verdict from how much of the stated reasoning survived."""
    if parsed and parsed == claimed and not prose:
        return DecisionQuality.STRUCTURED
    if parsed:
        return DecisionQuality.PARTIAL
    if claimed or prose:
        return DecisionQuality.MALFORMED
    return DecisionQuality.ABSENT


def _merge_decision_block(
    reasoning: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], str | None]:
    """Flatten the protocol's nested ``decision`` block over the top level.

    Top-level keys win, because that is the shape the existing runtimes emit and
    a benchmark should not change the meaning of a response it already accepted.
    """
    findings: list[str] = []
    merged: dict[str, Any] = {}
    block = reasoning.get(_DECISION_BLOCK)
    prose: str | None = None
    if isinstance(block, Mapping):
        merged.update({str(key): value for key, value in block.items()})
    elif isinstance(block, str) and block.strip():
        prose = block.strip()
        findings.append(
            "the 'decision' block was prose rather than a decision object; "
            "its text is preserved but none of its content could be scored"
        )
    elif block is not None:
        findings.append(
            f"ignored 'decision' block of type {type(block).__name__}; "
            "expected a mapping of decision fields"
        )
    overlap = sorted(set(merged) & set(reasoning) - {_DECISION_BLOCK})
    if overlap:
        findings.append(
            "top-level value used for key(s) also present in the nested "
            f"'decision' block: {', '.join(overlap)}"
        )
    merged.update(
        {str(key): value for key, value in reasoning.items() if key != _DECISION_BLOCK}
    )
    return merged, findings, prose


class _Rejected:
    """Sentinel marking a value that could not be coerced at all."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<rejected>"


_REJECTED = _Rejected()


def _coerce(key: str, value: Any) -> tuple[Any, str | None]:
    """Coerce one canonical field, returning the value and any finding."""
    if key in _PASSTHROUGH_KEYS:
        return value, None
    if key in _TEXT_KEYS:
        return _as_text(key, value)
    if key in _TEXT_LIST_KEYS:
        return _as_text_list(key, value)
    if key in _UNIT_FLOAT_KEYS:
        return _as_unit_float(key, value)
    if key in _FLOAT_MAP_KEYS:
        return _as_float_map(key, value)
    return _as_map(key, value)


def _as_text(key: str, value: Any) -> tuple[Any, str | None]:
    """Accept a scalar as text; refuse a container where a name belongs."""
    if value is None:
        return _REJECTED, None
    if isinstance(value, str):
        return value, None
    if isinstance(value, (int, float, bool)):
        return str(value), None
    return _REJECTED, (
        f"rejected '{key}' of type {type(value).__name__}; expected a single name or phrase"
    )


def _as_text_list(key: str, value: Any) -> tuple[Any, str | None]:
    """Accept a sequence of names, reading a bare string as one name.

    Iterating a bare string would record each character as a separate item,
    which invents evidence the agent never offered.
    """
    if value is None:
        return _REJECTED, None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return _REJECTED, None
        return [text], (
            f"read '{key}' as a single item; the protocol expects a list of names"
        )
    if isinstance(value, Sequence):
        return [str(item) for item in value], None
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value), (
            f"sorted '{key}' from an unordered set; declared order was not recoverable"
        )
    return _REJECTED, (
        f"rejected '{key}' of type {type(value).__name__}; expected a list of names"
    )


def _as_unit_float(key: str, value: Any) -> tuple[Any, str | None]:
    """Accept a confidence in [0, 1]; refuse anything that is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _REJECTED, (
            f"rejected '{key}' of type {type(value).__name__}; expected a number in [0, 1]"
        )
    number = float(value)
    if not 0.0 <= number <= 1.0:
        return _REJECTED, f"rejected '{key}' value {number}; expected a number in [0, 1]"
    return number, None


def _as_float_map(key: str, value: Any) -> tuple[Any, str | None]:
    """Accept a mapping of named numeric effects, dropping non-numeric entries."""
    if not isinstance(value, Mapping):
        return _REJECTED, (
            f"rejected '{key}' of type {type(value).__name__}; "
            "expected a mapping of named numeric effects"
        )
    numeric: dict[str, float] = {}
    dropped: list[str] = []
    for name, entry in value.items():
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            dropped.append(str(name))
            continue
        numeric[str(name)] = float(entry)
    if not numeric:
        return _REJECTED, (
            f"rejected '{key}'; none of its entries were numeric "
            f"({', '.join(sorted(dropped)) or 'it was empty'})"
        )
    note = (
        f"dropped non-numeric '{key}' entry/entries: {', '.join(sorted(dropped))}"
        if dropped
        else None
    )
    return numeric, note


def _as_map(key: str, value: Any) -> tuple[Any, str | None]:
    """Accept an arbitrary mapping, preserved verbatim for later verification."""
    if not isinstance(value, Mapping):
        return _REJECTED, (
            f"rejected '{key}' of type {type(value).__name__}; expected a mapping"
        )
    return {str(name): entry for name, entry in value.items()}, None


__all__ = [
    "CANONICAL_DECISION_KEYS",
    "DECISION_FINDINGS_KEY",
    "DECISION_QUALITY_KEY",
    "DECISION_TEXT_KEY",
    "DecisionQuality",
    "ExtractedDecision",
    "extract_decision",
]
