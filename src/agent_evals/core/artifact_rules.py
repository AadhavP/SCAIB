"""The vocabulary of artifact validation rules a benchmark may declare.

``ValidationRule.rule`` is written as English in the benchmark YAML, which is
what makes a benchmark readable by the scientist who has to trust it.  Until now
that English was never parsed, so ``validated`` on an artifact record meant only
that a file existed.  This module turns the prose into a small closed vocabulary
that can actually be evaluated.

The parser is deliberately strict and anchored.  A rule it cannot read raises
:class:`UnparseableValidationRule`, and the benchmark loader turns that into a
load-time integrity error rather than a runtime status.  The distinction is who
gets blamed: an unreadable rule is a defect in the benchmark, and charging it to
the agent as a failed validation would let a typo in the YAML look like bad
science.

This lives in ``core`` because the benchmark loader and the artifact validator
must agree on the vocabulary exactly, and they sit in packages that import each
other's neighbours -- the same reason :mod:`agent_evals.core.reference_columns`
and :mod:`agent_evals.core.intent_parameters` are here.  It imports nothing from
``agent_evals``, and must stay that way.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class UnparseableValidationRule(ValueError):
    """Raised when a declared rule is not in the supported vocabulary."""


class RuleKind(StrEnum):
    """The checks a benchmark can ask of an artifact."""

    #: Named columns are all present in a tabular artifact.
    COLUMNS_INCLUDE = "columns_include"
    #: Every value in the named target is finite: no NaN, no infinity.
    FINITE_VALUES = "finite_values"
    #: The named column holds at least ``minimum`` distinct values.
    DISTINCT_VALUES_AT_LEAST = "distinct_values_at_least"
    #: The named column is fully populated and drawn from a named vocabulary.
    NON_NULL_IN_VOCABULARY = "non_null_in_vocabulary"


class ParsedRule(BaseModel):
    """One declared rule, in a form something can evaluate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RuleKind
    #: Verbatim source text, kept so a finding can quote what was asked.
    source: str = Field(min_length=1)
    #: What the rule is about: a column name, ``X``, an ``obsm`` key, or an
    #: ``obs.<column>`` path. Empty when the rule addresses the artifact's whole
    #: numeric payload rather than one named part of it.
    target: str = ""
    columns: tuple[str, ...] = ()
    minimum: int | None = None
    #: Name of the value list this rule's vocabulary must be resolved from, which
    #: the benchmark supplies as an action parameter rather than inline.
    vocabulary: str = ""


_COLUMN_SEPARATOR = ","

#: Ordered because the first match wins, and anchored at both ends because a
#: pattern that matched a prefix would silently ignore the rest of the sentence
#: -- turning "columns include a,b unless empty" into an unconditional check.
_PATTERNS: tuple[tuple[re.Pattern[str], RuleKind], ...] = (
    (
        re.compile(r"^columns\s+include\s+(?P<columns>[\w.\s,]+)$", re.IGNORECASE),
        RuleKind.COLUMNS_INCLUDE,
    ),
    (
        re.compile(
            r"^(?P<target>[\w.]+)\s+contains\s+no\s+NaN\s+or\s+infinite\s+values$",
            re.IGNORECASE,
        ),
        RuleKind.FINITE_VALUES,
    ),
    (
        re.compile(r"^all\s+(?P<target>[\w.]+)\s+values\s+are\s+finite$", re.IGNORECASE),
        RuleKind.FINITE_VALUES,
    ),
    (
        re.compile(
            r"^(?P<target>[\w.]+)\s+column\s+has\s+(?P<minimum>\d+)\s+or\s+more\s+"
            r"distinct\s+values$",
            re.IGNORECASE,
        ),
        RuleKind.DISTINCT_VALUES_AT_LEAST,
    ),
    (
        re.compile(
            r"^(?P<target>[\w.]+)\s+is\s+non-null\s+and\s+belongs\s+to\s+"
            r"(?P<vocabulary>[\w.]+)$",
            re.IGNORECASE,
        ),
        RuleKind.NON_NULL_IN_VOCABULARY,
    ),
)

#: Word stems a ``FINITE_VALUES`` target may use to mean "this artifact's own
#: numbers" rather than a named part of a container. ``all embedding values are
#: finite`` is about a parquet file that *is* the embedding, so there is nothing
#: to look up inside it. Compared after stripping a trailing plural ``s``.
_WHOLE_PAYLOAD_STEMS = frozenset({"embedding", "value", "data", "coordinate"})


def _split_columns(raw: str) -> tuple[str, ...]:
    """Split a comma-separated column list, rejecting an empty entry."""
    names = tuple(part.strip() for part in raw.split(_COLUMN_SEPARATOR))
    if not names or any(not name for name in names):
        raise UnparseableValidationRule(
            f"column list '{raw.strip()}' has an empty entry; "
            "write names separated by single commas"
        )
    return names


def is_whole_payload_target(target: str) -> bool:
    """Whether a finiteness target means the artifact's own numbers.

    Exposed because the evaluator and the parser must agree: a target the parser
    treats as a name and the evaluator treats as the whole payload would check
    something nobody asked about and report it as the declared rule.
    """
    return target.lower().rstrip("s") in _WHOLE_PAYLOAD_STEMS


def parse_validation_rule(rule: str) -> ParsedRule:
    """Parse one declared rule, or explain why it is not in the vocabulary.

    Raises :class:`UnparseableValidationRule` rather than returning ``None`` so a
    caller cannot forget to handle the failure: an unparsed rule that quietly
    became "no rule" would report an unchecked artifact as a validated one, which
    is the exact failure this module exists to remove.
    """
    text = " ".join(rule.split())
    if not text:
        raise UnparseableValidationRule("a validation rule may not be blank")
    for pattern, kind in _PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        groups = match.groupdict()
        target = (groups.get("target") or "").strip()
        if kind is RuleKind.FINITE_VALUES and is_whole_payload_target(target):
            target = ""
        minimum = groups.get("minimum")
        return ParsedRule(
            kind=kind,
            source=rule,
            target=target,
            columns=_split_columns(groups["columns"]) if "columns" in groups else (),
            minimum=int(minimum) if minimum is not None else None,
            vocabulary=(groups.get("vocabulary") or "").strip(),
        )
    raise UnparseableValidationRule(
        f"rule '{text}' is not in the supported vocabulary; write one of: "
        "'columns include a,b', '<target> contains no NaN or infinite values', "
        "'all <target> values are finite', "
        "'<column> column has N or more distinct values', "
        "'<column> is non-null and belongs to <vocabulary>'"
    )


__all__ = [
    "ParsedRule",
    "RuleKind",
    "UnparseableValidationRule",
    "is_whole_payload_target",
    "parse_validation_rule",
]
