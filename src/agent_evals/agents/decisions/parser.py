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

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExtractionMode(StrEnum):
    """How the benchmark converted one agent boundary response into a decision."""

    STRUCTURED = "structured"
    JSON_TEXT = "json_text"
    FREE_TEXT = "free_text"


class ResponseExtractionEvidence(BaseModel):
    """Auditable, non-reasoning provenance for a normalized agent response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ExtractionMode
    source: str = "agent_boundary"
    raw_sha256: str = Field(min_length=64, max_length=64)
    raw_length: int = Field(ge=0)
    raw_content_retained: bool = False
    extracted_fields: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)

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
    # ``decision_text`` is emitted by this extractor when a nested decision was
    # prose. A second normalization pass is used for legacy/direct intents, so
    # recover that text here instead of downgrading a recorded malformed
    # decision to ``ABSENT``. This makes extraction idempotent while still never
    # trusting an agent-authored quality verdict.
    stored_text = reasoning.get(DECISION_TEXT_KEY)
    if prose is None and isinstance(stored_text, str) and stored_text.strip():
        prose = stored_text.strip()
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


_MAX_EXTRACTED_TEXT = 2_000
_ACTION_NAME_PATTERN = re.compile(
    r"(?:action(?:_type)?|operation|next[ _-]+action)\s*[:=]\s*[`\"']?"
    r"([A-Za-z][A-Za-z0-9_.-]*)",
    re.IGNORECASE,
)
_METHOD_PATTERN = re.compile(
    r"(?:method|algorithm|implementation)\s*[:=]\s*[`\"']?([^\n,;`\"']+)",
    re.IGNORECASE,
)
_CONFIDENCE_PATTERN = re.compile(
    r"confidence\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)",
    re.IGNORECASE,
)
_EVIDENCE_PATTERN = re.compile(
    r"(?:evidence(?:_used)?|based on)\s*[:=]\s*([^\n]+)",
    re.IGNORECASE,
)


def extract_action_response(
    response: Any,
    *,
    available_actions: Sequence[str] = (),
) -> tuple[dict[str, Any], ResponseExtractionEvidence]:
    """Convert structured or free-form boundary output into one action.

    Level-0 agents are allowed to return text, but text is never treated as
    authoritative state. This extractor only produces a candidate action; the
    environment still validates and executes it, and the returned evidence says
    whether the candidate was structured, JSON-in-text, or inferred from prose.
    The raw response is hashed rather than persisted so the benchmark records
    provenance without turning an arbitrary agent reply into retained private
    chain-of-thought.
    """
    raw = _serialize_response(response)
    raw_bytes = raw.encode("utf-8")
    payload, mode, fields, findings = _extract_action_payload(
        response,
        available_actions=available_actions,
    )
    evidence = ResponseExtractionEvidence(
        mode=mode,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_length=len(raw_bytes),
        extracted_fields=sorted(set(fields)),
        findings=findings,
    )
    return payload, evidence


def response_evidence(
    response: Any,
    *,
    mode: ExtractionMode,
    source: str = "agent_boundary",
    extracted_fields: Sequence[str] = (),
    findings: Sequence[str] = (),
) -> ResponseExtractionEvidence:
    """Hash an observable terminal response without trying to invent an action.

    Terminal submissions have a different schema from actions, so routing them
    through :func:`extract_action_response` would incorrectly reject a perfectly
    valid summary. They still need the same provenance contract: the harness
    records how the response was represented and hashes its bytes, while never
    retaining arbitrary private model output.
    """
    raw = _serialize_response(response)
    raw_bytes = raw.encode("utf-8")
    return ResponseExtractionEvidence(
        mode=mode,
        source=source,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_length=len(raw_bytes),
        raw_content_retained=False,
        extracted_fields=sorted(set(extracted_fields)),
        findings=list(findings),
    )


def _serialize_response(response: Any) -> str:
    """Create stable bytes for response provenance without requiring JSON input."""
    if isinstance(response, str):
        return response
    if hasattr(response, "model_dump"):
        response = response.model_dump(mode="json")
    try:
        return json.dumps(response, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(response)


def _extract_action_payload(
    response: Any,
    *,
    available_actions: Sequence[str],
) -> tuple[dict[str, Any], ExtractionMode, list[str], list[str]]:
    """Extract an action and retain only public, bounded reasoning metadata."""
    if isinstance(response, Mapping):
        return _extract_mapping(response, available_actions=available_actions)
    if isinstance(response, str):
        text = response.strip()
        if not text:
            raise ValueError("agent response was empty; an action is required")
        embedded = _embedded_json(text)
        if embedded is not None:
            payload, _, fields, findings = _extract_mapping(
                embedded,
                available_actions=available_actions,
            )
            findings.insert(0, "parsed a JSON object embedded in a text response")
            return payload, ExtractionMode.JSON_TEXT, fields, findings
        return _extract_free_text(text, available_actions=available_actions)
    raise ValueError(
        f"agent response must be a JSON object or text, got {type(response).__name__}"
    )


def _extract_mapping(
    response: Mapping[str, Any],
    *,
    available_actions: Sequence[str],
) -> tuple[dict[str, Any], ExtractionMode, list[str], list[str]]:
    """Handle direct, nested, and decision-object response shapes."""
    findings: list[str] = []
    fields: list[str] = []
    nested = response.get("action") or response.get("next_action")
    if isinstance(nested, Mapping):
        source = dict(nested)
        for key in ("usage", "plan_update", "state_claim", "next_step"):
            if key not in source and key in response:
                source[key] = response[key]
        if "reasoning_metadata" not in source and isinstance(
            response.get("reasoning_metadata"), Mapping
        ):
            source["reasoning_metadata"] = response["reasoning_metadata"]
        payload = _canonical_action(source, available_actions, fields, findings)
        return payload, ExtractionMode.STRUCTURED, fields, findings

    if isinstance(nested, str):
        source = dict(response)
        source["action_type"] = nested
        payload = _canonical_action(source, available_actions, fields, findings)
        findings.append("read the action name from the top-level action string")
        return payload, ExtractionMode.STRUCTURED, fields, findings

    decision = response.get("decision")
    if isinstance(decision, Mapping):
        source = dict(response)
        source.update(
            {
                key: value
                for key, value in decision.items()
                if key not in source
            }
        )
        source.setdefault("reasoning_metadata", {"decision": dict(decision)})
        payload = _canonical_action(source, available_actions, fields, findings)
        return payload, ExtractionMode.STRUCTURED, fields, findings

    if _action_name(response, available_actions) is not None:
        payload = _canonical_action(response, available_actions, fields, findings)
        return payload, ExtractionMode.STRUCTURED, fields, findings

    for key in ("text", "content", "message", "response"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            payload, mode, nested_fields, nested_findings = _extract_action_payload(
                value,
                available_actions=available_actions,
            )
            fields.extend(nested_fields)
            findings.extend(nested_findings)
            findings.insert(0, f"extracted the action from response field '{key}'")
            return payload, mode, fields, findings

    raise ValueError(
        "agent JSON response did not contain action_type, action, decision, or text"
    )


def _canonical_action(  # noqa: C901
    response: Mapping[str, Any],
    available_actions: Sequence[str],
    fields: list[str],
    findings: list[str],
) -> dict[str, Any]:
    """Normalize one mapping without accepting an agent-authored verdict."""
    action_name = _action_name(response, available_actions)
    if action_name is None:
        raise ValueError("agent response did not name an action")
    fields.append("action_type")
    raw_parameters = response.get("parameters", {})
    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, Mapping):
        findings.append(
            f"replaced non-object parameters of type {type(raw_parameters).__name__} with an empty object"
        )
        raw_parameters = {}
    parameters = {str(key): value for key, value in raw_parameters.items()}
    if "parameters" in response:
        fields.append("parameters")
    reasoning = response.get("reasoning_metadata")
    reasoning_metadata = dict(reasoning) if isinstance(reasoning, Mapping) else {}
    decision = response.get("decision")
    if isinstance(decision, Mapping):
        reasoning_metadata.setdefault("decision", dict(decision))
        fields.append("reasoning_metadata")
    for key in ("summary", "explanation", "rationale"):
        if key in response and key not in reasoning_metadata:
            reasoning_metadata[key] = response[key]
            fields.append(key)
    payload: dict[str, Any] = {
        "action_type": action_name,
        "parameters": parameters,
        "reasoning_metadata": reasoning_metadata,
    }
    for key in ("usage", "plan_update"):
        if response.get(key) is not None:
            payload[key] = response[key]
    for key in ("state_claim", "next_step"):
        value = response.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            payload[key] = dict(value)
            fields.append(key)
        else:
            findings.append(
                f"ignored non-object '{key}' of type {type(value).__name__}"
            )
    return payload


def _action_name(
    response: Mapping[str, Any],
    available_actions: Sequence[str],
) -> str | None:
    """Read an explicit action name, with a conservative terminal fallback."""
    for key in ("action_type", "action_id", "operation", "action"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    decision = response.get("decision")
    if isinstance(decision, Mapping):
        for key in ("action_type", "action_id", "operation", "action"):
            value = decision.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _embedded_json(text: str) -> dict[str, Any] | None:
    """Parse a fenced or embedded JSON object, returning ``None`` for prose."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _extract_free_text(  # noqa: C901
    text: str,
    *,
    available_actions: Sequence[str],
) -> tuple[dict[str, Any], ExtractionMode, list[str], list[str]]:
    """Extract only explicitly named actions from a bounded public text reply."""
    findings = [
        "action was inferred from free-form text; fields not explicitly named were not invented"
    ]
    bounded = text[:_MAX_EXTRACTED_TEXT]
    action_match = _ACTION_NAME_PATTERN.search(bounded)
    action_name = action_match.group(1) if action_match else None
    if action_name is None:
        normalized = bounded.casefold()
        candidates = sorted(
            (str(action) for action in available_actions if str(action).strip()),
            key=len,
            reverse=True,
        )
        action_name = next(
            (
                candidate
                for candidate in candidates
                if re.search(rf"(?<![\w-]){re.escape(candidate.casefold())}(?![\w-])", normalized)
            ),
            None,
        )
        if action_name is not None:
            findings.append("matched the longest available action name in the text")
    if action_name is None and re.search(r"\b(done|complete|finished|terminate)\b", bounded, re.I):
        action_name = "terminate"
        findings.append("mapped an explicit completion phrase to the terminal action")
    if action_name is None:
        raise ValueError(
            "free-form agent response did not explicitly name a legal action"
        )

    fields = ["action_type", "reasoning_metadata"]
    decision: dict[str, Any] = {"rationale": bounded}
    method_match = _METHOD_PATTERN.search(bounded)
    if method_match:
        decision["method"] = method_match.group(1).strip()
        fields.append("method")
    confidence_match = _CONFIDENCE_PATTERN.search(bounded)
    if confidence_match:
        decision["confidence"] = float(confidence_match.group(1))
        fields.append("confidence")
    evidence_match = _EVIDENCE_PATTERN.search(bounded)
    if evidence_match:
        decision["evidence_used"] = [
            item.strip()
            for item in re.split(r"[,;]", evidence_match.group(1))
            if item.strip()
        ]
        fields.append("evidence_used")
    parameters: dict[str, Any] = {}
    parameter_match = re.search(
        r"parameters?\s*[:=]\s*(\{.*?\})", bounded, re.IGNORECASE | re.DOTALL
    )
    if parameter_match:
        try:
            parsed = json.loads(parameter_match.group(1))
            if isinstance(parsed, Mapping):
                parameters = {str(key): value for key, value in parsed.items()}
                fields.append("parameters")
            else:
                findings.append("ignored a non-object free-text parameters block")
        except json.JSONDecodeError:
            findings.append("could not parse the free-text parameters block as JSON")
    return (
        {
            "action_type": action_name,
            "parameters": parameters,
            "reasoning_metadata": {"decision": decision, "summary": bounded},
        },
        ExtractionMode.FREE_TEXT,
        fields,
        findings,
    )


__all__ = [
    "CANONICAL_DECISION_KEYS",
    "DECISION_FINDINGS_KEY",
    "DECISION_QUALITY_KEY",
    "DECISION_TEXT_KEY",
    "DecisionQuality",
    "ExtractedDecision",
    "ExtractionMode",
    "ResponseExtractionEvidence",
    "extract_action_response",
    "extract_decision",
    "response_evidence",
]
