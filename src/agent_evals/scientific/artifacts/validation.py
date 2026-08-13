"""Evaluate a produced artifact against the rules its benchmark declared.

``validated`` on an artifact record used to mean "a file exists at this path",
which is not validation -- an empty file passed.  It is also *scored*, by
``evaluators/builtin.artifact_validity`` and by ``evaluation/trajectory``, so the
gap was not cosmetic: the two execution tiers set the bit differently, and the
free tier was being penalised for setting it honestly.  This module is what the
bit now means.

Three rules govern the verdicts, and they are what makes the result trustworthy
rather than merely present.

**Readable-but-absent is a failure, not a gap.**  If the artifact loads and the
column or array the rule names is missing, the rule FAILS.  Only a rule nobody
could evaluate -- an unreadable file, an absent reader, a vocabulary the intent
never declared -- is UNCHECKABLE.  Getting this backwards would reward an agent
for producing *less*: omit ``X_pca`` entirely and the check that would have
caught it becomes a harness gap that does not block validity.

**Format comes from the file, never from the declaration.**  The benchmark
declares ``format: parquet`` for tables that ``LocalArtifactStore.save_table``
writes as ``.csv``, and declares ``format: h5ad`` for an annotation artifact that
is really a table.  Dispatching on the declaration would make every rule on those
artifacts unevaluable against a file that was sitting right there.

**It must never raise.**  This runs inside the step loop, so an exception here
would surface as the agent's action failing.  Every failure to read becomes an
UNCHECKABLE rule and a recorded limitation.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agent_evals.benchmarks.schema import ValidationRule
from agent_evals.core.artifact_rules import (
    ParsedRule,
    RuleKind,
    UnparseableValidationRule,
    parse_validation_rule,
)
from agent_evals.environment.models import (
    ArtifactRecord,
    ArtifactValidation,
    RuleEvaluation,
    RuleOutcome,
)

#: How many offending values a failure message quotes before it stops.  A rule
#: that failed on every cell should produce a readable finding, not a dump of the
#: dataset.
_MAX_REPORTED_VALUES = 5

_TABLE_SEPARATORS = {".csv": ",", ".tsv": "\t"}

#: Read in 1 MiB blocks so re-digesting a large ``.h5ad`` does not hold the whole
#: file in memory. Matches the block size the free-execution tier already uses.
_DIGEST_BLOCK_BYTES = 1024 * 1024

_DIGEST_PREFIX = "sha256:"


@dataclass(frozen=True)
class LoadedArtifact:
    """A format-neutral view of one artifact's contents.

    ``is_container`` separates "this file has no single numeric payload because of
    what it is" from "this file's payload turned out to be empty".  An ``.h5ad``
    is a container of many arrays, so a rule about *the* payload cannot be
    evaluated against it; a table of strings is not a container, so the same rule
    has a real answer, and the answer is no.
    """

    columns: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, Any] = field(default_factory=dict)
    payload: Any | None = None
    is_container: bool = False


def artifact_path(uri: str | None) -> Path | None:
    """Resolve an artifact's location from either spelling of ``uri``.

    The two execution tiers disagree: the typed tier records ``str(path)`` and the
    free tier records ``path.as_uri()``.  Handling only one of them would make
    validation silently tier-specific, which is the drift the shared port exists
    to prevent.

    A one-character scheme is read as a Windows drive letter rather than a URI
    scheme.  ``urlparse`` cannot tell the two apart -- it parses ``C:\\dir\\file``
    as scheme ``c`` and drops the drive -- and getting that backwards resolves
    every typed-tier artifact on Windows to ``None``, which this layer would
    then report as an unreadable file rather than as the defect it is.
    """
    if not uri or not uri.strip():
        return None
    text = uri.strip()
    scheme = urlparse(text).scheme.lower()
    if len(scheme) > 1 and scheme != "file":
        # A genuine remote locator. Not a local artifact, so not ours to read.
        return None
    if scheme != "file":
        return _as_path(text)
    parsed = urlparse(text)
    if parsed.netloc.lower() not in ("", "localhost"):
        return None
    raw = unquote(parsed.path)
    # A Windows ``file:///C:/x`` URI parses to the path ``/C:/x``, which is not a
    # location on disk until the leading separator is dropped.
    if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
        raw = raw[1:]
    return _as_path(raw)


def _as_path(raw: str) -> Path | None:
    """Build a ``Path`` without letting a malformed name raise."""
    try:
        return Path(raw)
    except (OSError, ValueError):
        return None


def _file_digest(path: Path) -> str:
    """Return a file's SHA-256 as bare lowercase hex."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DIGEST_BLOCK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_digest(value: str) -> str:
    """Strip the optional algorithm prefix the two tiers disagree about."""
    text = value.strip().lower()
    return text[len(_DIGEST_PREFIX) :] if text.startswith(_DIGEST_PREFIX) else text


def verify_checksum(path: Path, recorded: str | None) -> bool | None:
    """Re-derive the file's digest and compare it to the recorded one.

    Returns ``None`` when there is nothing to compare against or the file could
    not be read, because "no evidence of tampering" and "verified untampered" are
    different claims and only the second one is worth reporting as such.
    """
    if not recorded:
        return None
    try:
        return _file_digest(path) == _normalize_digest(recorded)
    except OSError:
        return None


def _frame_columns(frame: Any) -> dict[str, Any]:
    """Index a dataframe's columns by name, keeping the first of any duplicate."""
    columns: dict[str, Any] = {}
    for position, raw_name in enumerate(frame.columns):
        name = str(raw_name)
        if name not in columns:
            columns[name] = frame.iloc[:, position]
    return columns


def _load_table(path: Path, suffix: str) -> LoadedArtifact:
    """Load a delimited or columnar table."""
    import pandas as pd

    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, sep=_TABLE_SEPARATORS[suffix])
    numeric = frame.select_dtypes(include="number")
    return LoadedArtifact(
        columns=_frame_columns(frame),
        payload=numeric.to_numpy() if numeric.shape[1] else None,
    )


def _load_dataset(path: Path) -> LoadedArtifact:
    """Load an annotated dataset, exposing ``obs`` columns and its arrays.

    ``obsm`` keys are exposed bare because that is how a benchmark names them
    (``X_pca``); ``layers`` keys are prefixed, because a bare layer called
    ``counts`` would be indistinguishable from an ``obs`` column of the same name
    and the rule would silently check the wrong one.
    """
    import anndata

    adata = anndata.read_h5ad(path)
    arrays: dict[str, Any] = {}
    if getattr(adata, "X", None) is not None:
        arrays["X"] = adata.X
    for obsm_key in adata.obsm:
        arrays[str(obsm_key)] = adata.obsm[obsm_key]
    # ``.keys()`` rather than iterating the mapping: anndata annotates the
    # generator ``Layers.__iter__`` as returning ``str | None`` instead of an
    # iterator of it, so direct iteration does not type-check.
    for layer_key in adata.layers.keys():
        # A ``None`` layer key is how anndata surfaces ``X`` on a backed object,
        # and ``X`` is already exposed above.  Admitting it here would publish the
        # same array under the name "layers.None", which no benchmark can name.
        if layer_key is None:
            continue
        arrays[f"layers.{layer_key}"] = adata.layers[layer_key]
    return LoadedArtifact(
        columns=_frame_columns(adata.obs),
        arrays=arrays,
        is_container=True,
    )


def _load_array(path: Path) -> LoadedArtifact:
    """Load a bare numeric array.

    ``allow_pickle=False`` is a security control, not a default.  An agent writes
    these files, and a pickled array would execute arbitrary code inside the
    evaluator the moment it read one.
    """
    import numpy as np

    return LoadedArtifact(payload=np.load(path, allow_pickle=False))


def load_artifact(path: Path) -> tuple[LoadedArtifact | None, str]:
    """Load an artifact by its real extension, or explain why it could not be.

    The reason is returned rather than raised because every caller turns it into
    an UNCHECKABLE rule, and an exception crossing this boundary would be
    reported as the agent's step failing.
    """
    suffix = path.suffix.lower()
    try:
        if suffix in _TABLE_SEPARATORS or suffix == ".parquet":
            return _load_table(path, suffix), ""
        if suffix == ".h5ad":
            return _load_dataset(path), ""
        if suffix == ".npy":
            return _load_array(path), ""
    except ImportError as error:
        return None, f"no reader for '{suffix}' is installed ({error})"
    except Exception as error:
        return None, f"'{suffix}' could not be read ({type(error).__name__}: {error})"
    return None, f"no reader is registered for the '{suffix or 'extensionless'}' format"


def _resolve_values(loaded: LoadedArtifact, target: str) -> Any | None:
    """Resolve a rule's target to values, or ``None`` if the artifact lacks it."""
    if target in loaded.arrays:
        return loaded.arrays[target]
    if target in loaded.columns:
        return loaded.columns[target]
    for prefix in ("obs.", "obsm."):
        if target.startswith(prefix):
            return _resolve_values(loaded, target[len(prefix) :])
    return None


def _is_finite(values: Any) -> bool:
    """Return whether every value is finite, densifying a sparse matrix first."""
    import numpy as np

    if hasattr(values, "toarray"):
        values = values.toarray()
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    return bool(np.isfinite(values).all())


def _check_columns_include(loaded: LoadedArtifact, rule: ParsedRule) -> tuple[RuleOutcome, str]:
    """Check that every named column is present."""
    missing = [name for name in rule.columns if name not in loaded.columns]
    if not missing:
        return RuleOutcome.PASSED, f"all {len(rule.columns)} required column(s) present"
    present = ", ".join(sorted(loaded.columns)) or "none"
    return RuleOutcome.FAILED, f"missing column(s): {', '.join(missing)}; present: {present}"


def _check_finite(loaded: LoadedArtifact, rule: ParsedRule) -> tuple[RuleOutcome, str]:
    """Check that a named target, or the artifact's own payload, is finite."""
    if not rule.target:
        if loaded.is_container:
            return (
                RuleOutcome.UNCHECKABLE,
                "this format holds many arrays, so it has no single payload; "
                "name one (for example 'X' or 'X_pca')",
            )
        values = loaded.payload
        if values is None:
            return RuleOutcome.FAILED, "the artifact holds no numeric values"
    else:
        values = _resolve_values(loaded, rule.target)
        if values is None:
            return RuleOutcome.FAILED, f"the artifact has no '{rule.target}'"
    try:
        finite = _is_finite(values)
    except (TypeError, ValueError) as error:
        dtype = getattr(values, "dtype", type(values).__name__)
        return RuleOutcome.FAILED, f"values of type '{dtype}' cannot be finite ({error})"
    if finite:
        return RuleOutcome.PASSED, "all values are finite"
    return RuleOutcome.FAILED, "at least one value is NaN or infinite"


def _check_distinct(loaded: LoadedArtifact, rule: ParsedRule) -> tuple[RuleOutcome, str]:
    """Check that a column carries at least the required number of values."""
    series = loaded.columns.get(rule.target)
    if series is None:
        present = ", ".join(sorted(loaded.columns)) or "none"
        return RuleOutcome.FAILED, f"no '{rule.target}' column; present: {present}"
    minimum = rule.minimum or 0
    try:
        distinct = int(series.nunique(dropna=True))
    except (TypeError, ValueError) as error:
        return RuleOutcome.UNCHECKABLE, f"'{rule.target}' values are not comparable ({error})"
    if distinct >= minimum:
        return RuleOutcome.PASSED, f"{distinct} distinct value(s), at least {minimum} required"
    return RuleOutcome.FAILED, f"only {distinct} distinct value(s), {minimum} required"


def _resolve_vocabulary(
    name: str,
    parameters: Mapping[str, Any],
) -> tuple[frozenset[str] | None, str]:
    """Resolve a named vocabulary from the producing intent's parameters."""
    if name not in parameters:
        return None, f"the producing step declared no '{name}' parameter to check against"
    raw = parameters[name]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return None, f"'{name}' is a {type(raw).__name__}, not a list of allowed values"
    return frozenset(str(item) for item in raw), ""


def _check_vocabulary(
    loaded: LoadedArtifact,
    rule: ParsedRule,
    parameters: Mapping[str, Any],
) -> tuple[RuleOutcome, str]:
    """Check that a column is fully populated and drawn from a declared list."""
    allowed, reason = _resolve_vocabulary(rule.vocabulary, parameters)
    if allowed is None:
        return RuleOutcome.UNCHECKABLE, reason
    series = _resolve_values(loaded, rule.target)
    if series is None or not hasattr(series, "isna"):
        present = ", ".join(sorted(loaded.columns)) or "none"
        return RuleOutcome.FAILED, f"no '{rule.target}' column; present: {present}"
    missing = int(series.isna().sum())
    if missing:
        return RuleOutcome.FAILED, f"{missing} value(s) are null"
    outside = sorted({str(value) for value in series.unique()} - allowed)
    if not outside:
        return RuleOutcome.PASSED, f"all values are drawn from the {len(allowed)} allowed"
    quoted = ", ".join(outside[:_MAX_REPORTED_VALUES])
    suffix = f" (and {len(outside) - _MAX_REPORTED_VALUES} more)" if len(outside) > _MAX_REPORTED_VALUES else ""
    return RuleOutcome.FAILED, f"value(s) outside '{rule.vocabulary}': {quoted}{suffix}"


def evaluate_rule(
    loaded: LoadedArtifact,
    rule: ParsedRule,
    parameters: Mapping[str, Any],
) -> tuple[RuleOutcome, str]:
    """Dispatch one parsed rule to its check."""
    if rule.kind is RuleKind.COLUMNS_INCLUDE:
        return _check_columns_include(loaded, rule)
    if rule.kind is RuleKind.FINITE_VALUES:
        return _check_finite(loaded, rule)
    if rule.kind is RuleKind.DISTINCT_VALUES_AT_LEAST:
        return _check_distinct(loaded, rule)
    return _check_vocabulary(loaded, rule, parameters)


class ArtifactRuleValidator:
    """Check produced artifacts against their declared rules.

    Implements :class:`~agent_evals.environment.ports.ArtifactValidator`, so both
    execution tiers reach the same verdict about the same file through the same
    code rather than each judging its own output.
    """

    async def validate(
        self,
        artifact: ArtifactRecord,
        rules: Sequence[ValidationRule],
        parameters: Mapping[str, Any],
    ) -> ArtifactValidation:
        """Check one artifact off the event loop.

        Re-digesting a large ``.h5ad`` and reading it back is slow enough to stall
        the loop that streams progress to the UI, which is the same reason
        ``ScientificActionExecutor`` runs Scanpy in a thread.
        """
        return await asyncio.to_thread(self.validate_now, artifact, rules, parameters)

    def validate_now(
        self,
        artifact: ArtifactRecord,
        rules: Sequence[ValidationRule],
        parameters: Mapping[str, Any],
    ) -> ArtifactValidation:
        """Check one artifact synchronously, without ever raising."""
        path = artifact_path(artifact.uri)
        if path is None or not path.is_file():
            where = f" at '{artifact.uri}'" if artifact.uri else ""
            return self._all_uncheckable(rules, f"the artifact does not exist{where}")
        limitations: list[str] = []
        checksum_verified = verify_checksum(path, artifact.checksum)
        if checksum_verified is None:
            limitations.append(
                "no digest was recorded when this artifact was produced, so it "
                "cannot be shown to be unchanged since"
            )
        loaded, reason = load_artifact(path) if rules else (None, "")
        if rules and loaded is None:
            return ArtifactValidation(
                exists=True,
                checksum_verified=checksum_verified,
                rules=self._uncheckable_rules(rules, reason),
                limitations=[*limitations, reason],
            )
        return ArtifactValidation(
            exists=True,
            checksum_verified=checksum_verified,
            rules=self._evaluate(loaded, rules, parameters),
            limitations=limitations,
        )

    @staticmethod
    def _evaluate(
        loaded: LoadedArtifact | None,
        rules: Sequence[ValidationRule],
        parameters: Mapping[str, Any],
    ) -> list[RuleEvaluation]:
        """Evaluate every declared rule against a loaded artifact."""
        if loaded is None:
            return []
        evaluations: list[RuleEvaluation] = []
        for rule in rules:
            try:
                parsed = parse_validation_rule(rule.rule)
            except UnparseableValidationRule as error:
                # Reachable only for a benchmark that bypassed the load-time
                # integrity check, so it is reported rather than trusted to be
                # impossible -- and as a harness gap, because an unreadable rule
                # is the benchmark author's defect and not the agent's result.
                evaluations.append(_evaluation(rule, RuleOutcome.UNCHECKABLE, str(error)))
                continue
            try:
                outcome, detail = evaluate_rule(loaded, parsed, parameters)
            except Exception as error:  # a check must not fail the science
                outcome = RuleOutcome.UNCHECKABLE
                detail = f"the check itself failed ({type(error).__name__}: {error})"
            evaluations.append(_evaluation(rule, outcome, detail))
        return evaluations

    @staticmethod
    def _uncheckable_rules(rules: Sequence[ValidationRule], reason: str) -> list[RuleEvaluation]:
        """Mark every rule unevaluated for one shared reason."""
        return [_evaluation(rule, RuleOutcome.UNCHECKABLE, reason) for rule in rules]

    def _all_uncheckable(self, rules: Sequence[ValidationRule], reason: str) -> ArtifactValidation:
        """Report an artifact that could not be examined at all."""
        return ArtifactValidation(
            exists=False,
            rules=self._uncheckable_rules(rules, reason),
            limitations=[reason],
        )


def _evaluation(rule: ValidationRule, outcome: RuleOutcome, detail: str) -> RuleEvaluation:
    """Bind one outcome to the declared rule that produced it."""
    return RuleEvaluation(name=rule.name, rule=rule.rule, outcome=outcome, detail=detail)


__all__ = [
    "ArtifactRuleValidator",
    "LoadedArtifact",
    "artifact_path",
    "evaluate_rule",
    "load_artifact",
    "verify_checksum",
]
