"""Render the agent's own differential-expression results as a document.

Reads nothing but the table the agent's own ``differential-expression`` step
archived. That restriction is the point: a report is an interpretation artifact,
and one assembled from anything the evaluator holds would be summarizing the
answer key rather than the agent's work.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from agent_evals.core.de_evidence import (
    DE_TABLE_EFFECT_COLUMN,
    DE_TABLE_GENE_COLUMNS,
    DE_TABLE_GROUP_COLUMN,
)
from agent_evals.scientific.context import OperationOutput, ScientificContext
from agent_evals.scientific.operations.de import DE_ARTIFACT_ID

#: Default number of genes rendered per group when the benchmark declares none.
DEFAULT_TOP_N = 50

#: Section heading used for a table that carries no group column at all.
UNGROUPED_SECTION = "all ranked genes"

#: Columns rendered after the gene name, in order, when the table carries them.
#: Rendered opportunistically rather than required: which statistics a table
#: reports follows from the test the agent chose, and the declared artifact rule
#: is what holds it to the columns the benchmark asks for.
_VALUE_COLUMNS = (DE_TABLE_EFFECT_COLUMN, "p_value", "q_value")

_STYLE = (
    "body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:52rem;"
    "color:#1a1a1a;line-height:1.5}"
    "h1{font-size:1.5rem;margin-bottom:.25rem}"
    "h2{font-size:1.05rem;margin-top:2rem;border-bottom:1px solid #d8d8d8;"
    "padding-bottom:.3rem}"
    "p.provenance{color:#555;font-size:.9rem;margin-top:0}"
    "table{border-collapse:collapse;width:100%;font-size:.9rem}"
    "th{text-align:left;font-weight:600;border-bottom:1px solid #bbb;padding:.3rem .5rem}"
    "td{padding:.25rem .5rem;border-bottom:1px solid #eee}"
    "td.rank{color:#888;width:3rem}"
    "td.gene{font-family:ui-monospace,monospace}"
    "td.value{text-align:right;font-variant-numeric:tabular-nums}"
)


def report(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:
    """Summarize the agent's ranked genes as a standalone HTML document."""
    top_n = int(parameters.get("top_n", DEFAULT_TOP_N))
    if top_n < 1:
        raise ValueError(f"report requires a top_n of at least 1, got {top_n}")
    source = context.artifacts.get(DE_ARTIFACT_ID)
    if source is None:
        raise ValueError(
            f"no '{DE_ARTIFACT_ID}' artifact exists, so there is nothing to "
            "report; differential expression has to succeed first"
        )
    path = Path(source.path)
    table = _read(path)
    gene_column = next(
        (name for name in DE_TABLE_GENE_COLUMNS if name in table.columns), None
    )
    if gene_column is None:
        raise ValueError(
            "the differential-expression table names no gene column "
            f"(looked for {list(DE_TABLE_GENE_COLUMNS)}), so its ranking cannot "
            "be reported"
        )
    value_columns = [name for name in _VALUE_COLUMNS if name in table.columns]
    sections = _sections(table, top_n)
    document = _render(
        sections,
        gene_column,
        value_columns,
        source=path.name,
        top_n=top_n,
    )
    artifact = context.artifact_store.save_text(
        "differential_expression_report",
        document,
        file_format="html",
        metadata={
            "top_n": top_n,
            "source_artifact": DE_ARTIFACT_ID,
            # The digest of the bytes this document was rendered from, computed at
            # read time rather than copied off the source record: a report citing a
            # checksum it never verified would claim provenance it does not have.
            "source_checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
            "groups_reported": [name for name, _ in sections],
            "columns_reported": [gene_column, *value_columns],
        },
    )
    return OperationOutput(
        artifacts=[artifact],
        outputs={
            "groups_reported": [name for name, _ in sections],
            "genes_reported": sum(int(frame.shape[0]) for _, frame in sections),
            "top_n": top_n,
        },
    )


def _read(path: Path) -> Any:
    """Read a persisted table, dispatching on the file rather than a declaration.

    Same rule Stage 3's artifact validator follows: the store decides the on-disk
    format, and a benchmark's ``format:`` declaration can disagree with it.
    """
    import pandas as pd

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _sections(table: Any, top_n: int) -> list[tuple[str, Any]]:
    """Split a table into the groups it reports, each truncated to ``top_n``.

    Row order is preserved rather than re-sorted. The ranking is the agent's own
    output and the metrics score it as given, so re-ranking here would produce a
    document describing a ranking the agent never submitted.
    """
    if DE_TABLE_GROUP_COLUMN not in table.columns:
        return [(UNGROUPED_SECTION, table.head(top_n))]
    return [
        (str(name), frame.head(top_n))
        for name, frame in table.groupby(DE_TABLE_GROUP_COLUMN, sort=False)
    ]


def _render(
    sections: list[tuple[str, Any]],
    gene_column: str,
    value_columns: list[str],
    *,
    source: str,
    top_n: int,
) -> str:
    """Assemble the whole document, which carries no external references."""
    from html import escape

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Differential expression report</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        "<h1>Differential expression report</h1>",
        '<p class="provenance">'
        f"Top {top_n} ranked gene(s) per group, from <code>{escape(source)}</code>."
        "</p>",
    ]
    for name, frame in sections:
        parts.append(f"<h2>{escape(name)}</h2>")
        parts.append(_table(frame, gene_column, value_columns))
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


def _table(frame: Any, gene_column: str, value_columns: list[str]) -> str:
    """Render one group's ranking as a table, or say that it is empty."""
    from html import escape

    if int(frame.shape[0]) == 0:
        return "<p>No genes were ranked for this group.</p>"
    header = "".join(
        f"<th>{escape(name)}</th>" for name in ("rank", gene_column, *value_columns)
    )
    rows = []
    for rank, (_index, row) in enumerate(frame.iterrows(), start=1):
        cells = "".join(
            f'<td class="value">{escape(_cell(row[name]))}</td>'
            for name in value_columns
        )
        rows.append(
            f'<tr><td class="rank">{rank}</td>'
            f'<td class="gene">{escape(str(row[gene_column]))}</td>{cells}</tr>'
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _cell(value: Any) -> str:
    """Format one numeric cell, leaving a non-finite value visible as itself.

    A NaN fold change is a real statement about a gene the agent's test could not
    size, and rendering it as ``0`` would turn that into a claim of no effect.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.4g}"


__all__ = ["DEFAULT_TOP_N", "UNGROUPED_SECTION", "report"]
