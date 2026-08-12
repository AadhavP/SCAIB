"""The one canonical vocabulary of reference-label observation columns.

Three subsystems need this vocabulary and must agree on it exactly: the dataset
redaction layer that strips these columns out of the agent-visible ``.h5ad``, the
scientific context that refuses to let an agent *write* one, and the metric layer
that decides whether a column it found is a prediction or the answer key. If any
two of them disagreed, the disagreement would not raise -- it would silently
either leak ground truth or discard a legitimate prediction.

It lives in ``core`` because ``core`` is the only leaf package. The obvious homes
are all cyclic: ``datasets`` and ``metrics`` and ``scientific`` all reach
``benchmarks.schema``, which reaches back through ``metrics.models``, so a
vocabulary defined in any of them makes the import order load-bearing.
"""

from __future__ import annotations

#: Observation columns a dataset may ship its reference labels in, in the order
#: the evaluator prefers them when a dataset carries more than one. This is the
#: list to *read* reference biology from; the order is load-bearing, so it must
#: stay a tuple rather than a set.
REFERENCE_LABEL_COLUMNS = (
    "cell_type",
    "cell_type_ref",
    "known_labels",
    "bulk_labels",
)

#: Observation columns that carry reference biology. These are evaluator inputs
#: and must never be read back as though the agent had predicted them. Broader
#: than :data:`REFERENCE_LABEL_COLUMNS` because blocking a *write* should be more
#: inclusive than choosing what to read.
RESERVED_REFERENCE_COLUMNS = frozenset(REFERENCE_LABEL_COLUMNS) | {"reference_labels"}

#: Observation columns an agent may write to record its own predictions.
AGENT_PREDICTION_COLUMNS = ("predicted_labels", "predicted_cell_type")

__all__ = [
    "AGENT_PREDICTION_COLUMNS",
    "REFERENCE_LABEL_COLUMNS",
    "RESERVED_REFERENCE_COLUMNS",
]
