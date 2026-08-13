"""Modular Scanpy operation implementations."""

from agent_evals.scientific.operations.annotate import annotate
from agent_evals.scientific.operations.batch_correction import batch_correct
from agent_evals.scientific.operations.cluster import cluster
from agent_evals.scientific.operations.de import differential_expression
from agent_evals.scientific.operations.hvg import select_hvg
from agent_evals.scientific.operations.normalize import normalize
from agent_evals.scientific.operations.pca import pca
from agent_evals.scientific.operations.qc import qc_filter
from agent_evals.scientific.operations.report import report

__all__ = [
    "annotate",
    "batch_correct",
    "cluster",
    "differential_expression",
    "normalize",
    "pca",
    "qc_filter",
    "report",
    "select_hvg",
]
