"""Deterministic baseline pipeline and agent contracts."""

from agent_evals.baselines.base import BaselineResult, BaselineRunner
from agent_evals.baselines.oracle_agent import OracleAgentBaseline
from agent_evals.baselines.random_agent import RandomAgentBaseline
from agent_evals.baselines.scanpy_default import ScanpyDefaultBaseline
from agent_evals.baselines.seurat_reference import SeuratReferenceBaseline

__all__ = [
    "BaselineResult",
    "BaselineRunner",
    "OracleAgentBaseline",
    "RandomAgentBaseline",
    "ScanpyDefaultBaseline",
    "SeuratReferenceBaseline",
]
