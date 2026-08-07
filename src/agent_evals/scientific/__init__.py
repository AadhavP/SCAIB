"""Concrete scientific execution primitives.

The package is intentionally optional: importing the core framework does not
import Scanpy or AnnData.  Scientific consumers install ``agent-evals[science]``.
"""

from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.pipeline import (
    PipelineSpecification,
    PipelineStep,
    load_pipeline,
)

__all__ = ["PipelineSpecification", "PipelineStep", "ScientificContext", "load_pipeline"]
