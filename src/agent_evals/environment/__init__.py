"""Scientific environment contracts, episode traces, and execution ports.

Free-form code execution lives in :mod:`agent_evals.environment.execution` and
is deliberately *not* re-exported here. It is reachable only as an
``ActionExecutor`` implementation, which is the layering this package depends on:
nothing above the port may know whether the science ran in a subprocess, a
container, or SCAIB's own Scanpy pipeline.
"""

from agent_evals.environment.context import EnvironmentContext
from agent_evals.environment.episode import Episode
from agent_evals.environment.models import *  # noqa: F403
from agent_evals.environment.ports import (
    ActionExecutor,
    ConstraintMonitor,
    DeclarativeActionValidator,
    ExecutionContext,
    ObservationBuilder,
    RewardEvaluator,
)
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.environment.workspace import (
    LocalWorkspace,
    WorkspaceManifest,
    WorkspaceStatus,
)

__all__ = [
    "ActionExecutor",
    "ConstraintMonitor",
    "DeclarativeActionValidator",
    "EnvironmentContext",
    "Episode",
    "ExecutionContext",
    "LocalWorkspace",
    "ObservationBuilder",
    "RewardEvaluator",
    "ScientificEnvironment",
    "WorkspaceManifest",
    "WorkspaceStatus",
]
