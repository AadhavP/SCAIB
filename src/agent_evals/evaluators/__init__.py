"""Evaluator base abstractions and metric scoring tools."""

# Import builtins for deterministic registration at package import time.
from agent_evals.evaluators import builtin as _builtin  # noqa: F401
from agent_evals.evaluators.base import BaseEvaluator
from agent_evals.evaluators.engine import EvaluationEngine
from agent_evals.evaluators.metrics import compute_accuracy, compute_execution_time
from agent_evals.evaluators.models import (
    DecisionEvaluation,
    EvaluationLevel,
    EvaluationReport,
    EvaluationSummary,
    ExecutionEvaluation,
    MetricResult,
    MetricStatus,
    TaskInstance,
)
from agent_evals.evaluators.registry import (
    MetricComputation,
    MetricContext,
    MetricRegistry,
    RegisteredMetric,
    metric_registry,
)
from agent_evals.evaluators.rewards import GlobalReward, RewardEvaluator
from agent_evals.evaluators.task import build_task_instance

__all__ = [
    "BaseEvaluator",
    "DecisionEvaluation",
    "EvaluationEngine",
    "EvaluationLevel",
    "EvaluationReport",
    "EvaluationSummary",
    "ExecutionEvaluation",
    "GlobalReward",
    "MetricComputation",
    "MetricContext",
    "MetricRegistry",
    "MetricResult",
    "MetricStatus",
    "RegisteredMetric",
    "RewardEvaluator",
    "TaskInstance",
    "build_task_instance",
    "compute_accuracy",
    "compute_execution_time",
    "metric_registry",
]
