"""Persisted evaluation dimensions for agent-driven science."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.evaluation.metrics.robustness import RobustnessReport
from agent_evals.evaluation.scoring.aggregation import DomainScore
from agent_evals.metrics.aggregation import AggregationResult
from agent_evals.metrics.applicability import ApplicabilityResult
from agent_evals.metrics.results import MetricResult


class DecisionEvaluation(BaseModel):
    """Evaluation of one observable category/method/parameter choice."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    step_index: int
    action_category: str
    decision_category: str = "other"
    intent: str | None = None
    hypothesis: str | None = None
    method: str | None
    parameters: dict[str, Any]
    evidence_used: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    expected_effect: dict[str, float] = Field(default_factory=dict)
    downstream_dependency: dict[str, Any] = Field(default_factory=dict)
    valid: bool
    scientific_applicable: bool
    execution_succeeded: bool
    produced_artifacts: list[str] = Field(default_factory=list)
    consumed_artifacts: list[str] = Field(default_factory=list)
    local_reward: float | None = None
    score: float = Field(ge=0, le=1)
    reason: str


class MethodEvaluation(BaseModel):
    """Method-level validity, applicability, and observed outcome evidence."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    method: str | None
    schema_valid: bool
    scientifically_applicable: bool
    execution_succeeded: bool
    produced_artifacts: list[str] = Field(default_factory=list)
    downstream_metric_ids: list[str] = Field(default_factory=list)
    local_objective: float | None = None
    eventual_global_outcome: float | None = None
    score: float = Field(ge=0, le=1)
    reason: str


class MethodScore(BaseModel):
    """Breakdown of one method choice into observable quality dimensions."""

    decision_id: str
    method: str | None
    appropriateness: float = Field(ge=0, le=1)
    parameter_quality: float = Field(ge=0, le=1)
    execution_quality: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class TrajectoryEvaluation(BaseModel):
    """Transparent trajectory quality and efficiency dimensions."""

    model_config = ConfigDict(extra="forbid")

    protocol_compliance: float = Field(ge=0, le=1)
    invalid_action_rate: float = Field(ge=0, le=1)
    failed_action_rate: float = Field(ge=0, le=1)
    artifact_validity: float = Field(ge=0, le=1)
    duplicate_action_rate: float = Field(ge=0, le=1)
    step_count: int = Field(ge=0)
    redundant_steps: int = Field(ge=0)
    runtime_seconds: float = Field(ge=0)
    token_usage: int | None = Field(default=None, ge=0)
    resource_usage: dict[str, float | bool | None] = Field(default_factory=dict)
    failed_retries: int = Field(ge=0)
    dependency_consistency: float = Field(ge=0, le=1)
    outcome_alignment: float = Field(ge=0, le=1)
    efficiency: float = Field(ge=0, le=1)
    decision_efficiency: float = Field(default=0.0, ge=0, le=1)
    decision_consistency: float = Field(default=0.0, ge=0, le=1)
    adaptation_ability: float = Field(default=0.0, ge=0, le=1)
    counterproductive_action_detection: float = Field(default=1.0, ge=0, le=1)
    short_term_gain: float = Field(default=0.0, ge=0, le=1)
    long_term_damage: float = Field(default=0.0, ge=0, le=1)
    good_signals: list[str] = Field(default_factory=list)
    bad_signals: list[str] = Field(default_factory=list)
    recommended_improvements: list[str] = Field(default_factory=list)
    method_exploration_score: float = Field(default=0.0, ge=0, le=1)
    alternative_coverage: float = Field(default=0.0, ge=0, le=1)
    unnecessary_retries: int = Field(default=0, ge=0)
    decision_regret: float = Field(default=0.0, ge=0, le=1)
    alternative_comparisons: list[dict[str, Any]] = Field(default_factory=list)
    trajectory_quality: float = Field(ge=0, le=1)
    step_table: list[dict[str, Any]] = Field(default_factory=list)
    formula: str


class ScientificEvaluation(BaseModel):
    """All independent scientific and agent-quality evaluation dimensions."""

    model_config = ConfigDict(extra="forbid")

    metric_results: list[MetricResult] = Field(default_factory=list)
    applicability: list[ApplicabilityResult] = Field(default_factory=list)
    groups: list[AggregationResult] = Field(default_factory=list)
    domain_scores: list[DomainScore] = Field(default_factory=list)
    robustness: RobustnessReport | None = None
    decision_evaluations: list[DecisionEvaluation] = Field(default_factory=list)
    method_evaluations: list[MethodEvaluation] = Field(default_factory=list)
    method_selection_evaluations: list[MethodScore] = Field(default_factory=list)
    local_decision_rewards: list[dict[str, Any]] = Field(default_factory=list)
    trajectory: TrajectoryEvaluation
    scientific_outcome_score: float | None = Field(default=None, ge=0, le=1)
    #: ``None`` when the run produced no decisions to score. A run that never
    #: told the benchmark what it was doing has an *unmeasured* decision
    #: dimension, not a perfect one; scoring it 1.0 rewarded an agent for being
    #: less structured than one that explained itself.
    decision_score: float | None = Field(default=None, ge=0, le=1)
    method_score: float = Field(ge=0, le=1)
    #: ``None`` for the same reason as :attr:`decision_score`, which it multiplies.
    decision_quality_score: float | None = Field(default=None, ge=0, le=1)
    trajectory_score: float = Field(ge=0, le=1)
    global_agent_score: float | None = Field(default=None, ge=0, le=1)
    benchmark_score: float | None = Field(default=None, ge=0, le=1)
    score_formula: str


__all__ = [
    "DecisionEvaluation",
    "MethodEvaluation",
    "MethodScore",
    "ScientificEvaluation",
    "TrajectoryEvaluation",
]
