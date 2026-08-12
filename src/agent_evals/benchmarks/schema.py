"""Declarative benchmark specification models.

This module is deliberately limited to the language of benchmark definitions.
The models describe scientific intent, data, observations, legal actions,
metrics, rewards, and expected artifacts; they do not load data, execute code,
or calculate scores.  Execution components should consume
:class:`BenchmarkSpecification` as their canonical input.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_evals.metrics.models import MetricRole

CURRENT_SCHEMA_VERSION = "1.0.0"
"""Newest benchmark specification schema version understood by this package."""


def _validate_identifier(value: str) -> str:
    """Validate the stable, portable identifier format used by references."""
    if not value or not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("must contain only letters, numbers, '-' or '_' and be non-empty")
    return value


def _validate_metric_identifier(value: str) -> str:
    """Validate metric IDs, which may use dotted namespaces."""
    candidate = value.replace(".", "")
    if not value or not candidate.replace("-", "").replace("_", "").isalnum():
        raise ValueError("must contain only letters, numbers, '-', '_' or '.' and be non-empty")
    return value


def _validate_semver(value: str, field_name: str) -> str:
    """Validate a schema or benchmark version using a practical SemVer subset."""
    parts = value.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"{field_name} must be a semantic version such as '1.0.0'")
    return value


class SpecificationModel(BaseModel):
    """Common strict Pydantic configuration for public specification models."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class HasIdentifier(Protocol):
    """Structural type for specification sections with stable IDs."""

    id: str


class Direction(StrEnum):
    """Optimization direction for a metric."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TARGET = "target"


class Aggregation(StrEnum):
    """How per-observation or per-dataset metric values are combined."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    WEIGHTED_MEAN = "weighted_mean"
    NONE = "none"


class ArtifactKind(StrEnum):
    """Common scientific output categories understood by downstream tooling."""

    TABLE = "table"
    ANNDATA = "anndata"
    EMBEDDING = "embedding"
    FIGURE = "figure"
    REPORT = "report"
    JSON = "json"
    DIRECTORY = "directory"
    OTHER = "other"


class Contributor(SpecificationModel):
    """A person or organization credited for a benchmark."""

    name: str = Field(min_length=1)
    institution: str | None = None
    orcid: str | None = None
    role: str | None = None


class Reference(SpecificationModel):
    """A citable scientific, software, or data reference.

    References are first-class objects so documentation generators can produce
    citations, BibTeX, and provenance pages without executing a benchmark.
    """

    id: str
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    institutions: list[str] = Field(default_factory=list)
    doi: str | None = None
    url: str | None = None
    github: str | None = None
    bibtex: str | None = None
    funding: list[str] = Field(default_factory=list)

    _identifier = field_validator("id")(_validate_identifier)


class Checksum(SpecificationModel):
    """Content checksum used to make a dataset artifact reproducible."""

    algorithm: str = Field(default="sha256", min_length=1)
    value: str = Field(min_length=1)


class DatasetSpecification(SpecificationModel):
    """Reusable dataset metadata referenced by one or more tasks.

    A dataset definition describes provenance and the expected scientific shape
    of data.  It intentionally contains no local path or download behavior;
    dataset resolvers decide how an identifier becomes a local object.
    """

    id: str
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_url: str | None = None
    organism: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    format: str = Field(default="h5ad", min_length=1)
    checksum: Checksum | None = None
    citation: list[str] = Field(default_factory=list)
    license: str = Field(min_length=1)
    expected_observations: dict[str, Any] = Field(default_factory=dict)
    recommended_tasks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _identifier = field_validator("id")(_validate_identifier)


class ObservationSpecification(SpecificationModel):
    """Declarative description of information visible to an agent.

    The ``source`` identifies the conceptual producer (for example, the
    current dataset or pipeline history).  It is not an implementation hook;
    the environment is responsible for populating the observation at runtime.
    """

    id: str
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    type: str = Field(min_length=1)
    source: str = Field(min_length=1)
    required: bool = True
    schema_definition: dict[str, Any] = Field(default_factory=dict, alias="schema")

    _identifier = field_validator("id")(_validate_identifier)


class ParameterSpecification(SpecificationModel):
    """Typed, constrained input parameter for a declared action."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    type: str = Field(min_length=1)
    required: bool = True
    default: Any = None
    choices: list[Any] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Keep numeric parameter bounds internally coherent."""
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("minimum cannot be greater than maximum")
        if self.required and self.default is not None:
            raise ValueError("required parameters cannot declare a default")
        return self


class EstimatedCost(SpecificationModel):
    """Optional resource estimate used for planning and reporting."""

    cpu_seconds: float | None = Field(default=None, ge=0)
    memory_mb: float | None = Field(default=None, ge=0)
    gpu_required: bool = False
    complexity: str | None = None


class ActionKind(StrEnum):
    """Whether an action names an operation or hands the agent the keyboard.

    A ``TYPED`` action is one the benchmark implements: the agent selects it and
    supplies declared parameters, and the harness performs the science. A
    ``FREE_EXECUTION`` action inverts that -- the agent supplies the program and
    the harness only runs it and observes what changed. The two coexist because
    they measure different things, and a benchmark that offers only the first
    measures how well an agent fills in someone else's pipeline.
    """

    TYPED = "typed"
    FREE_EXECUTION = "free_execution"


class ActionSpecification(SpecificationModel):
    """Interface contract for an operation an agent may request.

    Actions name inputs and outputs but never provide Python callables or
    execution details.  Adapters and environments implement the interface
    independently of the benchmark YAML.
    """

    id: str
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    kind: ActionKind = ActionKind.TYPED
    parameters: list[ParameterSpecification] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    estimated_cost: EstimatedCost | None = None

    _identifier = field_validator("id")(_validate_identifier)

    @model_validator(mode="after")
    def validate_free_execution(self) -> Self:
        """Keep the artifact contract per-intent for free-execution actions.

        A free-execution action is invoked many times for different purposes, so
        a fixed ``expected_outputs`` list would demand the same files from every
        invocation. The contract instead travels on each intent's ``produces``
        parameter, which the executor verifies against the workspace. Rejecting
        the declaration here rather than ignoring it is what keeps the
        environment's own output check unchanged for both kinds of action.
        """
        if self.kind is ActionKind.FREE_EXECUTION and self.expected_outputs:
            raise ValueError(
                f"free-execution action '{self.id}' must declare 'expected_outputs: []'; "
                "its artifact contract belongs on each intent's 'produces' parameter"
            )
        return self


class ExpectedRange(SpecificationModel):
    """Expected numeric range advertised by a metric definition."""

    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Ensure the lower range bound precedes the upper bound."""
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("range minimum cannot be greater than maximum")
        return self


class MetricSpecification(SpecificationModel):
    """Reusable measurement definition consumed by evaluators.

    Metrics describe what is measured and how it should be interpreted.  They
    do not contain a scoring function, dataset access, or reward logic.
    """

    id: str
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    direction: Direction
    expected_range: ExpectedRange | None = None
    aggregation: Aggregation = Aggregation.MEAN
    normalization: str | None = None
    normalization_parameters: dict[str, Any] = Field(default_factory=dict)
    contributes_to_reward: bool = False
    unit: str | None = None
    version: str = "1.0"
    category: str | None = None
    role: MetricRole | None = None
    native_min: float | None = None
    native_max: float | None = None
    applicability: dict[str, Any] = Field(default_factory=dict)
    required_artifacts: list[str] = Field(default_factory=list)
    required_metadata: list[str] = Field(default_factory=list)
    computation_backend: str | None = None
    normalization_policy: str | None = None
    aggregation_policy: str | None = None

    _identifier = field_validator("id")(_validate_metric_identifier)


class MetricGroupMember(SpecificationModel):
    """One weighted metric reference in a scientific score group."""

    metric_id: str
    weight: float = Field(gt=0)
    role: MetricRole | None = None


class MetricGroupSpecification(SpecificationModel):
    """Declarative aggregation policy for a family of metrics."""

    group_id: str
    metrics: list[MetricGroupMember] = Field(min_length=1)
    aggregation: str = "weighted_mean"
    minimum_required: int = Field(default=1, ge=1)
    contributes_to_primary: bool = True

    _identifier = field_validator("group_id")(_validate_identifier)

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        """Reject duplicate members and unsupported aggregation policies."""
        ids = [item.metric_id for item in self.metrics]
        if len(ids) != len(set(ids)):
            raise ValueError(f"metric group '{self.group_id}' contains duplicate metrics")
        if self.aggregation not in {"weighted_mean", "weighted_geometric_mean"}:
            raise ValueError(f"unsupported metric group aggregation '{self.aggregation}'")
        if self.minimum_required > len(self.metrics):
            raise ValueError("minimum_required cannot exceed metric count")
        return self


class DecisionEvaluationSpecification(SpecificationModel):
    """Declarative method and evidence contract for one decision category."""

    allowed_methods: list[str] = Field(default_factory=list)
    expected_inputs: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    parameter_ranges: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RewardComponent(SpecificationModel):
    """Declarative contribution of one metric to a reward strategy."""

    metric: str
    weight: float
    transform: str | None = None


class RewardSpecification(SpecificationModel):
    """Optimization signal assembled from independently measured metrics."""

    id: str
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    components: list[RewardComponent] = Field(default_factory=list)
    aggregation: str = "weighted_sum"
    formula: str | None = None

    _identifier = field_validator("id")(_validate_identifier)


class ValidationRule(SpecificationModel):
    """Declarative check that an expected artifact should satisfy."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rule: str = Field(min_length=1)


class ArtifactSpecification(SpecificationModel):
    """Expected scientific output and its validation contract."""

    id: str
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    kind: ArtifactKind
    format: str = Field(min_length=1)
    required: bool = True
    produced_by: list[str] = Field(default_factory=list)
    validation: list[ValidationRule] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _identifier = field_validator("id")(_validate_identifier)


class ConstraintSpecification(SpecificationModel):
    """Resource, network, dependency, and reproducibility constraints."""

    cpu_only: bool = False
    gpu_required: bool = False
    internet_access: bool = True
    max_runtime_seconds: int | None = Field(default=None, gt=0)
    max_memory_mb: int | None = Field(default=None, gt=0)
    allowed_python_packages: list[str] = Field(default_factory=list)
    deterministic: bool = False
    random_seed: int | None = None

    @model_validator(mode="after")
    def validate_hardware(self) -> Self:
        """Reject the contradictory declaration of CPU-only and GPU-required."""
        if self.cpu_only and self.gpu_required:
            raise ValueError("cpu_only and gpu_required cannot both be true")
        if self.deterministic and self.random_seed is None:
            raise ValueError("deterministic execution requires random_seed")
        return self


class EnvironmentBackend(StrEnum):
    """Which execution tier a benchmark asks for.

    ``LOCAL`` runs a subprocess on the evaluator's own host, which is fast and
    portable but cannot confine writes or cut off the network. ``CONTAINER``
    can, at the cost of requiring a container runtime. The declaration is a
    request, not a guarantee: what was actually enforced is reported per control
    in the run record, so a paper cannot claim isolation its runs lacked.
    """

    LOCAL = "local"
    CONTAINER = "container"


class EnvironmentSpecification(SpecificationModel):
    """A workspace a free-execution agent may bring its own workflow into.

    This block specifies the environment without prescribing the method: it says
    which interpreter is available and how the workspace is isolated, and says
    nothing about what analysis to run in it. Resource and reproducibility
    limits deliberately stay in ``ConstraintSpecification`` so both action kinds
    are bound by one set of numbers rather than two that can drift apart.
    """

    id: str
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    backend: EnvironmentBackend = EnvironmentBackend.LOCAL
    image: str | None = None
    languages: list[str] = Field(default_factory=lambda: ["python"])

    _identifier = field_validator("id")(_validate_identifier)

    @model_validator(mode="after")
    def validate_backend(self) -> Self:
        """Reject an under-specified container tier and an empty language list."""
        if not self.languages:
            raise ValueError(
                f"environment '{self.id}' must declare at least one language; "
                "an environment nothing can run in is not an environment"
            )
        if self.backend is EnvironmentBackend.CONTAINER and not self.image:
            raise ValueError(
                f"environment '{self.id}' requests the container backend and must "
                "name an 'image'; resolving one implicitly would make the run "
                "depend on whatever the host happened to have cached"
            )
        if self.backend is EnvironmentBackend.LOCAL and self.image:
            raise ValueError(
                f"environment '{self.id}' names an 'image' but requests the local "
                "backend, which ignores it; the declaration would misdescribe the run"
            )
        return self


class TerminationCondition(SpecificationModel):
    """Declarative condition describing when a task is complete or stopped."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    terminal: bool = True


class WorkflowStage(SpecificationModel):
    """Declarative stage in an expected, optionally conditional workflow."""

    id: str
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    allowed_actions: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    required: bool = False

    _identifier = field_validator("id")(_validate_identifier)


class EvaluationConfiguration(SpecificationModel):
    """Declarative evaluation scope for one task."""

    levels: list[str] = Field(
        default_factory=lambda: ["decision", "method", "parameter", "execution", "artifact"]
    )
    metrics: list[str] = Field(default_factory=list)


class TaskSpecification(SpecificationModel):
    """One scientific objective within a benchmark.

    A task is the join point between reusable datasets, observations, actions,
    metrics, rewards, and artifacts.  Its fields are deliberately references
    and declarative conditions so the execution engine can evolve without
    changing the scientific contract.
    """

    id: str
    name: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    description: str = Field(min_length=1)
    datasets: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    reward_strategy: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    workflow: list[WorkflowStage] = Field(default_factory=list)
    evaluation: EvaluationConfiguration = Field(default_factory=EvaluationConfiguration)
    depends_on: list[str] = Field(default_factory=list)
    termination: list[TerminationCondition] = Field(default_factory=list)
    constraints: ConstraintSpecification | None = None
    environment: str | None = None

    _identifier = field_validator("id")(_validate_identifier)


class BenchmarkMetadata(SpecificationModel):
    """Human-facing identity, discovery, and citation metadata."""

    id: str
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    version: str = "1.0.0"
    authors: list[Contributor] = Field(default_factory=list)
    license: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)

    _identifier = field_validator("id")(_validate_identifier)
    _version = field_validator("version")(
        lambda value: _validate_semver(value, "version")
    )


class BenchmarkSpecification(SpecificationModel):
    """Canonical, executable-independent representation of a benchmark.

    Benchmark authors should write this object as YAML.  The execution engine,
    evaluator, report generator, registry, and future APIs all consume the
    same validated representation.  Construction validates reference
    integrity, uniqueness, and task dependency acyclicity before the object is
    made available to downstream systems.
    """

    schema_version: str = CURRENT_SCHEMA_VERSION
    metadata: BenchmarkMetadata
    references: list[Reference] = Field(default_factory=list)
    datasets: list[DatasetSpecification] = Field(default_factory=list)
    observations: list[ObservationSpecification] = Field(default_factory=list)
    environments: list[EnvironmentSpecification] = Field(default_factory=list)
    actions: list[ActionSpecification] = Field(default_factory=list)
    metrics: list[MetricSpecification] = Field(default_factory=list)
    metric_groups: list[MetricGroupSpecification] = Field(default_factory=list)
    decision_evaluation: dict[str, DecisionEvaluationSpecification] = Field(default_factory=dict)
    rewards: list[RewardSpecification] = Field(default_factory=list)
    artifacts: list[ArtifactSpecification] = Field(default_factory=list)
    constraints: ConstraintSpecification = Field(default_factory=ConstraintSpecification)
    tasks: list[TaskSpecification] = Field(default_factory=list)

    _schema_version = field_validator("schema_version")(
        lambda value: _validate_semver(value, "schema_version")
    )

    @staticmethod
    def _unique_ids(items: Sequence[HasIdentifier], section: str) -> set[str]:
        """Return IDs after rejecting duplicates within a specification section."""
        ids = [item.id for item in items]
        duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate {section} identifier(s): {', '.join(duplicates)}")
        return set(ids)

    @staticmethod
    def _unknown(values: list[str], known: set[str]) -> list[str]:
        """Return sorted unresolved references for readable validation errors."""
        return sorted(set(values) - known)

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:  # noqa: C901
        """Validate all cross-object references and task dependency structure."""
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version '{self.schema_version}'; "
                f"supported version is '{CURRENT_SCHEMA_VERSION}'"
            )
        dataset_ids = self._unique_ids(self.datasets, "dataset")
        observation_ids = self._unique_ids(self.observations, "observation")
        environment_ids = self._unique_ids(self.environments, "environment")
        action_ids = self._unique_ids(self.actions, "action")
        free_execution_actions = {
            action.id for action in self.actions if action.kind is ActionKind.FREE_EXECUTION
        }
        metric_ids = self._unique_ids(self.metrics, "metric")
        group_ids = {group.group_id for group in self.metric_groups}
        if len(group_ids) != len(self.metric_groups):
            raise ValueError("duplicate metric group identifier(s)")
        for group in self.metric_groups:
            unknown = self._unknown([item.metric_id for item in group.metrics], metric_ids)
            if unknown:
                raise ValueError(
                    f"metric group '{group.group_id}' references unknown metric(s): {', '.join(unknown)}"
                )
        reward_ids = self._unique_ids(self.rewards, "reward")
        artifact_ids = self._unique_ids(self.artifacts, "artifact")
        task_ids = self._unique_ids(self.tasks, "task")
        reference_ids = self._unique_ids(self.references, "reference")

        metadata_refs = self._unknown(
            self.metadata.references + self.metadata.citations, reference_ids
        )
        if metadata_refs:
            raise ValueError(f"metadata references unknown reference(s): {', '.join(metadata_refs)}")

        for dataset in self.datasets:
            unknown_citations = self._unknown(dataset.citation, reference_ids)
            unknown_tasks = self._unknown(dataset.recommended_tasks, task_ids)
            if unknown_citations:
                raise ValueError(
                    f"dataset '{dataset.id}' references unknown citation(s): "
                    f"{', '.join(unknown_citations)}"
                )
            if unknown_tasks:
                raise ValueError(
                    f"dataset '{dataset.id}' recommends unknown task(s): "
                    f"{', '.join(unknown_tasks)}"
                )

        for action in self.actions:
            unknown_inputs = self._unknown(action.required_inputs, observation_ids | artifact_ids)
            unknown_outputs = self._unknown(action.expected_outputs, artifact_ids)
            if unknown_inputs:
                raise ValueError(f"action '{action.id}' has unknown input(s): {', '.join(unknown_inputs)}")
            if unknown_outputs:
                raise ValueError(f"action '{action.id}' has unknown output(s): {', '.join(unknown_outputs)}")

        for artifact in self.artifacts:
            unknown = self._unknown(artifact.produced_by, action_ids)
            if unknown:
                raise ValueError(f"artifact '{artifact.id}' has unknown producer(s): {', '.join(unknown)}")

        for reward in self.rewards:
            unknown = self._unknown([component.metric for component in reward.components], metric_ids)
            if unknown:
                raise ValueError(f"reward '{reward.id}' references unknown metric(s): {', '.join(unknown)}")

        for task in self.tasks:
            checks = (
                ("dataset", task.datasets, dataset_ids),
                ("observation", task.observations, observation_ids),
                ("action", task.allowed_actions, action_ids),
                ("metric", task.metrics, metric_ids),
                ("evaluation metric", task.evaluation.metrics, metric_ids),
                ("artifact", task.artifacts, artifact_ids),
                ("dependency", task.depends_on, task_ids),
            )
            for kind, values, known in checks:
                unknown = self._unknown(values, known)
                if unknown:
                    raise ValueError(f"task '{task.id}' references unknown {kind}(s): {', '.join(unknown)}")
            stage_ids = self._unique_ids(task.workflow, f"workflow stage for task '{task.id}'")
            for stage in task.workflow:
                unknown_actions = self._unknown(stage.allowed_actions, action_ids)
                unknown_stages = self._unknown(stage.depends_on, stage_ids)
                if unknown_actions:
                    raise ValueError(
                        f"workflow stage '{stage.id}' references unknown action(s): "
                        f"{', '.join(unknown_actions)}"
                    )
                if unknown_stages:
                    raise ValueError(
                        f"workflow stage '{stage.id}' references unknown stage(s): "
                        f"{', '.join(unknown_stages)}"
                    )
            if task.reward_strategy is not None and task.reward_strategy not in reward_ids:
                raise ValueError(
                    f"task '{task.id}' references unknown reward '{task.reward_strategy}'"
                )
            if task.environment is not None and task.environment not in environment_ids:
                raise ValueError(
                    f"task '{task.id}' references unknown environment '{task.environment}'"
                )
            free_actions = sorted(free_execution_actions.intersection(task.allowed_actions))
            if free_actions and task.environment is None:
                # Without this the benchmark would look complete and then fail at
                # run time with no workspace to execute in, blaming the agent for
                # the benchmark author's omission.
                raise ValueError(
                    f"task '{task.id}' allows free-execution action(s) "
                    f"{', '.join(free_actions)} but declares no 'environment'"
                )
            missing_artifacts = sorted(
                artifact.id
                for artifact in self.artifacts
                if artifact.required and artifact.id not in task.artifacts
            )
            if missing_artifacts:
                raise ValueError(
                    f"task '{task.id}' is missing required artifact(s): "
                    f"{', '.join(missing_artifacts)}"
                )

        self._validate_task_dependencies()
        return self

    def _validate_task_dependencies(self) -> None:
        """Reject circular task dependencies with a useful cycle path."""
        graph = {task.id: task.depends_on for task in self.tasks}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str, path: list[str]) -> None:
            if task_id in visiting:
                cycle_start = path.index(task_id) if task_id in path else 0
                cycle = [*path[cycle_start:], task_id]
                raise ValueError(f"circular task dependency: {' -> '.join(cycle)}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in graph.get(task_id, []):
                visit(dependency, [*path, task_id])
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in graph:
            visit(task_id, [])

    def model_dump_serializable(self) -> dict[str, Any]:
        """Return the stable, alias-aware dictionary used by YAML and JSON."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


# Concise aliases are useful in authoring code while the long names remain the
# canonical API and make generated documentation self-explanatory.
BenchmarkSpec = BenchmarkSpecification
DatasetSpec = DatasetSpecification
ObservationSpec = ObservationSpecification
ParameterSpec = ParameterSpecification
ActionSpec = ActionSpecification
MetricSpec = MetricSpecification
RewardSpec = RewardSpecification
ArtifactSpec = ArtifactSpecification
ConstraintSpec = ConstraintSpecification
EnvironmentSpec = EnvironmentSpecification
TaskSpec = TaskSpecification
ReferenceSpec = Reference
WorkflowStageSpec = WorkflowStage
EvaluationConfig = EvaluationConfiguration


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ActionKind",
    "ActionSpec",
    "ActionSpecification",
    "Aggregation",
    "ArtifactKind",
    "ArtifactSpec",
    "ArtifactSpecification",
    "BenchmarkMetadata",
    "BenchmarkSpec",
    "BenchmarkSpecification",
    "Checksum",
    "ConstraintSpec",
    "ConstraintSpecification",
    "Contributor",
    "DatasetSpec",
    "DatasetSpecification",
    "DecisionEvaluationSpecification",
    "Direction",
    "EnvironmentBackend",
    "EnvironmentSpec",
    "EnvironmentSpecification",
    "EstimatedCost",
    "EvaluationConfig",
    "EvaluationConfiguration",
    "ExpectedRange",
    "MetricGroupMember",
    "MetricGroupSpecification",
    "MetricSpec",
    "MetricSpecification",
    "ObservationSpec",
    "ObservationSpecification",
    "ParameterSpec",
    "ParameterSpecification",
    "Reference",
    "ReferenceSpec",
    "RewardComponent",
    "RewardSpec",
    "RewardSpecification",
    "TaskSpec",
    "TaskSpecification",
    "TerminationCondition",
    "ValidationRule",
    "WorkflowStage",
    "WorkflowStageSpec",
]
