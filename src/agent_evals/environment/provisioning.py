"""Turning a benchmark's declared environment into a workspace on disk.

This is the seam where the free-execution tier stops being a set of unreachable
classes and becomes something a run actually uses. Three rules shape it.

**Provisioning follows the benchmark, not a flag.** A task that declares an
``environment`` gets one, whether or not the operator passed ``--environment``.
The alternative -- an opt-in flag -- makes forgetting it a silent
misconfiguration: free-execution intents would reach the typed executor, which
does not implement them, and the agent would be blamed for the harness's gap.
That is the shape of the scoring hole this whole effort exists to close, so the
flag *selects among* declared environments and overrides which one is used. It
cannot switch provisioning off.

**The agent's dataset is a physically different file from the evaluator's.** The
harness keeps the full object for scoring; the workspace gets a copy with every
reference column dropped, and the removed values are written outside the
workspace root. On the container tier that boundary is a mount. On the local
tier it is not enforceable at all -- ``IsolationReport`` says so per control --
and the honest consequence is that the boundary here is *declared* and the
fingerprints in the reference manifest are what would later prove a violation.

**An environment this host cannot provide is refused before the run, not during
it.** A container image that does not exist yet would otherwise surface as the
agent's first action failing, which records a wiring fault as a scientific one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_evals.benchmarks.schema import (
    BenchmarkSpecification,
    EnvironmentBackend,
    EnvironmentSpecification,
    TaskSpecification,
)
from agent_evals.core.reference_columns import (
    AGENT_CLUSTER_COLUMNS,
    AGENT_PREDICTION_COLUMNS,
)
from agent_evals.datasets.redaction import (
    REFERENCE_MANIFEST_FILENAME,
    ReferencePartition,
    ReferenceStoreManifest,
    partition_reference_columns,
    write_reference_store,
)
from agent_evals.environment.execution import (
    ContainerBackend,
    IsolationOutcome,
    IsolationReport,
    LocalProcessBackend,
    WorkspaceActionExecutor,
    WorkspaceBackend,
    isolation_from_constraints,
)
from agent_evals.environment.execution.observer import H5adDatasetObserver

#: Layout under the run directory. The workspace is what the agent's code sees;
#: the evaluator directory is its sibling rather than its child, so the reference
#: store is not reachable by a relative path from inside the workspace.
WORKSPACE_DIRNAME = "workspace"
EVALUATOR_DIRNAME = "evaluator"
REFERENCE_STORE_DIRNAME = "reference"
WORKSPACE_INPUT_DIRNAME = "inputs"
SANITIZED_DATASET_FILENAME = "dataset.h5ad"


class EnvironmentProvisionError(ValueError):
    """Raised when a declared environment cannot be provided on this host."""


@dataclass(frozen=True)
class ProvisionedEnvironment:
    """A live workspace plus the evidence of what it does and does not enforce."""

    environment: EnvironmentSpecification
    executor: WorkspaceActionExecutor
    backend: WorkspaceBackend
    isolation: IsolationReport
    run_root: Path
    workspace_root: Path
    dataset_path: Path
    reference_store: Path
    manifest: ReferenceStoreManifest
    #: Precomputed analysis results the source dataset shipped that survived
    #: redaction. Not reference biology -- nothing here was derived from a
    #: withheld column -- but each one is work the agent can skip rather than do.
    retained_analysis_keys: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        """Summarize the provisioned environment for the persisted run record.

        Everything here is harness observation. ``limitations`` is the load-bearing
        field: a gap that is recorded can be read off a result file, and a gap
        that is merely absent reads as a guarantee nobody made.

        Paths are recorded relative to the run directory. An absolute path would
        be wrong as soon as results are archived, copied to another host, or --
        the case that actually happens every run -- renamed once the harness
        assigns the real run id. Everything described here lives inside the run
        directory, so a relative path is complete rather than merely portable.
        """
        return {
            "environment_id": self.environment.id,
            "backend": self.environment.backend.value,
            "workspace_root": self._relative(self.workspace_root),
            "agent_dataset": self._relative(self.dataset_path),
            "withheld_obs_columns": list(self.manifest.columns),
            "withheld_uns_keys": list(self.manifest.removed_uns_keys),
            "withheld_obsm_keys": list(self.manifest.removed_obsm_keys),
            "retained_analysis_keys": list(self.retained_analysis_keys),
            "reference_store": self._relative(self.reference_store),
            "isolation": self.isolation.model_dump(mode="json"),
            "limitations": self.limitations(),
        }

    def _relative(self, path: Path) -> str:
        """Render a path relative to the run directory, with forward slashes."""
        return path.relative_to(self.run_root).as_posix()

    def limitations(self) -> list[str]:
        """Name every guarantee this run does not have.

        Read by nothing that branches on it, which is deliberate -- these are
        disclosures, not control flow. They belong in the record because the two
        biggest gaps in this tier are invisible in a score: an unconfined process,
        and an outcome dimension that could not be measured at all.
        """
        gaps = [
            f"isolation control '{report.control.value}' was requested but is "
            f"{report.outcome.value} on this host"
            for report in self.isolation.controls
            if report.outcome
            in {IsolationOutcome.UNENFORCEABLE, IsolationOutcome.FAILED}
        ]
        if self.environment.backend is EnvironmentBackend.LOCAL:
            gaps.append(
                "local workspace artifacts are scored only with an explicit "
                "reference join, but the process is not filesystem- or network-"
                "confined; use the container backend for benchmark-grade results"
            )
        if self.retained_analysis_keys:
            # A shortcut, not a leak: nothing listed here was computed from a
            # withheld column, so no leakage finding would fire on it and none
            # should. But an agent that copies the fixture's own `louvain` into a
            # prediction column is credited with a clustering it never performed,
            # and the artifact contract cannot tell the difference. Recorded so
            # the choice of starting point is visible in the result rather than
            # discovered by whoever first wonders why a trivial run scored well.
            gaps.append(
                "the source dataset shipped precomputed analysis results that "
                "survived redaction and remain in the agent's copy ("
                + ", ".join(self.retained_analysis_keys)
                + "), so the workflow they represent can be reused rather than "
                "performed"
            )
        gaps.append(
            f"the reference store at '{REFERENCE_STORE_DIRNAME}/"
            f"{REFERENCE_MANIFEST_FILENAME}' is evaluator-only; confirmed copied "
            "candidate labels are rejected during the evaluation join"
        )
        return gaps


def select_environment(
    specification: BenchmarkSpecification,
    task: TaskSpecification,
    requested: str | None = None,
) -> EnvironmentSpecification | None:
    """Choose which declared environment to provision, or ``None`` for typed-only.

    An unknown id is an error naming what the benchmark declares, rather than a
    silent fall back to the task's own choice: an operator who asked for a
    specific environment and got a different one would be reading a run record
    that does not describe the run they requested.
    """
    declared = {spec.id: spec for spec in specification.environments}
    wanted = requested if requested is not None else task.environment
    if wanted is None:
        return None
    if wanted not in declared:
        available = ", ".join(sorted(declared)) or "(none)"
        raise EnvironmentProvisionError(
            f"unknown environment '{wanted}'; benchmark "
            f"'{specification.metadata.id}' declares: {available}"
        )
    return declared[wanted]


async def provision_environment(
    specification: BenchmarkSpecification,
    environment: EnvironmentSpecification,
    adata: Any,
    *,
    run_root: Path,
    task: TaskSpecification | None = None,
) -> ProvisionedEnvironment:
    """Materialize a sanitized workspace and the executor that runs code in it."""
    workspace_root = run_root / WORKSPACE_DIRNAME
    dataset_path = workspace_root / WORKSPACE_INPUT_DIRNAME / SANITIZED_DATASET_FILENAME
    partition = _materialize_dataset(adata, dataset_path)
    reference_store = write_reference_store(
        partition,
        run_root / EVALUATOR_DIRNAME / REFERENCE_STORE_DIRNAME,
    )
    constraints = (
        task.constraints if task is not None else None
    ) or specification.constraints
    isolation_request = isolation_from_constraints(constraints)
    if environment.backend is EnvironmentBackend.CONTAINER:
        if not environment.image:
            raise EnvironmentProvisionError(
                f"environment '{environment.id}' requests a container but has no image"
            )
        backend: WorkspaceBackend = ContainerBackend(
            workspace_root,
            image=environment.image,
            isolation=isolation_request,
            input_dir=dataset_path.parent,
        )
    else:
        backend = LocalProcessBackend(workspace_root, isolation=isolation_request)
    isolation = await backend.start()
    return ProvisionedEnvironment(
        environment=environment,
        # The observer is pointed at the sanitized copy, never the store. Its own
        # docstring states the requirement; this is the call site that owns it.
        executor=WorkspaceActionExecutor(
            backend,
            dataset_observer=H5adDatasetObserver(dataset_path),
            allowed_python_packages=constraints.allowed_python_packages,
        ),
        backend=backend,
        isolation=isolation,
        run_root=run_root,
        workspace_root=workspace_root,
        dataset_path=dataset_path,
        reference_store=reference_store,
        manifest=partition.manifest(),
        retained_analysis_keys=_retained_analysis_keys(partition.visible),
    )


def _retained_analysis_keys(visible: Any) -> tuple[str, ...]:
    """Name the precomputed analysis results the agent's copy still contains.

    Read off the sanitized object rather than declared, because which keys a
    dataset ships is a property of the file and a declaration would go stale the
    first time the fixture changed.
    """
    columns = [
        f"obs/{name}"
        for name in (*AGENT_CLUSTER_COLUMNS, *AGENT_PREDICTION_COLUMNS)
        if name in visible.obs.columns
    ]
    embeddings = [f"obsm/{key}" for key in sorted(getattr(visible, "obsm", {}))]
    return (*columns, *embeddings)


def _materialize_dataset(adata: Any, destination: Path) -> ReferencePartition:
    """Write the agent-visible dataset, returning what was held back.

    ``copy=True`` because the caller's object stays the evaluator's: stripping in
    place would make every reference-consuming metric ineligible and silently
    collapse the outcome score, which is the trap Stage 0 recorded. The copy is
    transient -- only the file outlives this function.
    """
    partition = partition_reference_columns(adata, copy=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partition.visible.write_h5ad(destination)
    return partition


__all__ = [
    "EVALUATOR_DIRNAME",
    "REFERENCE_STORE_DIRNAME",
    "SANITIZED_DATASET_FILENAME",
    "WORKSPACE_DIRNAME",
    "WORKSPACE_INPUT_DIRNAME",
    "EnvironmentProvisionError",
    "ProvisionedEnvironment",
    "provision_environment",
    "select_environment",
]
