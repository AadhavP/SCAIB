"""Every observation a benchmark promises must actually arrive.

An `observations:` entry in the YAML is a contract with the agent: it is described
in the generated system prompt, and the agent is scored on work it can only do if
the value shows up. Nothing enforces that contract. A declared id with no builder
behind it was served as an empty dict, which is not an error anywhere -- the
episode recorded it, the agent read `{}`, and the run looked normal. Four of them
were empty for six stages, and the whole suite stayed green.

This module closes the loop in both directions. Forward: every observation the
task selects arrives with something in it. Backward: nothing arrives that no
benchmark declared -- with one deliberate exception, which is the reason the
backward direction is worth testing at all. `scientific-observation` is appended
by the builder itself for the baseline agent's state view, so it appears in no
YAML and therefore in no benchmark review. An undeclared agent-visible channel is
exactly where a reference leak would go unnoticed, so it gets its own assertion.

Run against the real cached PBMC dataset rather than a synthetic fixture. A
fixture would have to declare its own batch column and precomputed embeddings, at
which point the test asserts that the *fixture* is complete rather than that the
benchmark's declarations are servable against data it names. Benchmarks whose data
contract this dataset cannot satisfy are skipped through the same gate the loop
uses, because an unservable observation on a dataset the benchmark does not name
is not a defect in the observation layer.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("anndata")

import anndata

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.datasets.preflight import (
    DatasetContractError,
    describe_readiness,
    validate_dataset_contract,
)
from agent_evals.environment.execution import (
    ActionKindRouter,
    WorkspaceObservationBuilder,
)
from agent_evals.environment.models import ActionIntent
from agent_evals.environment.ports import (
    CompositeObservationBuilder,
    DeclaredObservationBuilder,
)
from agent_evals.environment.provisioning import (
    provision_environment,
    select_environment,
)
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.environment.scientific_loop import ScientificActionExecutor
from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.observations import ScientificObservationBuilder

EXAMPLES = Path(__file__).parents[1] / "examples" / "benchmarks"
PBMC_CACHE = Path(".cache/datasets/pbmc68k_reduced.h5ad")

BENCHMARKS = sorted(path.name for path in EXAMPLES.glob("*.yaml"))

#: Observations that accumulate across a run and are therefore honestly empty
#: before the first step. Listed by id rather than inferred from the declared
#: type, so adding one is a decision somebody made rather than a type string
#: quietly matching. Each is still required to be *present*: an id missing from
#: the served map has no builder at all, which is the original defect.
ACCUMULATING_AT_RESET = frozenset({"pipeline-history"})

#: Served by :class:`ScientificObservationBuilder` for the baseline's own state
#: view, so no benchmark declares it. Anything else appearing here is a builder
#: publishing a channel the benchmark never described.
UNDECLARED_BY_DESIGN = frozenset({"scientific-observation"})


def _dataset(cells: int = 400) -> anndata.AnnData:
    if not PBMC_CACHE.exists():
        pytest.skip("real PBMC cache is not available")
    adata = anndata.read_h5ad(PBMC_CACHE)
    return adata[:cells].copy()


def _substantive(value: Any) -> bool:
    """Whether a served value carries information, not just a shape.

    ``{}``, ``[]`` and ``""`` are the failure mode this module exists to catch, so
    they count as absence. ``0`` and ``False`` are real measurements and count as
    present -- treating every falsy value as missing would fail a benchmark for
    honestly reporting that nothing has happened yet.
    """
    if value is None:
        return False
    if isinstance(value, str | dict | list | tuple | set):
        return len(value) > 0
    return True


def _runnable(specification: BenchmarkSpecification, task: TaskSpecification) -> Any:
    """Load the dataset, or skip when this benchmark does not describe it.

    The same gate :meth:`ScientificLoop.run` applies before spending a model call.
    Bypassing it would test ``pbmc-batch-correction`` -- which names
    ``pbmc-multi-batch`` and requires a batch covariate this dataset has none of --
    against data it never claimed to work on.
    """
    adata = _dataset()
    readiness = describe_readiness(adata, specification, task, source="test-cache")
    try:
        validate_dataset_contract(readiness, specification, task)
    except DatasetContractError as error:
        pytest.skip(f"cached dataset does not satisfy this benchmark: {error}")
    return adata


class _Provisioned:
    """The observation chain, assembled the way the real loop assembles it."""

    def __init__(self, environment: ScientificEnvironment, closer: Any) -> None:
        self.environment = environment
        self._closer = closer

    async def close(self) -> None:
        if self._closer is not None:
            await self._closer.close()


async def _environment(
    specification: BenchmarkSpecification,
    task: TaskSpecification,
    adata: Any,
    run_root: Path,
) -> _Provisioned:
    """Build the benchmark's own declared tier, chain and all.

    Argument order of the composite builder is precedence order and is part of the
    contract, so it is reproduced here verbatim: a guard assembled from a different
    chain would pass while the real one fails.
    """
    context = ScientificContext(
        adata=adata,
        dataset_metadata={"id": "pbmc68k_reduced"},
        artifact_store=LocalArtifactStore(run_root / "artifacts"),
        workspace=run_root,
    )
    selected = select_environment(specification, task)
    provisioned = (
        None
        if selected is None
        else await provision_environment(
            specification, selected, adata, run_root=run_root
        )
    )
    return _Provisioned(
        ScientificEnvironment(
            specification,
            task_id=task.id,
            executor=ActionKindRouter.from_specification(
                specification,
                typed=ScientificActionExecutor(
                    context,
                    expected_outputs={
                        action.id: list(action.expected_outputs)
                        for action in specification.actions
                    },
                ),
                free=None if provisioned is None else provisioned.executor,
            ),
            observation_builder=CompositeObservationBuilder(
                DeclaredObservationBuilder(),
                ScientificObservationBuilder(context),
                *(
                    ()
                    if provisioned is None
                    else (WorkspaceObservationBuilder(provisioned.backend),)
                ),
            ),
        ),
        None if provisioned is None else provisioned.backend,
    )


async def _served_at_reset(benchmark: str, run_root: Path) -> dict[str, Any]:
    specification = load_benchmark(EXAMPLES / benchmark)
    task = specification.tasks[0]
    adata = _runnable(specification, task)
    built = await _environment(specification, task, adata, run_root)
    try:
        snapshot = await built.environment.reset(seed=0)
    finally:
        await built.close()
    return {
        observation_id: observation.value
        for observation_id, observation in snapshot.state.observations.items()
    }


@pytest.mark.parametrize("benchmark", BENCHMARKS)
async def test_every_selected_observation_is_served(
    benchmark: str, tmp_path: Path
) -> None:
    """A declared observation with no builder behind it is a broken promise."""
    specification = load_benchmark(EXAMPLES / benchmark)
    task = specification.tasks[0]
    selected = set(task.observations)

    served = await _served_at_reset(benchmark, tmp_path)

    absent = sorted(
        declared.id
        for declared in specification.observations
        if declared.required and declared.id in selected and declared.id not in served
    )
    assert not absent, f"{benchmark} declares these but no builder serves them: {absent}"
    empty = sorted(
        declared.id
        for declared in specification.observations
        if declared.required
        and declared.id in selected
        and declared.id not in ACCUMULATING_AT_RESET
        and not _substantive(served[declared.id])
    )
    assert not empty, f"{benchmark} serves these as empty placeholders: {empty}"


@pytest.mark.parametrize("benchmark", BENCHMARKS)
async def test_nothing_undeclared_is_served_except_the_baseline_state_view(
    benchmark: str, tmp_path: Path
) -> None:
    """The reverse direction, which the generated prompt makes load-bearing.

    An undeclared observation is worse than a missing one. The agent's prompt is
    built from the declarations, so a value served under an id nothing declares is
    information the agent cannot know to ask for -- and a channel no benchmark
    lists is a channel no benchmark review reads.
    """
    specification = load_benchmark(EXAMPLES / benchmark)
    declared = {observation.id for observation in specification.observations}

    served = await _served_at_reset(benchmark, tmp_path)

    assert set(served) - declared <= UNDECLARED_BY_DESIGN


@pytest.mark.parametrize("benchmark", BENCHMARKS)
async def test_the_undeclared_channel_carries_no_reference_vocabulary(
    benchmark: str, tmp_path: Path
) -> None:
    """The one observation nobody declared is the one nobody would audit.

    ``biological_information`` names the reference: whether it exists is a fact the
    task statement already gives away, but the label key, the vocabulary, the group
    count and the per-group counts are the answer. They are hardcoded blank, and
    this is what notices if a future edit starts filling them in from ``obs``.
    """
    served = await _served_at_reset(benchmark, tmp_path)
    view = served.get("scientific-observation")
    if view is None:
        pytest.skip("this benchmark's tier serves no baseline state view")

    biology = view["biological_information"]

    assert biology["label_key"] is None
    assert biology["num_groups"] is None
    assert biology["labels"] == []
    assert biology["counts"] == {}
    assert biology["hidden"] is True
    # Whether a reference exists at all is disclosed by the task objective, so it
    # is allowed to be true here. Nothing that identifies *which* labels may be.
    assert isinstance(biology["reference_available"], bool)


async def test_the_free_tier_records_history_once_a_step_has_run(
    tmp_path: Path,
) -> None:
    """``pipeline-history`` is exempt at reset, so its real check lives here.

    Empty before the first step is honest; empty *after* one is the defect the
    exemption would otherwise hide. It is checked on the free tier specifically
    because that is where it was blind: the scientific builder derives history from
    ``context.operations``, which only the typed executor writes, so the tier where
    the agent runs its own code recorded nothing at all.
    """
    specification = load_benchmark(EXAMPLES / "pbmc-cell-annotation-free.yaml")
    task = specification.tasks[0]
    adata = _runnable(specification, task)
    built = await _environment(specification, task, adata, tmp_path)
    try:
        await built.environment.reset(seed=0)
        step = await built.environment.step(
            ActionIntent(
                action_id="analyze",
                parameters={
                    # Deliberately trivial. This test is about whether the step is
                    # *recorded*, and a real scanpy workflow would make it a slow
                    # test of something else.
                    "code": "print('observed')",
                    "language": "python",
                    "produces": [],
                },
            )
        )
    finally:
        await built.close()

    # A rejected step would leave the history empty for a reason that has nothing
    # to do with the recording, so the premise is asserted before the conclusion.
    assert step.accepted, step.validation.errors
    history = step.observation.state.observations["pipeline-history"].value

    assert _substantive(history), "a completed step left no trace in pipeline-history"


def test_the_guard_would_notice_an_empty_declaration() -> None:
    """The guard's own predicate, asserted directly.

    Everything above can only fail loudly if ``_substantive`` reads the defect's
    shape as absence, which is cheaper to check here than to infer from a
    parametrized end-to-end test passing.
    """
    assert not _substantive({})
    assert not _substantive([])
    assert not _substantive("")
    assert not _substantive(None)
    # A real measurement of nothing-yet, which must not read as a missing builder.
    assert _substantive(0)
    assert _substantive(False)
    assert _substantive({"n_cells": 0})
