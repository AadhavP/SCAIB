"""The observations a free-execution benchmark declares and no one else can see.

Under free execution the agent's working context is a directory, the output of
the last command it ran, and the record of what it has already tried.  None of
the three is visible to the scientific observation builder, which holds an
in-memory ``AnnData`` and an operation history that only the *typed* executor
writes -- so when the harness stops executing the science, all three go blank
there.  They have to be built from what the harness does still observe: the
workspace fingerprint and the episode's own action history.

This module exists because of a defect worth naming rather than quietly fixing.
The free benchmark declares four ``required: true`` observations and served
**one**.  The other three were filled with ``{}`` by a ``values.get(
observation_id, {})`` default in the scientific builder.  Nothing raised, because
an empty observation blocks nothing: the agent worked blind while the benchmark's
own descriptions promised it a file listing, its command output, and its action
history, and the harness recorded four observations as delivered.  Worse, since
observations are stored by id, that placeholder *overwrote* a real value whenever
another producer had already served the same id.  An observation nobody serves is
now *absent* rather than empty, and absence is what a declared-versus-served test
can see.

Three smaller rules are enforced here.  **Paths are workspace-relative and the
host root is never published**, matching the same choice in
``ProvisionedEnvironment.describe`` -- the run root is the directory whose
sibling holds the evaluator's reference store, and there is no reason to hand out
a path to it.  **A truncated listing says so**: a cap that silently drops entries
reads as "that is the whole workspace", which is the one thing a file listing
must not get wrong.  And **the history carries methods, never source**: an
entry that echoed the agent's whole script back at it would grow the persisted
episode quadratically while telling the agent nothing it did not write.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.core.intent_parameters import EXECUTION_PARAMETERS
from agent_evals.environment.execution.backend import WorkspaceBackend
from agent_evals.environment.execution.executor import (
    EXECUTION_STATUS_OBSERVATION,
    EXECUTION_STDERR_OBSERVATION,
    EXECUTION_STDOUT_OBSERVATION,
)
from agent_evals.environment.execution.fingerprint import WorkspaceFingerprint
from agent_evals.environment.models import ActionRecord, EpisodeSnapshot, Observation

#: Observation ids this builder serves. A benchmark that declares none of them
#: gets nothing from this builder, not a placeholder for each.
WORKSPACE_TREE_OBSERVATION = "workspace-tree"
EXECUTION_OUTPUT_OBSERVATION = "execution-output"
PIPELINE_HISTORY_OBSERVATION = "pipeline-history"

#: Entries a workspace listing may contain before it is truncated. A run that
#: writes tens of thousands of files would otherwise put all of them into every
#: subsequent observation and into the persisted episode.
DEFAULT_MAX_LISTED_FILES = 500

_SOURCE = "workspace-observation-builder"


class WorkspaceObservationBuilder:
    """Serve the free tier's workspace, output, and history observations."""

    def __init__(
        self,
        backend: WorkspaceBackend,
        *,
        max_listed_files: int = DEFAULT_MAX_LISTED_FILES,
    ) -> None:
        self.backend = backend
        self.max_listed_files = max_listed_files

    def _dispatch(self) -> dict[str, Callable[[EpisodeSnapshot], Awaitable[Observation]]]:
        """The one table that decides what this builder serves and how.

        ``served`` used to be a hand-written set beside an ``elif`` chain, and
        nothing anywhere read it -- so the declaration and the behaviour could
        disagree with no test and no type error to show it. That is the same shape
        as a flag nothing checks: removing ``pipeline-history`` from the set left
        every observation still served and the whole suite green. Deriving the
        declaration from the dispatch makes the two incapable of disagreeing.
        """
        return {
            EXECUTION_OUTPUT_OBSERVATION: self._execution_output,
            PIPELINE_HISTORY_OBSERVATION: self._pipeline_history,
            WORKSPACE_TREE_OBSERVATION: self._workspace_tree,
        }

    @property
    def served(self) -> frozenset[str]:
        """Observation ids this builder can produce."""
        return frozenset(self._dispatch())

    async def build(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
    ) -> list[Observation]:
        """Build only the declared observations this builder actually serves."""
        del specification
        dispatch = self._dispatch()
        built: list[Observation] = []
        for observation_id in dict.fromkeys(task.observations):
            builder = dispatch.get(observation_id)
            if builder is not None:
                built.append(await builder(snapshot))
        return built

    async def _workspace_tree(self, snapshot: EpisodeSnapshot) -> Observation:
        """List what is on disk now, or say that it could not be listed.

        Observation must never raise, so an unreadable workspace becomes
        ``observed: false`` with the reason attached. The distinction matters: an
        empty ``files`` list with no marker would tell an agent its own outputs
        had vanished.
        """
        try:
            value = self._listing(await self.backend.fingerprint())
        except OSError as error:
            value = {
                "observed": False,
                "files": [],
                "file_count": None,
                "total_bytes": None,
                "truncated": False,
                "limitations": [
                    "the workspace could not be listed "
                    f"({type(error).__name__}: {error})"
                ],
            }
        return Observation(
            observation_id=WORKSPACE_TREE_OBSERVATION,
            value=value,
            source=_SOURCE,
            step=snapshot.state.current_step,
        )

    def _listing(self, fingerprint: WorkspaceFingerprint) -> dict[str, Any]:
        """Render a fingerprint as the file listing the benchmark describes.

        The digest method travels with the digest. ``fingerprint`` falls back to
        a size-and-mtime identity for large files, and a consumer handed that
        string under the name "content digest" would treat evidence as proof --
        the same distinction :mod:`fingerprint` records for the evaluator, made
        visible to the agent that has to reason about its own outputs.
        """
        paths = sorted(fingerprint.files)
        listed = paths[: self.max_listed_files]
        limitations = [
            f"'{path}' exists but could not be read ({reason})"
            for path, reason in sorted(fingerprint.unreadable.items())
        ]
        dropped = len(paths) - len(listed)
        if dropped:
            limitations.append(
                f"{dropped} further file(s) are present but not listed; this "
                f"listing is capped at {self.max_listed_files} entries"
            )
        return {
            "observed": True,
            "files": [
                {
                    "path": path,
                    "size_bytes": fingerprint.files[path].size_bytes,
                    "digest": fingerprint.files[path].digest,
                    "digest_method": fingerprint.files[path].method.value,
                    "digest_proves_content": fingerprint.files[path].is_proof,
                }
                for path in listed
            ],
            "file_count": len(paths),
            "total_bytes": fingerprint.total_bytes,
            "truncated": bool(dropped),
            "limitations": limitations,
        }

    @staticmethod
    async def _pipeline_history(snapshot: EpisodeSnapshot) -> Observation:
        """Render the episode's own action history as the declared event log.

        The scientific builder serves this id from ``context.operations``, which
        only ``ScanpyExecutor`` writes -- so on the free tier it returned ``[]``
        while the benchmark's description promised "accepted actions, declared
        methods, and artifact references". The episode's action history holds
        exactly that and is populated on both tiers, so it is the honest producer
        here. The typed tier keeps the richer per-operation record, because this
        builder is only wired when a workspace was provisioned.

        Failed steps are included. The benchmark promises failures can be
        debugged and retried, and a history that quietly omitted them would let
        an agent repeat a step it had already tried without seeing that it had.
        """
        return Observation(
            observation_id=PIPELINE_HISTORY_OBSERVATION,
            value=[
                WorkspaceObservationBuilder._history_entry(record)
                for record in snapshot.state.actions
            ],
            source=_SOURCE,
            step=snapshot.state.current_step,
            metadata={"entries": len(snapshot.state.actions)},
        )

    @staticmethod
    def _history_entry(record: ActionRecord) -> dict[str, Any]:
        """Summarize one recorded action as method, outcome, and artifacts.

        ``EXECUTION_PARAMETERS`` are filtered out for the same reason decision
        extraction skips them: they are execution mechanics, and ``code`` in
        particular is the agent's whole program. Echoing it back would grow the
        persisted episode quadratically in step count while telling the agent
        nothing it did not write itself.

        Artifacts are named by their declared workspace-relative path rather than
        by ``ArtifactRecord.uri``, which is a host absolute path pointing at the
        run root whose sibling is the evaluator's reference store.
        """
        result = record.result
        return {
            "step": record.step,
            "action": record.intent.action_id,
            "status": result.status.value,
            "execution_status": (
                None
                if result.execution_status is None
                else result.execution_status.value
            ),
            "method": {
                name: value
                for name, value in record.intent.parameters.items()
                if name not in EXECUTION_PARAMETERS
            },
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "path": artifact.metadata.get("declared_path"),
                    "format": artifact.format,
                    "validated": artifact.validated,
                }
                for artifact in result.artifacts
            ],
            "error": result.error,
        }

    @staticmethod
    async def _execution_output(snapshot: EpisodeSnapshot) -> Observation:
        """Render the last recorded execution's output as text.

        Read from the action history rather than from the observation map, because
        ``record_outputs`` commits an execution's own observations only on
        success while ``record_action`` commits the record either way. A failed
        step's output is therefore reachable here, which is what the benchmark
        promises when it says failures can be debugged and retried.

        The step it describes is reported in ``metadata``, and it will lag
        ``Observation.step`` whenever the most recent action failed --
        ``_refresh_observations`` runs only after a successful step. A stale value
        that says which step it came from can be recognised as stale; one that
        does not is read as current.

        Only agent-visible observations are consulted. This builder asks for three
        specific ids and the executor's evaluator-only isolation report is not one
        of them, but filtering here means a future evaluator-only observation
        cannot reach an agent through this new channel by being added upstream.
        """
        records = snapshot.state.actions
        if not records:
            return Observation(
                observation_id=EXECUTION_OUTPUT_OBSERVATION,
                value="No execution has run yet.",
                source=_SOURCE,
                step=snapshot.state.current_step,
                metadata={"executed": False, "describes_step": None},
            )
        record = records[-1]
        result = record.result
        streams = {
            item.observation_id: item
            for item in result.observations
            if item.visible_to_agent
        }
        status = (
            result.execution_status.value
            if result.execution_status is not None
            else result.status.value
        )
        status_observation = streams.get(EXECUTION_STATUS_OBSERVATION)
        exit_code = (
            status_observation.metadata.get("exit_code")
            if status_observation is not None
            else None
        )
        lines = [
            f"step {record.step}: action '{record.intent.action_id}'",
            f"status: {status}"
            + ("" if exit_code is None else f" (exit code {exit_code})"),
        ]
        if result.error:
            lines.append(f"harness error: {result.error}")
        for label, observation_id in (
            ("stdout", EXECUTION_STDOUT_OBSERVATION),
            ("stderr", EXECUTION_STDERR_OBSERVATION),
        ):
            stream = streams.get(observation_id)
            if stream is None:
                lines.extend([f"--- {label} ---", "(not captured)"])
                continue
            text = str(stream.value)
            lines.extend([f"--- {label} ---", text if text.strip() else "(empty)"])
            if stream.metadata.get("truncated"):
                lines.append(f"({label} was truncated at the capture limit)")
        return Observation(
            observation_id=EXECUTION_OUTPUT_OBSERVATION,
            value="\n".join(lines),
            source=_SOURCE,
            step=snapshot.state.current_step,
            metadata={
                "executed": True,
                "describes_step": record.step,
                "action_id": record.intent.action_id,
                "status": status,
                "exit_code": exit_code,
            },
        )


__all__ = [
    "DEFAULT_MAX_LISTED_FILES",
    "EXECUTION_OUTPUT_OBSERVATION",
    "PIPELINE_HISTORY_OBSERVATION",
    "WORKSPACE_TREE_OBSERVATION",
    "WorkspaceObservationBuilder",
]
