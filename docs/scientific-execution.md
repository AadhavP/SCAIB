# Scientific execution vertical slice

The first concrete benchmark path is a real PBMC AnnData object executed by
Scanpy. Install the optional scientific dependencies with
uv sync --extra science (or pip install -e ".[science]").

Run the default preprocessing pipeline with:

    agent-evals benchmark run
      --benchmark pbmc-cell-annotation
      --pipeline configs/pipelines/pbmc_default.yaml
      --output-dir results

Each run writes pipeline.json, trajectory.json, metrics.json, report.json,
report.md, and checksum-backed scientific artifacts below a unique run
directory. The trajectory stores the typed ActionIntent and
ActionExecutionResult for every pipeline step.

The bundled pbmc68k_reduced object is a public, already transformed PBMC
AnnData dataset. The normalization operation detects that provenance state and
records that it reused the transformed representation instead of applying a
second log transform. A user-provided raw-count .h5ad path is normalized with
real scanpy.pp.normalize_total and scanpy.pp.log1p.

Objective metric implementations are deliberately evidence-bound. Annotation
metrics use observed label columns and embeddings; batch metrics remain
unavailable when the dataset has no batch column; differential-expression
metrics require both a ranked DE result and an explicitly declared reference
marker set. No scientific score is synthesized when required evidence is
missing.

The first agent-driven loop uses the same executor and artifact contracts:

    agent-evals run
      --benchmark pbmc-cell-annotation
      --agent rule-based

The rule-based agent receives only ScientificObservation summaries. It never
receives AnnData. Each run stores the detailed decision trajectory, mapped
ActionIntent, execution result, local decision rewards, final objective metrics,
and a separate global reward under results/<agent_run_id>/.
