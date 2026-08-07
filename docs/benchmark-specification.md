# Benchmark specification

`BenchmarkSpecification` is the canonical language for describing a scientific
benchmark. It is intentionally separate from `BaseBenchmark`: the former is a
validated research contract, while the latter is an execution integration
point retained for the framework's runtime adapters.

## Design principles

- A YAML file should explain the scientific problem, data, agent-visible information, legal actions, success criteria, and expected outputs without requiring Python.
- Datasets, actions, metrics, rewards, and artifacts are reusable objects. Tasks connect them through stable identifiers.
- The schema contains references and constraints, not callables, local paths, or execution algorithms.
- Pydantic validation makes an invalid definition fail before registration or execution.
- `schema_version` is independent from a benchmark's own `metadata.version`. The first identifies the language; the second identifies the scientific benchmark release.

## Top-level structure

```yaml
schema_version: 1.0.0
metadata: {id: ..., title: ..., version: ..., ...}
references: []
datasets: []
observations: []
actions: []
metrics: []
rewards: []
artifacts: []
constraints: {}
tasks: []
```

Tasks are the join point. A task declares its datasets, visible observations,
allowed actions, metrics, reward strategy, output artifacts, dependencies, and
termination conditions. The execution engine can interpret these declarations
without knowing what any individual biological action does.

## Loading and round-tripping

```python
from agent_evals.benchmarks.io import dump_benchmark, load_benchmark

specification = load_benchmark("examples/benchmarks/pbmc-cell-annotation.yaml")
dump_benchmark(specification, "build/pbmc-cell-annotation.json")
```

`load_benchmark` accepts YAML and JSON. `benchmark_to_dict` returns a
JSON-compatible Python dictionary, and serialization excludes `None` values
while preserving model semantics.

## Validation guarantees

Loading rejects duplicate identifiers, unknown cross-references, contradictory
hardware constraints, invalid versions, invalid parameter ranges, and circular
task dependencies. Error messages identify the section and identifier that
needs attention. A registry stores only already-validated specifications and
adds duplicate `(benchmark_id, benchmark_version)` detection.

## Registry and versioning

```python
from agent_evals.benchmarks.registry import BenchmarkSpecificationRegistry

registry = BenchmarkSpecificationRegistry()
registry.discover("examples/benchmarks")
latest = registry.get("pbmc-cell-annotation")
all_integration = registry.search(tags={"integration"})
```

The registry supports exact version lookup, metadata search, tag filtering,
directory discovery, and validation. `SchemaMigrationRegistry` provides a
pure-payload migration seam for future schema versions; migration functions
are expected to change only representation, never execute scientific work.

## Authoring guidance

Use stable IDs that will remain meaningful in reports and APIs. Put scientific
intent in `objective` and `description`, keep action parameters typed and
constrained, and declare every artifact that downstream tools should expect.
Use references for papers, software, and datasets so documentation and citation
exports remain possible. The three examples in `examples/benchmarks/` show
PBMC annotation, batch correction, and differential expression at a level
appropriate for a future published suite.
