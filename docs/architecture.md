# Architecture & Design Principles

The `agent-evals` framework adopts a modular, decoupled architecture where each package component has a single, well-defined responsibility.

## Component Layout

1. **`core`**: Contains settings management (`Pydantic Settings` + `PyYAML`), `structlog` logging initialization, base models, enums, and domain exception classes.
2. **`benchmarks`**: Abstract base class (`BaseBenchmark`), benchmark context management, and the central registry.
3. **`agents`**: Legacy agent lifecycle support plus framework-neutral adapters, the execution harness, raw/normalized trajectories, and scientific decision cascades.
4. **`environment`**: Typed scientific environment and episode state machine. It validates action intents, records replayable episodes, builds observations, enforces constraints, and delegates computation through executor ports. The sandbox remains an injectable backend rather than the environment's public contract.
5. **`evaluators`**: Metric calculators and scoring logic for comparing agent outputs against ground truth datasets.
6. **`datasets`**: Single-cell dataset wrappers (e.g. AnnData / `.h5ad` file metadata helpers) and data loading abstractions.
7. **`reports`**: Report generation pipeline for rendering HTML, Markdown, and JSON evaluation reports.
8. **`api`**: FastAPI app factory providing REST endpoints for benchmark discovery, execution status, and results retrieval.
9. **`cli`**: Typer-based command line tool for executing benchmarks, inspecting registries, and serving the API.
10. **`utils`**: Generic async helper utilities and filesystem I/O operations.
