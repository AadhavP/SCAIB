# Welcome to agent-evals

`agent-evals` is an open-source evaluation framework for evaluating autonomous AI agents on computational single-cell biology tasks.

## Key Features

- **Extensible Registry**: Easily register new benchmark definitions and agent adapters.
- **Isolated Execution**: Modular sandboxing for safe execution of agent-generated code.
- **Structured Reporting**: Export detailed evaluation results in JSON, Markdown, or HTML.
- **CLI & REST Interfaces**: Unified CLI (`agent-evals`) and FastAPI backend.
- **Type-Safe Configuration**: Pydantic Settings powered by YAML override support.

The canonical benchmark language is documented in
[Benchmark specification](benchmark-specification.md). It describes scientific
experiments as validated YAML without embedding execution logic.

The proposed publication dataset suite, accessions, split rules, and quality
gates are documented in the
[SCAIB paper dataset portfolio](paper-dataset-portfolio.md).
