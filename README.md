# agent-evals

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Package Manager: uv](https://img.shields.io/badge/uv-fast-purple.svg)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: ruff](https://img.shields.io/badge/lint-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

An extensible, open-source evaluation suite for autonomous AI agents performing computational single-cell biology workflows.

---

## 🏛️ Architecture Overview

The `agent-evals` framework separates Concerns into modular components to support research reproducibility, custom agent adapters, isolated execution sandboxes, and structured reporting.

```
src/agent_evals/
├── core/          # Settings system (Pydantic + YAML), logging (structlog), base types
├── benchmarks/    # Benchmark definition interface, loader, and registry
├── agents/        # Agent lifecycle adapter interface and registry
├── environment/   # Isolated execution environment sandbox abstractions
├── evaluators/    # Metric calculators, evaluation logic, and scoring engines
├── datasets/      # Single-cell dataset abstractions & data loaders
├── reports/       # Report generators (JSON, HTML, Markdown)
├── api/           # FastAPI application factory and REST endpoints
├── cli/           # Typer-powered CLI interface
└── utils/         # Common async and file I/O helpers
```

---

## 🚀 Quickstart

### Prerequisites
- **Python 3.12+**
- **[`uv`](https://github.com/astral-sh/uv)** (recommended Python package manager)

### Installation

```bash
# Clone repository
git clone https://github.com/agent-evals/agent-evals.git
cd agent-evals

# Create environment and sync dependencies
uv venv
uv sync --extra dev --extra docs
```

---

## 💻 CLI Usage

`agent-evals` provides a CLI built with Typer:

```bash
# Print version
uv run agent-evals version

# List available benchmarks
uv run agent-evals list-benchmarks

# Run a benchmark against an agent
uv run agent-evals run --config configs/benchmark_config.yaml

# Launch FastAPI web server
uv run agent-evals serve --port 8000
```

---

## 🌐 API Server

Launch the REST server to trigger evaluations remotely:

```bash
uv run agent-evals serve
```
Interactive OpenAPI documentation will be available at `http://127.0.0.1:8000/docs`.

### Scientific console

The React/Vite console lives in `frontend/` and expects the API on port 8004 by
default (override with `VITE_API_PROXY_TARGET`). Start both processes in two
terminals:

```powershell
uv run agent-evals serve --port 8004
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Provider credentials remain backend environment
variables; they are never entered into or persisted by the browser.

---

## ⚙️ Configuration System

`agent-evals` uses **Pydantic Settings** combined with **YAML** configuration overrides:

```yaml
# configs/default.yaml
app_name: "agent-evals"
environment: "development"
log_level: "INFO"
api:
  host: "127.0.0.1"
  port: 8000
```

Settings can also be overridden via environment variables prefixed with `AGENT_EVALS_`:

```bash
export AGENT_EVALS_LOG_LEVEL=DEBUG
```

---

## 🧪 Development & Quality Control

Common development tasks are accessible via `make`:

```bash
make lint         # Run ruff check
make format       # Format code with black and ruff
make typecheck    # Run mypy strict type check
make test         # Run pytest unit test suite
make docs-serve   # Launch documentation site locally
```

---

## 📄 License

MIT License.
