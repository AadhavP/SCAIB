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
├── research/      # Certification gates, statistics, golden fixtures, study protocols
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

# Create and evaluate a research-readiness evidence checklist
uv run agent-evals research init \
  --benchmark-id pbmc-cell-annotation \
  --benchmark-version 1.0.0 \
  --output research-readiness.yaml
uv run agent-evals research certify --manifest research-readiness.yaml
uv run agent-evals research verify \
  --manifest research-readiness.yaml \
  --certificate research-readiness-certificate.json

# Run provider-neutral endpoint fixtures
uv run agent-evals research protocol-check --strict

# Verify the replay-oriented event ledger and public run bundle
uv run agent-evals verify-bundle results/<run-id>
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

## 🐳 Docker

The project ships a `Dockerfile`, a `frontend/Dockerfile`, and a
`docker-compose.yml` to run the FastAPI backend (with the scientific
`scanpy`/`anndata` and provider extras) together with the Vite/React console in
dev mode with hot reload. `docker-compose.prod.yml` removes direct API port
publishing, enables two API workers, applies container capability restrictions,
and serves the static frontend through nginx.

```bash
# 1. Configure environment
cp .env.example .env          # then fill in LLM_API_KEY etc.

# 2. Build and start the stack
docker compose up --build -d
#  API:      http://localhost:8000/docs
#  Console:  http://localhost:5173
```

Datasets, run artifacts, scientific results, and reports persist on the host via
mounted volumes (`./data`, `./runs`, `./results`, `./reports_output`); the
frontend proxies `/v1` to the API service inside the compose network. To add the
optional OpenHands adapter extra at build time:

```bash
docker build --build-arg UV_EXTRAS="--extra science --extra providers --extra openhands" -t agent-evals .
```

If the host port 8000 is already in use, override the published port:
`AGENT_EVALS_API__PORT=18000 docker compose up -d`.

Environment variables are loaded from your `.env` and take precedence over
`configs/*.yaml` (see [Configuration System](#configuration-system)).

### Production deployment

`docker-compose.prod.yml` replaces the dev server with a static build served by
nginx, removes direct API port publishing, and runs a dedicated durable evaluation
worker alongside the API:

```bash
# Required outside development: any other environment refuses requests
# until an API key is set.
AGENT_EVALS_ENVIRONMENT=production
AGENT_EVALS_API__API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
AGENT_EVALS_API_IMAGE=registry.example/scaib-api@sha256:<api-digest>
SCAIB_CONSOLE_IMAGE=registry.example/scaib-console@sha256:<console-digest>

docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
#  Console: http://localhost:8080
```

The console never receives the API key. Compose passes it to the reverse proxy
— nginx in production, the Vite dev proxy otherwise — which attaches
`Authorization: Bearer …` to each proxied `/v1` request server-side. Do not
create a `VITE_`-prefixed copy: Vite inlines those into the public bundle, and a
browser `EventSource` cannot send the header itself, so proxy-side injection is
also what keeps the live event stream working.

The API control plane is backed by SQLite at
`AGENT_EVALS_STORAGE__JOB_DB_PATH` (default:
`./data/evaluation_jobs.sqlite3`). Jobs, idempotency keys, leases, and SSE event
history survive an API process restart. Job state transitions and their audit
events are committed in one SQLite transaction, so a crash cannot expose a
terminal state without its corresponding event (or an event for a state that was
never persisted). A job interrupted while running is marked failed rather than
replayed automatically, because a remote agent may have performed non-idempotent
work before the response was lost. Queued jobs are resumed by the startup
supervisor when in-process execution is enabled. The production profile sets
`AGENT_EVALS_API__EXECUTE_JOBS_IN_PROCESS=false` and runs `agent-evals worker` as
a separate service; API-only web workers observe live jobs but never perform
worker recovery, keeping a web restart from falsely terminating active science.
The worker also takes a short renewable singleton lease in SQLite; starting a
second execution worker fails instead of allowing two processes to race a
non-idempotent endpoint. The production worker health check reads that lease
(`agent-evals worker-health`) rather than probing the API port. Mount the database
parent on durable storage; SQLite is intended for a single deployment/control-plane
volume, not a shared network filesystem.

`/v1/health` is a public liveness probe. `/v1/ready` is the readiness probe and
checks the durable job store plus the scheduler lifecycle. API request bodies are
bounded to 64 KiB before JSON validation. Production-like configurations fail
fast without `AGENT_EVALS_API__API_KEY`, reject wildcard CORS, require HTTPS for
black-box agent endpoints, resolve DNS destinations defensively, and reject
obvious private/loopback endpoints by default. Set
`AGENT_EVALS_API__ALLOW_PRIVATE_AGENT_ENDPOINTS=true` only for trusted local
integration testing; do not use it as a production SSRF bypass.

---

## 🔬 Research-readiness certification

A populated score is not automatically a research-grade measurement. The
repository includes a strict evidence protocol covering benchmark freeze,
sandbox isolation, metric validation, expert calibration, baselines/ablations,
replicated statistics, endpoint interoperability, and archive reproduction.
Missing evidence is represented explicitly and never converted into a zero or a
pass. Start with:

```bash
uv run agent-evals research init \
  --benchmark-id pbmc-cell-annotation \
  --benchmark-version 1.0.0 \
  --output research/pbmc-readiness.yaml
uv run agent-evals research certify \
  --manifest research/pbmc-readiness.yaml \
  --strict
```

See [`docs/research-readiness.md`](docs/research-readiness.md) for the gate
contract, digest-verified evidence/reviewer attestations, certificate integrity
verification, deterministic bootstrap/paired statistics, golden metric fixtures,
baseline/ablation study models, replay-oriented run bundles, and the boundary
between code-certified evidence and empirical work that still requires a real
dataset, Linux runner, expert reviewers, and independent reproduction.

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
