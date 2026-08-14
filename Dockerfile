# syntax=docker/dockerfile:1

# agent-evals runtime image: FastAPI API server + CLI with scientific
# dependencies (scanpy/anndata) and provider SDKs.

FROM python:3.12-slim

ARG APP_HOME=/app
# Extras installed by `uv sync`. Override at build time to add `openhands`,
# e.g.: docker build --build-arg UV_EXTRAS="--extra science --extra providers --extra openhands" .
ARG UV_EXTRAS="--extra science --extra providers"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    PATH="${APP_HOME}/.venv/bin:$PATH"

WORKDIR ${APP_HOME}

# Install uv and resolve dependencies against the locked environment first so
# source changes do not invalidate the dependency layer.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project ${UV_EXTRAS}

# Copy the project (source, configs, and example benchmarks used by the CLI).
COPY . .

# Install the project itself; dependencies are already satisfied by the layer
# above so this only links the package into the virtual environment.
RUN uv sync --frozen --no-dev ${UV_EXTRAS}

# Production compose runs this image as a non-root UID. Creating the persistent
# mount points in the image lets Docker initialize named volumes with the right
# ownership instead of making the worker depend on a privileged entrypoint.
RUN groupadd --system --gid 10001 agent-evals \
    && useradd --system --uid 10001 --gid 10001 --no-create-home agent-evals \
    && mkdir -p /app/data /app/runs /app/results /app/reports_output /app/.cache \
    && chown -R agent-evals:agent-evals /app/data /app/runs /app/results /app/reports_output /app/.cache

USER agent-evals

EXPOSE 8000

# Readiness verifies the durable job store, not merely that the HTTP socket is open.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/ready')" || exit 1

STOPSIGNAL SIGTERM


CMD ["agent-evals", "serve"]
