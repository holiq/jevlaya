# syntax=docker/dockerfile:1

# -----------------------------------------------------------------------------
# Base Image: Python 3.12 slim with uv package manager
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Install uv from official binary image
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

# Environment configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Set virtual environment in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create unprivileged user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

EXPOSE 8000

# Built-in healthcheck using Python's standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["jevlaya"]
CMD ["serve"]

# -----------------------------------------------------------------------------
# Target: full (Includes Laya local ML model with PyTorch ~2.5GB)
# -----------------------------------------------------------------------------
FROM base AS full

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra server --extra laya

COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --extra server --extra laya \
    && chown -R appuser:appuser /app

USER appuser

# -----------------------------------------------------------------------------
# Target: gateway (Default / Slim ~180MB - mock & jev providers)
# Placed last so standard `docker build .` defaults to this lightweight stage.
# -----------------------------------------------------------------------------
FROM base AS gateway

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra server

COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --extra server \
    && chown -R appuser:appuser /app

USER appuser
