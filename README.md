# Jevlaya — Open Decision Intelligence Stack

Jevlaya is an open-source framework for integrating structured decision models into AI agents.

The core idea is simple: agents often need thousands or millions of small decisions—routing, verification, moderation, prioritization, safety checks—rather than free-form text generation. Jevlaya provides one provider-agnostic interface for those decisions.

## Architecture & Design

```text
Agent
  |
  v
Decision Gateway
  |
  +--> Laya adapter ----> local inference
  |
  +--> Jev adapter -----> hosted decision API
  |
  +--> other adapters --> classifiers / future System-1 models
  |
  v
Normalized Decision
  |
  +--> policy / routing
  +--> observability
  +--> evaluation / calibration
```

## Decision Primitives

- `choice`: select one option from a rubric/criteria and expose probability distribution.
- `score`: select/estimate a position on an ordered scale and expose its distribution.
- `noul`: binary probability, $P(\text{yes})$.

These primitives are documented by Jev and reflected in Laya's typed-decision interface.

## Quickstart & Development Setup

Jevlaya uses **`uv`** for Python and **`bun`** for JavaScript/TypeScript tooling.

### Prerequisites

- Python 3.12+ (managed with `uv`)
- Bun 1.4+

### Setup

```bash
# 1. Initialize Python virtual environment
uv venv

# 2. Sync Python dependencies
uv sync

# 3. Install Bun dependencies
bun install
```

### Running Tests

```bash
# Run Python test suite
uv run pytest

# Run Bun test runner
bun test
```

For complete development workflow, linting, and architecture details, see the [Development Guide](docs/development.md).

## Documentation

- [v0.1 Specification](specs/jevlaya-v0.1.md) — Canonical request/response contracts and provider requirements.
- [Architecture](docs/architecture.md) — System components, decision flow, and lifecycle.
- [Decision Protocol](docs/decision-protocol.md) — Primitives specification (`choice`, `score`, `noul`).
- [Development Guide](docs/development.md) — Local environment setup with `uv` and `bun`.
- [Providers](docs/providers.md) — Provider details, adapter contracts, and benchmark guidelines.
- [Routing](docs/routing.md) — Policy routing, confidence thresholds, and uncertainty handling.
- [Testing & Benchmarking](docs/testing.md) — Unit tests, provider contract tests, and DecisionBench.

## Important Boundary

Jevlaya is **not** Jev and is **not** Laya. It is the integration layer around decision models.

Provider-specific behavior must remain isolated in adapters.

## Status

This repository specification describes the proposed v0.1 architecture. Provider APIs and benchmark numbers must be verified against current upstream documentation before implementation.

## Sources

- Laya model card: https://huggingface.co/convaiinnovations/laya
- Laya repository: https://github.com/NandhaKishorM/laya
- Jev documentation: https://openrouter.ai/docs/guides/community/jev
- TypeSafe provider page: https://openrouter.ai/provider/typesafe
