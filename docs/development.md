# Jevlaya Development Guide

This guide covers setting up your local development environment, project architecture, toolchains, testing procedures, and contributing to Jevlaya.

---

## 1. Prerequisites & Toolchain

Jevlaya uses a modern, fast development toolchain:

| Tool | Purpose | Source / Version |
| :--- | :--- | :--- |
| **`uv`** | Python environment & package management | Defined by [`.python-version`](../.python-version) (Python 3.12+) |
| **`bun`** | JavaScript / TypeScript runtime & package management | Bun 1.4+ |
| **`git`** | Version control | Standard Git |

> [!NOTE]
> Jevlaya uses **`uv`** exclusively for Python and **`bun`** for JavaScript/TypeScript tooling. Do not introduce `npm`, `yarn`, `pnpm`, `pip`, or `poetry` lockfiles.

---

## 2. Python Environment Setup (`uv`)

### 2.1 Initialize Virtual Environment

Ensure you have `uv` installed. To create a virtual environment matching the pinned Python version:

```bash
uv venv
```

To activate the virtual environment manually (optional; `uv run` handles this automatically):

```bash
source .venv/bin/activate
```

### 2.2 Install Dependencies

Install all runtime and development dependencies locked in `uv.lock`:

```bash
uv sync
```

### 2.3 Managing Python Dependencies

Do not edit dependency versions in `pyproject.toml` manually. Use `uv add`:

```bash
# Add a runtime dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Update dependencies and lockfile
uv lock
uv sync
```

### 2.4 Running Tests and Quality Tools

Execute test suites and linters using `uv run`:

```bash
# Run pytest test suite
uv run pytest

# Run lint checks with ruff
uv run ruff check .

# Auto-format code with ruff
uv run ruff format .
```

---

## 3. JavaScript / TypeScript Setup (`bun`)

Jevlaya utilizes `bun` for TypeScript execution, evaluation harnesses, and web or Node-compatible integrations:

### 3.1 Install Dependencies

```bash
bun install
```

### 3.2 Running Bun Tests & Scripts

```bash
# Run Bun test runner
bun test

# Execute any TypeScript/JavaScript script directly
bun run <path-to-script.ts>
```

---

## 4. Repository Structure

```text
jevlaya/
├── AGENTS.md               # Instructions and principles for AI coding agents
├── README.md               # Project overview and introduction
├── pyproject.toml          # PEP 621 Python package configuration
├── package.json            # Bun package configuration
├── .python-version         # Pinned Python version (3.12)
├── .gitignore              # Git ignore rules for Python & Bun
├── src/                    # Python source code
│   └── jevlaya/            # Core package
│       ├── __init__.py     # Package entry point
│       ├── protocol/       # Canonical decision schemas (choice, score, noul)
│       ├── gateway/        # Decision Gateway & request validation
│       ├── providers/      # Adapter layer (Laya, Jev, Mock)
│       ├── router/         # Policy and routing logic
│       └── telemetry/      # Telemetry & observability
├── tests/                  # Test suites
│   ├── unit/               # Unit tests (schemas, validation, errors)
│   ├── contract/           # Provider contract tests (Mock provider)
│   └── integration/        # Adapter integration tests
├── docs/                   # Architectural & technical documentation
│   ├── architecture.md     # System architecture & component flow
│   ├── decision-protocol.md# Primitives specification (choice, score, noul)
│   ├── development.md      # Development setup & contributor guide
│   ├── providers.md        # Provider facts and adapter contracts
│   ├── routing.md          # Routing policies and uncertainty handling
│   └── testing.md          # Testing, benchmarks, and calibration
├── specs/                  # Versioned RFCs and specifications
│   └── jevlaya-v0.1.md     # v0.1 design specification
└── tasks/                  # Phased milestone tasks (001 - 005)
```

---

## 5. Core Architectural Concepts

Jevlaya provides a provider-agnostic decision intelligence layer for AI agents:

```mermaid
flowchart LR
    Agent[Agent Application] --> Gateway[Decision Gateway]
    Gateway --> Validator[Validator]
    Validator --> Router[Policy Router]
    Router --> Laya[Laya Adapter (Local)]
    Router --> Jev[Jev Adapter (Hosted)]
    Router --> Mock[Mock Provider (Tests)]
    Laya --> Normalizer[Response Normalizer]
    Jev --> Normalizer
    Mock --> Normalizer
    Normalizer --> Telemetry[Telemetry & Observability]
    Normalizer --> Response[Normalized Decision Response]
```

### Decision Primitives

All requests and responses strictly adhere to three core primitives:

1. **`choice`**: Selects a single option from a named map of criteria. Emits categorical probability distribution and winning `confidence`.
2. **`score`**: Assesses position across an ordered scale or rubric. Emits an ordered probability array preserving criteria sequence.
3. **`noul`**: Evaluates a binary condition. Emits `noul` as $P(\text{yes})$ (probability of affirmative outcome).

> [!IMPORTANT]
> **Confidence vs. Probability**: Never confuse model probability with application confidence. A `noul` of 0.12 means $P(\text{yes}) = 0.12$. Whether that triggers an action depends on the caller's decision threshold.

---

## 6. Provider Adapter Contract

Each adapter implements the `DecisionProvider` protocol:

```python
class DecisionProvider(Protocol):
    name: str

    def decide(
        self,
        state: dict[str, Any],
        questions: dict[str, Any],
    ) -> DecisionResponse:
        ...
```

### Adapter Responsibilities:
1. **Request Translation**: Translate canonical state/questions to the provider's specific API payload.
2. **Execution**: Invoke the underlying inference engine (e.g. local Laya model or hosted Jev API).
3. **Normalization**: Map provider output into canonical `DecisionResponse`.
4. **Error Mapping**: Map failures into typed exceptions:
   - `InvalidRequest`
   - `UnsupportedPrimitive`
   - `ProviderUnavailable`
   - `ProviderAuthenticationError`
   - `ProviderTimeout`
   - `ProviderResponseError`
   - `NormalizationError`
5. **Secret Redaction**: Never leak raw API tokens or client credentials into error traces or logs.

---

## 7. Development Rules & Workflow

1. **Read Specifications First**: Review [`specs/jevlaya-v0.1.md`](../specs/jevlaya-v0.1.md) and [`AGENTS.md`](../AGENTS.md) before implementing changes.
2. **Smallest Correct Change**: Implement features incrementally following the MVP order:
   - Decision protocol -> Provider interface -> Mock provider -> Gateway -> CLI -> Laya adapter -> Jev adapter -> Routing -> DecisionBench.
3. **Test-Driven**: Add tests for every bug fix or behavioral change. Run `uv run pytest` and `bun test`.
4. **Reproducibility**: Ensure clean checkout reproducibility. Verify with `uv sync` and `bun install`.