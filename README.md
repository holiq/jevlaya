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

## Quickstart & Usage Examples

### 1. Basic Decision with Gateway and Mock Provider

```python
from jevlaya import (
    ChoiceQuestion,
    DecisionGateway,
    DecisionRequest,
    MockProvider,
    NoulQuestion,
    ScoreQuestion,
)

# 1. Initialize Gateway with a provider
gateway = DecisionGateway(provider=MockProvider())

# 2. Define canonical state and questions
request = DecisionRequest(
    state={
        "ticket_id": "T-101",
        "message": "We were double billed this month. Please issue a refund.",
    },
    questions={
        "department": ChoiceQuestion(
            instructions="Which department should handle this request?",
            criteria={
                "billing": "Invoices, payments, and refunds",
                "tech": "Software defects and outages",
                "sales": "Plan upgrades and new contracts",
            },
        ),
        "urgency": ScoreQuestion(
            instructions="Rate urgency.",
            criteria=["low", "medium", "high"],
        ),
        "churn_risk": NoulQuestion(
            instructions="Does the user threaten to cancel?",
        ),
    },
)

# 3. Get normalized decision
response = gateway.decide(request)

print("Department :", response.answers["department"].choice)
print("Urgency    :", response.answers["urgency"].score)
print("Churn Risk :", response.answers["churn_risk"].noul)  # P(yes)
print("Latency    :", f"{response.latency_ms} ms")
```

### 2. Using Local Laya Inference

```python
from jevlaya import DecisionGateway, LayaAdapter

# Initialize Laya adapter (requires: pip install laya)
laya_adapter = LayaAdapter(model_name="convaiinnovations/laya", preload=True)
gateway = DecisionGateway(provider=laya_adapter)

response = gateway.decide(request)
```

### 3. Using Hosted Jev via OpenRouter

```python
import os
from jevlaya import DecisionGateway, JevAdapter

# Initialize Jev adapter using your OpenRouter API key
jev_adapter = JevAdapter(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model_name="typesafe/jev-1.13",
)
gateway = DecisionGateway(provider=jev_adapter)

response = gateway.decide(request)
print("Cost (USD) :", response.usage.cost_usd)
```

### 4. Running Benchmarks with DecisionBench

```python
from jevlaya import DecisionBench, MockProvider, get_sample_dataset

# Load standard 1.0.0 evaluation dataset
dataset = get_sample_dataset()
bench = DecisionBench(dataset)

# Run benchmark on a provider
report = bench.run(MockProvider())

# Print human-readable Markdown report or export JSON
print(report.to_markdown())
print(report.to_json())
```

---

## Development Setup

Jevlaya uses **`uv`** for Python and **`bun`** for JavaScript/TypeScript tooling.

### Prerequisites

- Python 3.12+ (managed with `uv`)
- Bun 1.4+

### Setup

```bash
# 1. Initialize Python virtual environment
uv venv

# 2. Sync Python dependencies including development tools
uv sync --extra dev

# 3. Install Bun dependencies
bun install
```

### Running Tests

```bash
# Run Python test suite (53 tests)
uv run pytest

# Run Ruff linter
uv run ruff check .

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

**Tasks 001–005 Completed:**
- Core Protocol & Error Hierarchy (`specs/jevlaya-v0.1.md`)
- Decision Gateway with Dependency Injection & Latency tracking
- Laya Adapter (`convaiinnovations/laya`)
- Jev Adapter (`typesafe/jev-1.13` via OpenRouter)
- DecisionBench runner with ECE, Brier, accuracy, throughput, and cost metrics

## Sources

- Laya model card: https://huggingface.co/convaiinnovations/laya
- Laya repository: https://github.com/NandhaKishorM/laya
- Jev documentation: https://openrouter.ai/docs/guides/community/jev
- TypeSafe provider page: https://openrouter.ai/provider/typesafe
