# Jevlaya v0.1 Specification

## 1. Goal

Provide a stable internal abstraction for structured AI decisions while allowing different decision engines to be swapped without changing application code.

## 2. Non-goals

- Reimplementing Jev.
- Reimplementing Laya.
- Bundling proprietary model weights.
- Pretending that all providers expose identical capabilities.
- Replacing a general-purpose LLM for tasks requiring explanation or long-form generation.

## 3. Canonical request

```json
{
  "state": {},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this?",
      "criteria": {
        "sales": "pricing and contracts",
        "support": "bugs and issues",
        "billing": "invoices and payments",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["low", "medium", "high"]
    },
    "escalate": {
      "type": "noul",
      "instructions": "Should this be escalated?"
    }
  }
}
```

## 4. Canonical response

```json
{
  "provider": "example",
  "model": "example-model",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "support",
      "probabilities": {
        "sales": 0.03,
        "support": 0.92,
        "billing": 0.03,
        "other": 0.02
      },
      "confidence": 0.92
    },
    "urgency": {
      "type": "score",
      "score": "medium",
      "probabilities": [0.10, 0.85, 0.05],
      "confidence": 0.85
    },
    "escalate": {
      "type": "noul",
      "noul": 0.12
    }
  },
  "usage": {
    "input_tokens": null,
    "output_tokens": null,
    "cost_usd": null
  },
  "latency_ms": 31.2
}
```

`confidence` is not a synonym for `probability`. For `noul`, the provider may expose only the probability of yes.

## 5. Provider interface

Conceptually:

```python
class DecisionProvider(Protocol):
    name: str

    def decide(
        self,
        state: dict,
        questions: dict,
    ) -> DecisionResponse:
        ...
```

The exact public API can change during implementation. The invariant is that all providers normalize to the canonical response.

## 6. Error model

Use explicit error categories:

- `InvalidRequest`
- `UnsupportedPrimitive`
- `ProviderUnavailable`
- `ProviderAuthenticationError`
- `ProviderTimeout`
- `ProviderResponseError`
- `NormalizationError`

Errors should preserve provider diagnostics without leaking secrets.

## 7. Determinism and reproducibility

Store:

- provider
- model/version identifier
- request schema version
- routing policy version
- timestamp
- latency
- normalized decision
- probability distribution where available
- outcome/ground truth when later known

Do not claim deterministic behavior unless the provider actually guarantees it.
