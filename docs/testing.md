# Testing, Evaluation, Deployment

## Unit tests

Test:

- schema validation
- each primitive
- probability normalization
- score ordering
- provider error mapping
- routing decisions
- secret redaction

## Contract tests

Every provider adapter should pass the same normalized-response contract tests.

A mock provider should exist before real providers are integrated.

## Evaluation

Maintain a versioned benchmark suite.

Minimum metrics:

- choice accuracy
- soft accuracy
- Brier score
- ECE
- score MAE
- P50/P95 latency
- throughput
- cost

Always compare on identical inputs when comparing providers.

## Calibration

Use a held-out calibration set. Temperature scaling is one possible technique.

Never assume a provider's published calibration transfers unchanged to your domain.

## Security

- Keep API keys in environment variables or a secret manager.
- Never log raw secrets.
- Define PII handling before enabling a hosted provider.
- Limit request size.
- Validate arbitrary JSON.
- Record provider data-flow implications in deployment docs.

## Deployment

Start with:

```text
Application -> Jevlaya Gateway -> Laya
                                                       -> Jev fallback
```

Add a separate gateway service only when multiple applications need shared routing, policy, telemetry, or centralized credentials.
