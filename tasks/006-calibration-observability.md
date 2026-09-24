# Task 006 — Calibration and Observability

Implement post-processing model calibration and decision telemetry for Jevlaya.

Acceptance criteria:
- Model calibration:
  - Temperature scaling for categorical distributions, scores, and binary (noul) probabilities
  - Confidence scaling / Platt calibration
  - Automated calibration fitting on calibration sets / benchmark datasets
  - Comparison of pre- and post-calibration ECE and Brier score
  - Pure Python implementation without new heavy dependencies
- Telemetry and Observability:
  - Canonical `DecisionEvent` schema tracking latency, provider, cost, routing decisions, and error status
  - `InMemoryTelemetrySink` with aggregations (P50/P95 latency, fallback rate, cost, error rate)
  - `JsonLinesTelemetrySink` with automatic secret masking
  - Integration with `DecisionGateway` (synchronous and asynchronous)
  - Telemetry endpoints in FastAPI server (`GET /v1/telemetry/summary`, `GET /v1/telemetry/events`)
- Comprehensive unit and integration tests
