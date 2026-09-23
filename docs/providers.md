# Providers

## Verified provider facts

### Jev

Public documentation from OpenRouter and TypeSafe describes Jev as a hosted System One decision model:

- **Model Identifiers:** `typesafe/jev-1.13` (alias: `~typesafe/jev-latest`). Dated snapshots (e.g. `typesafe/jev-1.13-20260917`) are returned in response metadata.
- **API Surfaces:**
  - Decisions API: `POST https://openrouter.ai/api/alpha/decisions` (canonical JSON payload with `model`, `state`, `questions`).
  - System One API: `POST https://openrouter.ai/api/v1/systemone` (compatible with TypeSafe SDKs).
- **Authentication:** Standard OpenRouter API key passed via `Authorization: Bearer $OPENROUTER_API_KEY`.
- **Pricing & Context:** 32,000 token context window. Input tokens are billed at the published rate; output tokens are free.
- **Primitives:** Fully supports `choice`, `score`, and `noul`.
- **High Cardinality:** Supports up to 255 options per decision without token budget degradation.
- **Inference Latency:** Hosted API typically operates at ~236–276 ms P50 latency.

### Laya

The Laya model card describes Laya as an open-weights, multilingual, non-autoregressive System 1 decision model family developed by Convai Innovations:

- **Checkpoints:**
  - `convaiinnovations/laya`: ModernBERT-large backbone (421M params), 512 context, optimized for English text and guardrails.
  - `convaiinnovations/laya-multilingual`: mmBERT-base backbone (322M params), 1024 context, covers 100+ languages.
  - `convaiinnovations/laya-typed-decisions`: ModernBERT-large backbone (421M params), fine-tuned specifically on typed-decision workflows.
- **Inference mechanism:** Non-autoregressive single forward pass (~33 ms on Tesla T4 GPU).
- **Package:** `pip install laya` provides `from laya import Router` for automatic script/language detection and checkpoint dispatch.
- **Cardinality constraint:** Recommended for $\le 20$ options at default head token budgets; higher cardinality (>20 options) may saturate token budget without increasing `head_max_len`.

## Adapter contract

Each adapter is responsible for:

1. translating canonical request -> provider request;
2. invoking the provider;
3. parsing provider response;
4. normalizing into canonical response;
5. reporting provider-specific metadata/errors.

## Critical rule

Do not copy hypothetical SDK imports from research documents into production code.

For example, names such as:

```python
from jev_sdk import JevClient
```

or:

```ts
import { JevClient } from "@typesafe/jev-sdk";
```

are examples only until the current official SDK is verified.

## Benchmark caution

Published Jev/Laya benchmark figures are source-specific. Some comparisons explicitly note differences in prompts, samples, or measurement conditions. Jevlaya benchmarks must therefore publish:

- dataset/version
- prompt/question definitions
- hardware
- provider/model version
- batch size
- warm-up policy
- measurement method
- sample count

Never present a third-party benchmark as a universal performance guarantee.
