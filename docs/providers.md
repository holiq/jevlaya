# Providers

## Verified provider facts

### Jev

Public documentation describes Jev as a System One decision model from TypeSafe. It exposes `choice`, `score`, and `noul` primitives with typed answers and probabilities.

Jev is accessed as a hosted service/API. Do not design the project around proprietary local Jev weights.

### Laya

The Laya model card describes Laya as a multilingual, non-autoregressive System 1 decision model. It accepts state plus typed questions and returns typed decisions/probabilities. The project publishes open model weights and a Router.

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
