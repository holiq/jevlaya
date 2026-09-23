# Jevlaya Agent Instructions

You are working on **Jevlaya**, an open-source decision-intelligence stack for AI agents.

## Read before coding
1. `specs/jevlaya-v0.1.md`
2. `docs/architecture.md`
3. `docs/decision-protocol.md`
4. `docs/providers.md`
5. `docs/routing.md`
6. `docs/testing.md`

## Core principles
- Provider-agnostic: Jevlaya owns the abstraction, not any individual provider.
- Laya and Jev are adapters behind the same internal interface.
- Never invent provider capabilities or SDK APIs.
- Clearly distinguish verified provider facts from Jevlaya design proposals.
- Preserve typed decision semantics: `choice`, `score`, `noul`.
- Never confuse model probability with application confidence.
- Every provider adapter must normalize into the same internal response schema.
- Prefer the smallest correct implementation.
- Add tests for every behavior change.
- Keep benchmark results reproducible and label source-specific measurements.
- Do not add dependencies unless they materially simplify the implementation.
- Do not bundle proprietary Jev weights or imply Jevlaya owns Jev.

## Implementation workflow
1. Read the relevant specification.
2. Inspect existing code and tests.
3. Implement the smallest correct change.
4. Add/update tests.
5. Run the relevant test suite.
6. Update docs when behavior changes.
7. Report assumptions and unresolved provider-specific details.

## MVP order
1. Decision protocol
2. Provider interface
3. Mock provider
4. Decision Gateway
5. CLI
6. Laya adapter
7. Jev adapter
8. Routing policy
9. DecisionBench
10. Calibration and observability

## Development Environment

Jevlaya uses the following toolchain:

- Python package/dependency management: `uv`
- JavaScript/TypeScript runtime & package manager: `bun`
- Python version: defined by `.python-version`

### Python

Always use `uv` for Python environment and dependency management.

Do not use `pip install` directly for project dependencies.

Preferred commands:

```bash
uv venv
uv sync
uv run pytest
uv run python
uv add <package>
uv add --dev <package>
uv lock
```

Do not manually edit dependency versions in `pyproject.toml` when
`uv add` can be used.

### Bun / TypeScript

Use `bun` as the JavaScript/TypeScript runtime and package manager.

Preferred commands:

```bash
bun install
bun test
bun run build
bun add <package>
bun add -d <package>
```

Do not introduce `npm`, `yarn`, or `pnpm` lockfiles into the repository.

### Environment Rules

Before modifying code:

1. Check `.python-version`.
2. Run `uv sync` for Python dependencies.
3. Run `bun install` for JavaScript/TypeScript dependencies.
4. Inspect existing lockfiles before installing packages.
5. Do not introduce another package manager unless explicitly required.

The repository must remain reproducible from a clean checkout.
