"""Command-line interface (CLI) for Jevlaya Decision Intelligence Stack."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from jevlaya import __version__
from jevlaya.bench.runner import DecisionBench
from jevlaya.bench.sample_data import get_sample_dataset
from jevlaya.gateway.gateway import DecisionGateway
from jevlaya.providers.base import DecisionProvider
from jevlaya.providers.jev import JevAdapter
from jevlaya.providers.laya import LayaAdapter
from jevlaya.providers.mock import MockProvider


def _build_provider(name: str) -> DecisionProvider:
    """Instantiate a provider adapter by name."""
    name_clean = name.strip().lower()
    if name_clean == "mock":
        return MockProvider()
    if name_clean == "laya":
        return LayaAdapter()
    if name_clean == "jev":
        return JevAdapter()
    raise ValueError(f"Unknown provider '{name}'. Options: mock, laya, jev")


def run_serve(args: argparse.Namespace) -> int:
    """Start the self-hosted Decision Gateway HTTP server."""
    try:
        import uvicorn

        from jevlaya.server.app import create_app
    except ImportError as exc:
        print(
            f"Error: Server dependencies not installed ({exc}). "
            "Install with 'pip install jevlaya[server]' or 'uv sync --extra server'.",
            file=sys.stderr,
        )
        return 1

    try:
        provider = _build_provider(args.provider)
    except Exception as exc:
        print(f"Error configuring provider: {exc}", file=sys.stderr)
        return 1

    gateway = DecisionGateway(provider=provider)
    app = create_app(gateway=gateway)

    print(
        f"Starting Jevlaya Decision Gateway v{__version__} "
        f"on http://{args.host}:{args.port} (Provider: {provider.name})"
    )
    print(f"Interactive Swagger documentation available at http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


def run_bench(args: argparse.Namespace) -> int:
    """Execute standard DecisionBench evaluation suite."""
    try:
        provider = _build_provider(args.provider)
    except Exception as exc:
        print(f"Error configuring provider: {exc}", file=sys.stderr)
        return 1

    dataset = get_sample_dataset()
    bench = DecisionBench(dataset)

    print(f"Running DecisionBench against provider '{provider.name}'...")
    try:
        report = bench.run(provider, warmup_count=args.warmup)
        print()
        print(report.to_markdown())
        return 0
    except Exception as exc:
        print(f"Benchmark execution failed: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for Jevlaya commands."""
    parser = argparse.ArgumentParser(
        prog="jevlaya",
        description="Jevlaya: Open-source decision-intelligence stack for AI agents",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"jevlaya {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- Command: serve ---
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start self-hosted Decision Gateway HTTP server",
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    serve_parser.add_argument(
        "--provider",
        default="mock",
        help="Decision engine to serve: mock, laya, jev (default: mock)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code change",
    )

    # --- Command: bench ---
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run DecisionBench evaluation suite",
    )
    bench_parser.add_argument(
        "--provider",
        default="mock",
        help="Provider to evaluate: mock, laya, jev (default: mock)",
    )
    bench_parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup samples before recording metrics (default: 0)",
    )

    args = parser.parse_args(argv)

    if args.command == "serve":
        return run_serve(args)
    if args.command == "bench":
        return run_bench(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
