"""Command-line interface (CLI) for Jevlaya Decision Intelligence Stack."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from jevlaya import __version__
from jevlaya.bench.runner import DecisionBench
from jevlaya.bench.sample_data import get_sample_dataset
from jevlaya.calibration.fitter import fit_from_provider_and_dataset
from jevlaya.gateway.gateway import DecisionGateway
from jevlaya.providers.base import DecisionProvider
from jevlaya.providers.jev import JevAdapter
from jevlaya.providers.laya import LayaAdapter
from jevlaya.providers.mock import MockProvider
from jevlaya.telemetry.sinks import (
    CompositeTelemetrySink,
    InMemoryTelemetrySink,
    JsonLinesTelemetrySink,
    TelemetrySink,
)


def _build_provider(
    name: str,
    laya_variant: str | None = None,
    laya_model: str | None = None,
) -> DecisionProvider:
    """Instantiate a provider adapter by name."""
    name_clean = name.strip().lower()
    if name_clean == "mock":
        return MockProvider()
    if name_clean == "laya":
        return LayaAdapter(variant=laya_variant, model_name=laya_model)  # type: ignore[arg-type]
    if name_clean == "jev":
        return JevAdapter()
    raise ValueError(f"Unknown provider '{name}'. Options: mock, laya, jev")


def _build_all_providers(
    laya_variant: str | None = None,
    laya_model: str | None = None,
) -> dict[str, DecisionProvider]:
    """Instantiate all available standard providers (mock, jev, laya)."""
    return {
        "mock": MockProvider(),
        "jev": JevAdapter(),
        "laya": LayaAdapter(variant=laya_variant, model_name=laya_model),  # type: ignore[arg-type]
    }


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
        providers = _build_all_providers(
            laya_variant=getattr(args, "laya_variant", None),
            laya_model=getattr(args, "laya_model", None),
        )
    except Exception as exc:
        print(f"Error configuring providers: {exc}", file=sys.stderr)
        return 1

    default_provider = args.provider.strip().lower()
    if default_provider not in providers:
        print(
            f"Error: Unknown default provider '{args.provider}'. "
            f"Options: {list(providers.keys())}",
            file=sys.stderr,
        )
        return 1

    mem_sink = InMemoryTelemetrySink()
    telemetry: TelemetrySink
    if getattr(args, "telemetry_file", None):
        file_sink = JsonLinesTelemetrySink(args.telemetry_file)
        telemetry = CompositeTelemetrySink([mem_sink, file_sink])
    else:
        telemetry = mem_sink

    gateway = DecisionGateway(
        providers=providers,
        default_provider=default_provider,
        telemetry=telemetry,
    )
    app = create_app(gateway=gateway)

    laya_prov = providers.get("laya")
    variant_info = (
        f", Laya Variant: {getattr(laya_prov, 'variant', 'n/a')}"
        if laya_prov is not None
        else ""
    )
    prov_list = ", ".join(providers.keys())
    print(
        f"Starting Jevlaya Decision Gateway v{__version__} "
        f"on http://{args.host}:{args.port} (Default: {default_provider}{variant_info})"
    )
    print(f"Active Providers: [{prov_list}] (Selectable via ?provider=<name>)")
    print(f"Interactive Swagger documentation available at http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


def run_bench(args: argparse.Namespace) -> int:
    """Execute standard DecisionBench evaluation suite."""
    try:
        provider = _build_provider(
            args.provider,
            laya_variant=getattr(args, "laya_variant", None),
            laya_model=getattr(args, "laya_model", None),
        )
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


def run_calibrate(args: argparse.Namespace) -> int:
    """Execute calibration fitting against a provider and dataset."""
    try:
        provider = _build_provider(
            args.provider,
            laya_variant=getattr(args, "laya_variant", None),
            laya_model=getattr(args, "laya_model", None),
        )
    except Exception as exc:
        print(f"Error configuring provider: {exc}", file=sys.stderr)
        return 1

    dataset = get_sample_dataset()
    print(f"Fitting calibration parameters for provider '{provider.name}'...")
    examples = [(ex.request, ex.ground_truth) for ex in dataset.examples]

    try:
        reports = fit_from_provider_and_dataset(provider, examples, method=args.method)
        print()
        print(f"# Jevlaya Calibration Results ({provider.name})\n")
        for q_id, rep in reports.items():
            print(f"## Question: `{q_id}`")
            print(rep.summary_markdown())
        return 0
    except Exception as exc:
        print(f"Calibration fitting failed: {exc}", file=sys.stderr)
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
        default=os.environ.get("JEVLAYA_HOST", "0.0.0.0"),
        help="Host interface to bind (default: 0.0.0.0, env: JEVLAYA_HOST)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("JEVLAYA_PORT", "8000")),
        help="Port to listen on (default: 8000, env: JEVLAYA_PORT)",
    )
    serve_parser.add_argument(
        "--provider",
        default=os.environ.get("JEVLAYA_PROVIDER", "mock"),
        help="Default provider: mock, laya, jev. All are registered & selectable via ?provider=",
    )
    serve_parser.add_argument(
        "--laya-variant",
        choices=["base", "multilingual", "typed"],
        default=os.environ.get("LAYA_VARIANT", "base"),
        help="Laya model variant: base, multilingual, typed (default: base, env: LAYA_VARIANT)",
    )
    serve_parser.add_argument(
        "--laya-model",
        default=os.environ.get("LAYA_MODEL", None),
        help="Custom Laya HuggingFace model repo or local path (env: LAYA_MODEL)",
    )
    serve_parser.add_argument(
        "--telemetry-file",
        default=os.environ.get("JEVLAYA_TELEMETRY_FILE", None),
        help="Path to JSONL file to persist decision events (env: JEVLAYA_TELEMETRY_FILE)",
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
        "--laya-variant",
        choices=["base", "multilingual", "typed"],
        default=os.environ.get("LAYA_VARIANT", "base"),
        help="Laya model variant: base, multilingual, typed (default: base, env: LAYA_VARIANT)",
    )
    bench_parser.add_argument(
        "--laya-model",
        default=os.environ.get("LAYA_MODEL", None),
        help="Custom Laya HuggingFace model repo or local path (env: LAYA_MODEL)",
    )
    bench_parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup samples before recording metrics (default: 0)",
    )

    # --- Command: calibrate ---
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Fit model calibration parameters (temperature scaling) on sample or evaluation data",
    )
    calibrate_parser.add_argument(
        "--provider",
        default="mock",
        help="Provider to calibrate: mock, laya, jev (default: mock)",
    )
    calibrate_parser.add_argument(
        "--method",
        choices=["temperature"],
        default="temperature",
        help="Calibration fitting method (default: temperature)",
    )
    calibrate_parser.add_argument(
        "--laya-variant",
        choices=["base", "multilingual", "typed"],
        default=os.environ.get("LAYA_VARIANT", "base"),
        help="Laya model variant: base, multilingual, typed (default: base, env: LAYA_VARIANT)",
    )
    calibrate_parser.add_argument(
        "--laya-model",
        default=os.environ.get("LAYA_MODEL", None),
        help="Custom Laya HuggingFace model repo or local path (env: LAYA_MODEL)",
    )

    args = parser.parse_args(argv)

    if args.command == "serve":
        return run_serve(args)
    if args.command == "bench":
        return run_bench(args)
    if args.command == "calibrate":
        return run_calibrate(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
