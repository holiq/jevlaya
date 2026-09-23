"""Integration tests for the self-hosted Decision Gateway FastAPI server and CLI."""

from fastapi.testclient import TestClient

from jevlaya import __version__
from jevlaya.cli import main
from jevlaya.errors import ProviderTimeout, ProviderUnavailable
from jevlaya.gateway.gateway import DecisionGateway
from jevlaya.providers.mock import MockProvider
from jevlaya.server.app import create_app


def test_server_health_endpoint() -> None:
    """GET /health returns server status and version."""
    app = create_app()
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "version": __version__}


def test_server_list_providers_endpoint() -> None:
    """GET /v1/providers returns active provider registry."""
    gw = DecisionGateway(
        providers={"mock": MockProvider(), "custom": MockProvider(name="custom")},
        default_provider="mock",
    )
    app = create_app(gateway=gw)
    client = TestClient(app)

    res = client.get("/v1/providers")
    assert res.status_code == 200
    body = res.json()
    assert body["default_provider"] == "mock"
    assert "mock" in body["available_providers"]
    assert "custom" in body["available_providers"]


def test_server_submit_decision_success() -> None:
    """POST /v1/decisions evaluates canonical questions and returns normalized decision."""
    app = create_app()
    client = TestClient(app)

    payload = {
        "state": {"ticket_id": "T-100", "body": "Refund my card"},
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "Route to department",
                "criteria": {"billing": "Billing issues", "tech": "Tech support"},
            },
            "urgency": {
                "type": "score",
                "instructions": "Severity",
                "criteria": ["low", "high"],
            },
            "flag": {
                "type": "noul",
                "instructions": "Escalate?",
            },
        },
    }

    res = client.post("/v1/decisions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "mock"
    assert "dept" in data["answers"]
    assert data["answers"]["dept"]["choice"] == "billing"
    assert "urgency" in data["answers"]
    assert data["answers"]["urgency"]["score"] == "low"
    assert "flag" in data["answers"]
    assert data["answers"]["flag"]["noul"] == 0.5
    assert data["latency_ms"] >= 0.0


def test_server_invalid_request_returns_400() -> None:
    """POST /v1/decisions returns 400 when question criteria are invalid."""
    app = create_app()
    client = TestClient(app)

    invalid_payload = {
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "Pick one",
                "criteria": {"single": "Only one option"},
            }
        }
    }

    res = client.post("/v1/decisions", json=invalid_payload)
    assert res.status_code == 422 or res.status_code == 400


def test_server_provider_timeout_returns_504() -> None:
    """POST /v1/decisions returns 504 when provider request times out."""
    failing_provider = MockProvider(
        simulate_error=ProviderTimeout("Simulated upstream gateway timeout")
    )
    gw = DecisionGateway(provider=failing_provider)
    app = create_app(gateway=gw)
    client = TestClient(app)

    payload = {"questions": {"flag": {"type": "noul", "instructions": "Review?"}}}

    res = client.post("/v1/decisions", json=payload)
    assert res.status_code == 504
    assert res.json()["error"] == "ProviderTimeout"


def test_server_provider_unavailable_returns_503() -> None:
    """POST /v1/decisions returns 503 when provider is unreachable."""
    failing_provider = MockProvider(simulate_error=ProviderUnavailable("Backend service down"))
    gw = DecisionGateway(provider=failing_provider)
    app = create_app(gateway=gw)
    client = TestClient(app)

    payload = {"questions": {"flag": {"type": "noul", "instructions": "Review?"}}}

    res = client.post("/v1/decisions", json=payload)
    assert res.status_code == 503
    assert res.json()["error"] == "ProviderUnavailable"


def test_cli_version_flag(capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI prints version string when requested."""
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_bench_subcommand(capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI runs DecisionBench subcommand successfully."""
    exit_code = main(["bench", "--provider", "mock"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "DecisionBench Report" in captured.out


def test_cli_build_provider_laya_variant() -> None:
    """_build_provider configures LayaAdapter with custom variant and model."""
    from jevlaya.cli import _build_provider
    from jevlaya.providers.laya import LayaAdapter

    provider = _build_provider("laya", laya_variant="multilingual")
    assert isinstance(provider, LayaAdapter)
    assert provider.variant == "multilingual"
    assert provider.model == "convaiinnovations/laya-multilingual"
    assert provider.capabilities.context_window == 1024


def test_cli_serve_help_shows_laya_flags(capsys) -> None:  # type: ignore[no-untyped-def]
    """CLI serve --help lists --laya-variant and --laya-model."""
    try:
        main(["serve", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "--laya-variant" in captured.out
    assert "--laya-model" in captured.out

