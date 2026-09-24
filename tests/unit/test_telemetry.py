"""Unit tests for telemetry module: DecisionEvent, sinks, and DecisionGateway integration."""

from pathlib import Path
from typing import Any

import pytest

from jevlaya.errors import ProviderResponseError
from jevlaya.gateway.gateway import DecisionGateway
from jevlaya.protocol.models import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulQuestion,
    UsageInfo,
)
from jevlaya.providers.mock import MockProvider
from jevlaya.telemetry import (
    CallbackTelemetrySink,
    CompositeTelemetrySink,
    DecisionEvent,
    InMemoryTelemetrySink,
    JsonLinesTelemetrySink,
)


def test_decision_event_creation_from_response() -> None:
    """Test DecisionEvent creation from canonical DecisionResponse."""
    resp = DecisionResponse(
        provider="mock",
        model="mock-v1",
        answers={
            "q1": ChoiceAnswer(choice="A", probabilities={"A": 0.8, "B": 0.2}, confidence=0.8),
        },
        usage=UsageInfo(input_tokens=10, output_tokens=5, cost_usd=0.0001),
        latency_ms=45.2,
        request_id="req-123",
        routing={"fallback_used": True, "fallback_provider": "jev"},
    )

    event = DecisionEvent.from_response(resp)

    assert event.request_id == "req-123"
    assert event.provider == "mock"
    assert event.model == "mock-v1"
    assert event.status == "fallback"
    assert event.latency_ms == 45.2
    assert event.cost_usd == 0.0001
    assert event.input_tokens == 10
    assert event.question_count == 1
    assert event.question_types == ["choice"]


def test_decision_event_creation_from_error_redacts_secrets() -> None:
    """Test DecisionEvent creation from an error masks sensitive keys."""
    raw_error = ValueError("Authentication failed with key sk-abcdef1234567890abcdef1234")
    event = DecisionEvent.from_error(
        provider="jev",
        error=raw_error,
        elapsed_ms=120.0,
    )

    assert event.status == "error"
    assert event.provider == "jev"
    assert event.latency_ms == 120.0
    assert event.error_type == "ValueError"
    assert "sk-" not in (event.error_message or "")
    assert "[REDACTED]" in (event.error_message or "")


def test_in_memory_telemetry_sink_summary() -> None:
    """Test InMemoryTelemetrySink metrics aggregation and percentile computation."""
    sink = InMemoryTelemetrySink()

    # Record 3 successes, 1 fallback, 1 error
    sink.record(
        DecisionEvent(
            provider="laya",
            model="laya-base",
            status="success",
            latency_ms=30.0,
            cost_usd=0.0,
        )
    )
    sink.record(
        DecisionEvent(
            provider="laya",
            model="laya-base",
            status="success",
            latency_ms=40.0,
            cost_usd=0.0,
        )
    )
    sink.record(
        DecisionEvent(
            provider="jev",
            model="jev-1.13",
            status="success",
            latency_ms=250.0,
            cost_usd=0.002,
        )
    )
    sink.record(
        DecisionEvent(
            provider="jev",
            model="jev-1.13",
            status="fallback",
            latency_ms=280.0,
            cost_usd=0.002,
        )
    )
    sink.record(
        DecisionEvent(
            provider="laya",
            model="laya-base",
            status="error",
            latency_ms=10.0,
            error_type="ProviderTimeout",
        )
    )

    summary = sink.get_summary()

    assert summary.total_requests == 5
    assert summary.successful_requests == 3
    assert summary.fallback_requests == 1
    assert summary.error_requests == 1
    assert summary.fallback_rate == 0.20
    assert summary.error_rate == 0.20
    assert summary.total_cost_usd == 0.004
    assert summary.provider_breakdown == {"laya": 3, "jev": 2}
    assert summary.status_breakdown == {"success": 3, "fallback": 1, "error": 1}
    assert summary.p50_latency_ms == 40.0
    assert summary.avg_latency_ms == 122.0

    # Test event retrieval and pagination
    events = sink.get_events(limit=2, offset=0)
    assert len(events) == 2
    assert events[0].status == "error"  # Most recent first


def test_json_lines_telemetry_sink(tmp_path: Path) -> None:
    """Test JsonLinesTelemetrySink file append with secret redaction."""
    log_file = tmp_path / "telemetry.jsonl"
    sink = JsonLinesTelemetrySink(log_file)

    event = DecisionEvent(
        provider="jev",
        model="jev-1.13",
        status="error",
        latency_ms=50.0,
        error_message="Invalid key sk-abcdef1234567890abcdef1234",
    )
    sink.record(event)

    assert log_file.exists()
    content = log_file.read_text()
    assert "sk-" not in content
    assert "[REDACTED]" in content
    assert '"provider":"jev"' in content


def test_composite_and_callback_sinks() -> None:
    """Test CompositeTelemetrySink and CallbackTelemetrySink."""
    recorded_events: list[DecisionEvent] = []
    cb_sink = CallbackTelemetrySink(lambda e: recorded_events.append(e))
    mem_sink = InMemoryTelemetrySink()

    composite = CompositeTelemetrySink([cb_sink, mem_sink])

    event = DecisionEvent(
        provider="mock",
        model="mock-v1",
        latency_ms=10.0,
    )
    composite.record(event)

    assert len(recorded_events) == 1
    assert len(mem_sink.get_events()) == 1


def test_gateway_telemetry_integration() -> None:
    """Test DecisionGateway automatically logs events to its configured telemetry sink."""
    sink = InMemoryTelemetrySink()
    mock_prov = MockProvider()
    gw = DecisionGateway(provider=mock_prov, telemetry=sink)

    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test condition?")})

    # Successful call
    resp = gw.decide(req)
    assert resp.provider == "mock"

    events = sink.get_events()
    assert len(events) == 1
    assert events[0].provider == "mock"
    assert events[0].status == "success"
    assert events[0].question_types == ["noul"]

    # Error call
    class FailingProvider:
        name = "failing"

        def decide(self, request: Any) -> Any:
            raise RuntimeError("Backend connection refused")

    gw.register_provider(FailingProvider(), set_default=True)

    with pytest.raises(ProviderResponseError):
        gw.decide(req)

    events = sink.get_events()
    assert len(events) == 2
    assert events[0].status == "error"
    assert events[0].provider == "failing"
    assert events[0].error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_gateway_async_telemetry_integration() -> None:
    """Test DecisionGateway adecide emits telemetry events."""
    sink = InMemoryTelemetrySink()
    mock_prov = MockProvider()
    gw = DecisionGateway(provider=mock_prov, telemetry=sink)

    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test condition?")})
    resp = await gw.adecide(req)

    assert resp.provider == "mock"
    events = sink.get_events()
    assert len(events) == 1
    assert events[0].status == "success"
