"""Unit tests for DecisionGateway validation, orchestration, and latency tracking."""

import pytest

from jevlaya.errors import (
    InvalidRequest,
    NormalizationError,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderUnavailable,
)
from jevlaya.gateway import DecisionGateway
from jevlaya.protocol import DecisionRequest, DecisionResponse, NoulAnswer, NoulQuestion
from jevlaya.providers import MockProvider


def test_gateway_with_single_injected_provider() -> None:
    """Gateway successfully orchestrates decisions using injected provider."""
    mock = MockProvider(name="test-mock")
    gateway = DecisionGateway(provider=mock)

    req = DecisionRequest(questions={"q1": NoulQuestion(instructions="Review?")})
    res = gateway.decide(req)

    assert isinstance(res, DecisionResponse)
    assert res.provider == "test-mock"
    assert res.latency_ms >= 0.0
    assert mock.call_count == 1


def test_gateway_with_raw_dict_request() -> None:
    """Gateway accepts raw Python dictionaries and converts to canonical request."""
    gateway = DecisionGateway(provider=MockProvider())
    raw_payload = {
        "state": {"user_id": 42},
        "questions": {
            "escalate": {
                "type": "noul",
                "instructions": "Escalate ticket?",
            }
        },
    }

    res = gateway.decide(raw_payload)
    assert isinstance(res, DecisionResponse)
    assert isinstance(res.answers["escalate"], NoulAnswer)


def test_gateway_rejects_invalid_request_schema() -> None:
    """Gateway validates payload and raises InvalidRequest on schema error."""
    gateway = DecisionGateway(provider=MockProvider())

    # Missing instructions and criteria < 2
    invalid_payload = {
        "questions": {
            "dept": {
                "type": "choice",
                "criteria": {"single": "only one"},
            }
        }
    }

    with pytest.raises(InvalidRequest):
        gateway.decide(invalid_payload)

    # Invalid input type
    with pytest.raises(InvalidRequest, match="Expected DecisionRequest or dict"):
        gateway.decide("not-a-dict")  # type: ignore[arg-type]


def test_gateway_provider_resolution_and_switching() -> None:
    """Gateway routes to specified provider or falls back to default."""
    p1 = MockProvider(name="laya-mock")
    p2 = MockProvider(name="jev-mock")
    gateway = DecisionGateway(providers={"laya": p1, "jev": p2}, default_provider="laya")

    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    # Default provider
    res1 = gateway.decide(req)
    assert res1.provider == "laya-mock"
    assert p1.call_count == 1

    # Explicit provider selection
    res2 = gateway.decide(req, provider_name="jev")
    assert res2.provider == "jev-mock"
    assert p2.call_count == 1

    # Unregistered provider raises ProviderUnavailable
    with pytest.raises(ProviderUnavailable, match="not registered"):
        gateway.decide(req, provider_name="unknown-provider")


def test_gateway_no_provider_configured() -> None:
    """Gateway raises ProviderUnavailable if no provider is registered."""
    gateway = DecisionGateway()
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(ProviderUnavailable, match="No decision provider is configured"):
        gateway.decide(req)


def test_gateway_preserves_typed_jev_errors() -> None:
    """Gateway re-raises JevlayaError subclasses without wrapping."""
    failing_provider = MockProvider(
        name="failing",
        simulate_error=ProviderAuthenticationError("Invalid API key"),
    )
    gateway = DecisionGateway(provider=failing_provider)
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(ProviderAuthenticationError):
        gateway.decide(req)


def test_gateway_wraps_unexpected_exceptions() -> None:
    """Gateway catches unexpected third-party exceptions and raises ProviderResponseError."""
    broken_provider = MockProvider(
        name="broken",
        simulate_error=RuntimeError("Underlying network connection broke"),
    )
    gateway = DecisionGateway(provider=broken_provider)
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(ProviderResponseError, match="Underlying network connection broke"):
        gateway.decide(req)


def test_gateway_normalizes_dict_response() -> None:
    """Gateway normalizes a dictionary response into a canonical DecisionResponse."""

    class DictProvider:
        name = "dict-provider"

        def decide(self, request: DecisionRequest) -> dict:  # type: ignore[type-arg]
            return {
                "provider": "dict-provider",
                "model": "dict-v1",
                "answers": {"q": {"type": "noul", "noul": 0.8}},
                "latency_ms": 5.0,
            }

    gateway = DecisionGateway(provider=DictProvider())  # type: ignore[arg-type]
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})
    res = gateway.decide(req)

    assert isinstance(res, DecisionResponse)
    assert res.answers["q"].noul == 0.8


def test_gateway_raises_normalization_error_on_corrupt_response() -> None:
    """Gateway raises NormalizationError if provider response is invalid type."""

    class BadProvider:
        name = "bad-provider"

        def decide(self, request: DecisionRequest) -> str:
            return "unexpected-string"

    gateway = DecisionGateway(provider=BadProvider())  # type: ignore[arg-type]
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(NormalizationError, match="invalid response type"):
        gateway.decide(req)
