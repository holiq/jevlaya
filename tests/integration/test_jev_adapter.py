"""Integration tests for JevAdapter, OpenRouter Decisions API normalization, and secret safety."""

from typing import Any

import pytest

from jevlaya.errors import (
    NormalizationError,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderTimeout,
)
from jevlaya.gateway import DecisionGateway
from jevlaya.protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from jevlaya.providers import DecisionProvider, JevAdapter, JevCapabilities


def make_mock_jev_response() -> dict[str, Any]:
    """Return a realistic OpenRouter Decisions API response."""
    return {
        "id": "gen-dec-1790015143-AIaTutprXsJ5EwohRSjb",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {
            "is_bug": {"type": "noul", "noul": 0.96},
            "team": {
                "type": "choice",
                "choice": "payments",
                "confidence": 0.67,
                "probabilities": {"payments": 0.78, "frontend": 0.22, "account": 0.0},
            },
            "urgency": {
                "type": "score",
                "score": 1.99,
                "confidence": 0.99,
                "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
                "legend": {
                    "0": "Can wait for next release",
                    "1": "Should be fixed this week",
                    "2": "Blocking revenue right now",
                },
            },
        },
        "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
    }


def test_jev_capabilities_declaration() -> None:
    """JevAdapter declares correct capabilities reflecting OpenRouter specs."""
    adapter = JevAdapter(api_key="sk-or-dummy")
    caps = adapter.capabilities

    assert isinstance(caps, JevCapabilities)
    assert caps.supports_choice is True
    assert caps.supports_score is True
    assert caps.supports_noul is True
    assert caps.hosted_api is True
    assert caps.context_window == 32000
    assert caps.max_recommended_options == 255


def test_jev_protocol_conformance() -> None:
    """JevAdapter satisfies DecisionProvider protocol."""
    adapter = JevAdapter(api_key="sk-or-dummy")
    assert isinstance(adapter, DecisionProvider)
    assert adapter.name == "jev"


def test_jev_request_translation() -> None:
    """DecisionRequest is correctly translated to OpenRouter Decisions API payload."""
    adapter = JevAdapter(api_key="sk-or-dummy")
    req = DecisionRequest(
        state={"customer_tier": "enterprise"},
        questions={
            "dept": ChoiceQuestion(
                instructions="Route to team",
                criteria={"support": "Support", "sales": "Sales"},
            ),
            "crit": ScoreQuestion(
                instructions="Urgency",
                criteria=["low", "high"],
            ),
            "flag": NoulQuestion(instructions="Escalate?"),
        },
    )

    payload = adapter.translate_request(req)
    assert payload["model"] == "typesafe/jev-1.13"
    assert payload["state"] == {"customer_tier": "enterprise"}
    assert payload["questions"]["dept"]["type"] == "choice"
    assert payload["questions"]["crit"]["type"] == "score"
    assert payload["questions"]["flag"]["type"] == "noul"


def test_jev_response_normalization() -> None:
    """JevAdapter normalizes raw OpenRouter Decisions API payload into DecisionResponse."""

    def mock_http(endpoint: str, payload: Any, key: str, timeout: float) -> dict[str, Any]:
        return make_mock_jev_response()

    adapter = JevAdapter(api_key="sk-or-test-key", http_client=mock_http)

    req = DecisionRequest(
        state={"user": "bob"},
        questions={
            "is_bug": NoulQuestion(instructions="Bug?"),
            "team": ChoiceQuestion(
                instructions="Team?",
                criteria={"payments": "Pay", "frontend": "Front", "account": "Acc"},
            ),
            "urgency": ScoreQuestion(
                instructions="Urgency?",
                criteria=[
                    "Can wait for next release",
                    "Should be fixed this week",
                    "Blocking revenue right now",
                ],
            ),
        },
    )

    res = adapter.decide(req)

    assert isinstance(res, DecisionResponse)
    assert res.provider == "jev"
    assert res.model == "typesafe/jev-1.13-20260917"

    # Noul answer
    assert isinstance(res.answers["is_bug"], NoulAnswer)
    assert res.answers["is_bug"].noul == 0.96

    # Choice answer
    assert isinstance(res.answers["team"], ChoiceAnswer)
    assert res.answers["team"].choice == "payments"
    assert res.answers["team"].confidence == 0.67
    assert res.answers["team"].probabilities["payments"] == 0.78

    # Score answer mapped from float index 1.99 to criteria[2]
    assert isinstance(res.answers["urgency"], ScoreAnswer)
    assert res.answers["urgency"].score == "Blocking revenue right now"
    assert res.answers["urgency"].confidence == 0.99
    assert res.answers["urgency"].probabilities == [0.0, 0.0, 1.0]

    # Usage
    assert res.usage.cost_usd == 0.000019992
    assert res.usage.input_tokens == 476
    assert res.usage.output_tokens == 70


def test_missing_api_key_raises_auth_error() -> None:
    """Missing API key raises ProviderAuthenticationError."""
    adapter = JevAdapter(api_key="")
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(ProviderAuthenticationError, match="No API key provided"):
        adapter.decide(req)


def test_secret_redaction_in_auth_error() -> None:
    """API key is never leaked in exception traces."""
    secret_key = "sk-12345678901234567890abcdef"

    def failing_http(endpoint: str, payload: Any, key: str, timeout: float) -> Any:
        raise ProviderAuthenticationError(f"Failed using key {key}")

    adapter = JevAdapter(api_key=secret_key, http_client=failing_http)
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        adapter.decide(req)

    assert secret_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_http_timeout_handling() -> None:
    """HTTP client timeout raises ProviderTimeout."""

    def timeout_http(endpoint: str, payload: Any, key: str, timeout: float) -> Any:
        raise ProviderTimeout("Jev request timed out after 15.0s")

    adapter = JevAdapter(api_key="sk-or-valid", http_client=timeout_http)
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(ProviderTimeout, match="timed out"):
        adapter.decide(req)


def test_http_server_error_handling() -> None:
    """HTTP 500 error raises ProviderResponseError."""

    def server_error_http(endpoint: str, payload: Any, key: str, timeout: float) -> Any:
        raise ProviderResponseError("Jev Decisions API error: HTTP 500 Internal Server Error")

    adapter = JevAdapter(api_key="sk-or-valid", http_client=server_error_http)
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(ProviderResponseError, match="HTTP 500"):
        adapter.decide(req)


def test_corrupt_jev_payload_raises_normalization_error() -> None:
    """Missing answers dictionary in Jev response triggers NormalizationError."""

    def corrupt_http(endpoint: str, payload: Any, key: str, timeout: float) -> dict[str, Any]:
        return {"id": "123"}

    adapter = JevAdapter(api_key="sk-or-valid", http_client=corrupt_http)
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(NormalizationError, match="missing 'answers' dictionary"):
        adapter.decide(req)


def test_gateway_with_jev_adapter() -> None:
    """DecisionGateway operates smoothly with JevAdapter."""

    def mock_http(endpoint: str, payload: Any, key: str, timeout: float) -> dict[str, Any]:
        return make_mock_jev_response()

    adapter = JevAdapter(api_key="sk-or-test", http_client=mock_http)
    gateway = DecisionGateway(provider=adapter)

    req = DecisionRequest(
        questions={
            "is_bug": NoulQuestion(instructions="Is it a bug?"),
        }
    )

    res = gateway.decide(req)
    assert res.provider == "jev"
    assert isinstance(res.answers["is_bug"], NoulAnswer)
    assert res.answers["is_bug"].noul == 0.96
    assert res.latency_ms >= 0.0
