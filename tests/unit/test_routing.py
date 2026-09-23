"""Unit tests for Jevlaya PolicyRouter and routing policies."""

import pytest

from jevlaya.errors import ProviderUnavailable
from jevlaya.gateway.gateway import DecisionGateway
from jevlaya.protocol.models import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    NoulAnswer,
    NoulQuestion,
)
from jevlaya.providers.mock import MockProvider
from jevlaya.routing.policy import RoutingPolicy
from jevlaya.routing.router import PolicyRouter


def _sample_request() -> DecisionRequest:
    return DecisionRequest(
        state={"ticket_id": "T-100"},
        questions={
            "triage": ChoiceQuestion(
                instructions="Route request",
                criteria={"support": "Customer support", "sales": "Sales"},
            ),
            "urgent": NoulQuestion(instructions="Escalate immediately?"),
        },
    )


def test_routing_primary_accepted_when_confidence_high() -> None:
    """Primary provider is accepted when observed confidence meets min_confidence."""
    # MockProvider default confidence is 1.0
    primary_provider = MockProvider(name="fast_local")
    fallback_provider = MockProvider(name="cloud_fallback")

    gw = DecisionGateway(
        providers={"fast_local": primary_provider, "cloud_fallback": fallback_provider},
        default_provider="fast_local",
    )
    policy = RoutingPolicy(
        primary="fast_local",
        fallback="cloud_fallback",
        min_confidence=0.85,
        on_uncertain="fallback",
    )
    router = PolicyRouter(gateway=gw, policy=policy)

    req = _sample_request()
    resp = router.decide(req)

    assert resp.provider == "fast_local"
    assert resp.routing is not None
    assert resp.routing["fallback_used"] is False
    assert resp.routing["initial_provider"] == "fast_local"
    assert resp.routing["final_provider"] == "fast_local"


def test_routing_fallback_on_low_confidence() -> None:
    """Router triggers fallback provider when primary confidence is below threshold."""
    # Low confidence primary response (confidence = 0.6)
    low_conf_answer = ChoiceAnswer(
        choice="support",
        probabilities={"support": 0.6, "sales": 0.4},
        confidence=0.6,
    )
    noul_answer = NoulAnswer(noul=0.5)  # certainty = 0.0

    primary_provider = MockProvider(
        name="fast_local",
        preset_answers={"triage": low_conf_answer, "urgent": noul_answer},
    )
    fallback_provider = MockProvider(name="cloud_fallback")

    gw = DecisionGateway(
        providers={"fast_local": primary_provider, "cloud_fallback": fallback_provider},
    )
    policy = RoutingPolicy(
        primary="fast_local",
        fallback="cloud_fallback",
        min_confidence=0.85,
        on_uncertain="fallback",
    )
    router = PolicyRouter(gateway=gw, policy=policy)

    req = _sample_request()
    resp = router.decide(req)

    # Must have fallen back to cloud_fallback
    assert resp.provider == "cloud_fallback"
    assert resp.routing is not None
    assert resp.routing["fallback_used"] is True
    assert resp.routing["initial_provider"] == "fast_local"
    assert resp.routing["final_provider"] == "cloud_fallback"
    assert "low_confidence" in resp.routing["fallback_reason"]


def test_routing_fallback_on_provider_error() -> None:
    """Router falls back to secondary provider when primary raises a provider error."""
    failing_primary = MockProvider(
        name="failing_local",
        simulate_error=ProviderUnavailable("GPU out of memory or driver fault"),
    )
    fallback_provider = MockProvider(name="cloud_fallback")

    gw = DecisionGateway(
        providers={"failing_local": failing_primary, "cloud_fallback": fallback_provider},
    )
    policy = RoutingPolicy(
        primary="failing_local",
        fallback="cloud_fallback",
        on_provider_error="fallback",
    )
    router = PolicyRouter(gateway=gw, policy=policy)

    req = _sample_request()
    resp = router.decide(req)

    assert resp.provider == "cloud_fallback"
    assert resp.routing is not None
    assert resp.routing["fallback_used"] is True
    assert resp.routing["initial_provider"] == "failing_local"
    assert resp.routing["final_provider"] == "cloud_fallback"
    assert "provider_error" in resp.routing["fallback_reason"]


@pytest.mark.asyncio
async def test_async_routing_adecide() -> None:
    """Asynchronous adecide routes and records metadata correctly."""
    primary_provider = MockProvider(name="primary_async")
    fallback_provider = MockProvider(name="fallback_async")

    gw = DecisionGateway(
        providers={"primary_async": primary_provider, "fallback_async": fallback_provider},
    )
    policy = RoutingPolicy(primary="primary_async", fallback="fallback_async")
    router = PolicyRouter(gateway=gw, policy=policy)

    req = _sample_request()
    resp = await router.adecide(req)

    assert resp.provider == "primary_async"
    assert resp.routing is not None
    assert resp.routing["fallback_used"] is False
