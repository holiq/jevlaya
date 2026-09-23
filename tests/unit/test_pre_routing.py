"""Unit tests for constraint-based pre-routing and confidence scaling in PolicyRouter."""

import pytest

from jevlaya.gateway.gateway import DecisionGateway
from jevlaya.protocol.models import ChoiceAnswer, ChoiceQuestion, DecisionRequest
from jevlaya.providers.mock import MockProvider
from jevlaya.routing.policy import RoutingPolicy
from jevlaya.routing.router import PolicyRouter


def test_pre_routing_bypasses_when_cardinality_exceeds_limit() -> None:
    """Request with >20 options directly bypasses primary to fallback provider."""
    primary = MockProvider(name="laya_local")
    fallback = MockProvider(name="jev_cloud")

    gw = DecisionGateway(providers={"laya_local": primary, "jev_cloud": fallback})
    policy = RoutingPolicy(
        primary="laya_local",
        fallback="jev_cloud",
        max_primary_cardinality=20,
    )
    router = PolicyRouter(gateway=gw, policy=policy)

    # 25 options (exceeds Laya's 20 max recommended options)
    criteria_25 = {f"opt_{i}": f"Criteria description {i}" for i in range(25)}
    req = DecisionRequest(
        state={"text": "Classify this complex ticket"},
        questions={
            "tag": ChoiceQuestion(
                instructions="Select appropriate category",
                criteria=criteria_25,
            )
        },
    )

    resp = router.decide(req)

    # Must be served by fallback without primary being invoked
    assert resp.provider == "jev_cloud"
    assert primary.call_count == 0
    assert fallback.call_count == 1
    assert resp.routing is not None
    assert resp.routing["fallback_used"] is True
    assert resp.routing["pre_routing_bypass"] is True
    assert "cardinality_exceeded: 25 > 20" in resp.routing["bypass_reason"]


def test_pre_routing_bypasses_when_context_exceeds_limit() -> None:
    """Request with oversized state context directly bypasses primary to fallback."""
    primary = MockProvider(name="laya_local")
    fallback = MockProvider(name="jev_cloud")

    gw = DecisionGateway(providers={"laya_local": primary, "jev_cloud": fallback})
    policy = RoutingPolicy(
        primary="laya_local",
        fallback="jev_cloud",
        max_primary_context_chars=1000,
    )
    router = PolicyRouter(gateway=gw, policy=policy)

    # Large context text (~3000 chars)
    large_text = "Detailed incident log report " * 100
    req = DecisionRequest(
        state={"raw_document": large_text},
        questions={
            "tag": ChoiceQuestion(
                instructions="Select severity",
                criteria={"low": "Low impact", "high": "High impact"},
            )
        },
    )

    resp = router.decide(req)

    assert resp.provider == "jev_cloud"
    assert primary.call_count == 0
    assert fallback.call_count == 1
    assert resp.routing is not None
    assert resp.routing["pre_routing_bypass"] is True
    assert "context_exceeded" in resp.routing["bypass_reason"]


def test_pre_routing_allows_normal_request_to_primary() -> None:
    """Request within limits routes normally to primary provider."""
    primary = MockProvider(name="laya_local")
    fallback = MockProvider(name="jev_cloud")

    gw = DecisionGateway(providers={"laya_local": primary, "jev_cloud": fallback})
    policy = RoutingPolicy(
        primary="laya_local",
        fallback="jev_cloud",
        max_primary_cardinality=20,
        max_primary_context_chars=2000,
    )
    router = PolicyRouter(gateway=gw, policy=policy)

    req = DecisionRequest(
        state={"ticket_id": "T-100"},
        questions={
            "dept": ChoiceQuestion(
                instructions="Route to team",
                criteria={"billing": "Billing", "support": "Support"},
            )
        },
    )

    resp = router.decide(req)

    assert resp.provider == "laya_local"
    assert primary.call_count == 1
    assert fallback.call_count == 0
    assert resp.routing is not None
    assert resp.routing["fallback_used"] is False


def test_confidence_scaling_calibration() -> None:
    """Calibration scaling factor scales raw confidence before threshold check."""
    # Raw confidence is 0.85
    preset_answer = ChoiceAnswer(
        choice="billing",
        probabilities={"billing": 0.85, "support": 0.15},
        confidence=0.85,
    )
    primary = MockProvider(name="laya_local", preset_answers={"dept": preset_answer})
    fallback = MockProvider(name="jev_cloud")

    gw = DecisionGateway(providers={"laya_local": primary, "jev_cloud": fallback})
    # If unscaled, 0.85 >= 0.80 would pass.
    # With scaling 0.90: effective = 0.85 * 0.90 = 0.765 < 0.80 -> triggers fallback!
    policy = RoutingPolicy(
        primary="laya_local",
        fallback="jev_cloud",
        min_confidence=0.80,
        confidence_scaling={"laya_local": 0.90},
    )
    router = PolicyRouter(gateway=gw, policy=policy)

    req = DecisionRequest(
        questions={
            "dept": ChoiceQuestion(
                instructions="Route to team",
                criteria={"billing": "Billing", "support": "Support"},
            )
        }
    )

    resp = router.decide(req)

    assert resp.provider == "jev_cloud"
    assert resp.routing is not None
    assert resp.routing["fallback_used"] is True
    assert "low_confidence" in resp.routing["fallback_reason"]


@pytest.mark.asyncio
async def test_async_pre_routing_bypass() -> None:
    """Async adecide performs pre-routing bypass when constraints are exceeded."""
    primary = MockProvider(name="laya_local")
    fallback = MockProvider(name="jev_cloud")

    gw = DecisionGateway(providers={"laya_local": primary, "jev_cloud": fallback})
    policy = RoutingPolicy(
        primary="laya_local",
        fallback="jev_cloud",
        max_primary_cardinality=10,
    )
    router = PolicyRouter(gateway=gw, policy=policy)

    criteria_15 = {f"opt_{i}": f"Criteria {i}" for i in range(15)}
    req = DecisionRequest(
        questions={
            "tag": ChoiceQuestion(instructions="Tag", criteria=criteria_15)
        }
    )

    resp = await router.adecide(req)
    assert resp.provider == "jev_cloud"
    assert resp.routing is not None
    assert resp.routing["pre_routing_bypass"] is True
