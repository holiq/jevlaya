"""Integration tests for LayaAdapter, capability declarations, and normalization."""

from typing import Any

import pytest

from jevlaya.errors import (
    NormalizationError,
    ProviderResponseError,
    ProviderUnavailable,
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
from jevlaya.providers import DecisionProvider, LayaAdapter, LayaCapabilities


class FakeLayaRouter:
    """Mock router simulating upstream Laya Router behavior without model weights."""

    def __init__(self, canned_response: dict[str, Any] | None = None) -> None:
        self.canned_response = canned_response
        self.last_state: dict[str, Any] | None = None
        self.last_questions: dict[str, Any] | None = None

    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        self.last_state = state
        self.last_questions = questions

        if self.canned_response is not None:
            return self.canned_response

        # Default realistic Laya response structure
        answers: dict[str, Any] = {}
        for q_id, q_data in questions.items():
            q_type = q_data.get("type")
            if q_type == "choice":
                first_crit = next(iter(q_data["criteria"].keys()))
                answers[q_id] = {
                    "choice": first_crit,
                    "confidence": 0.94,
                    "probabilities": {
                        k: 0.94 if k == first_crit else 0.06 for k in q_data["criteria"]
                    },
                }
            elif q_type == "score":
                first_level = q_data["criteria"][0]
                answers[q_id] = {
                    "score": first_level,
                    "confidence": 0.88,
                    "probabilities": [0.88]
                    + [0.12 / (len(q_data["criteria"]) - 1)] * (len(q_data["criteria"]) - 1),
                }
            elif q_type == "noul":
                answers[q_id] = {"noul": 0.72}

        return {
            "answers": answers,
            "routing": {
                "model": "english",
                "repo": "convaiinnovations/laya",
                "reason": "English text detected",
            },
        }


def test_laya_capabilities_declaration() -> None:
    """LayaAdapter exposes frozen capabilities matching known specs."""
    adapter = LayaAdapter(router=FakeLayaRouter())
    caps = adapter.capabilities

    assert isinstance(caps, LayaCapabilities)
    assert caps.supports_choice is True
    assert caps.supports_score is True
    assert caps.supports_noul is True
    assert caps.supports_multilingual is True
    assert caps.local_inference is True
    assert caps.max_recommended_options == 20


def test_laya_protocol_conformance() -> None:
    """LayaAdapter satisfies DecisionProvider protocol."""
    adapter = LayaAdapter(router=FakeLayaRouter())
    assert isinstance(adapter, DecisionProvider)
    assert adapter.name == "laya"


def test_request_translation() -> None:
    """DecisionRequest is correctly translated to Laya's state/questions payload."""
    router = FakeLayaRouter()
    adapter = LayaAdapter(router=router)

    req = DecisionRequest(
        state={"body": "Hello world"},
        questions={
            "q_choice": ChoiceQuestion(
                instructions="Pick topic",
                criteria={"a": "Opt A", "b": "Opt B"},
            ),
            "q_score": ScoreQuestion(
                instructions="Rate quality",
                criteria=["poor", "good"],
            ),
            "q_noul": NoulQuestion(instructions="Escalate?"),
        },
    )

    state, questions = adapter.translate_request(req)
    assert state == {"body": "Hello world"}
    assert questions["q_choice"] == {
        "type": "choice",
        "instructions": "Pick topic",
        "criteria": {"a": "Opt A", "b": "Opt B"},
    }
    assert questions["q_score"] == {
        "type": "score",
        "instructions": "Rate quality",
        "criteria": ["poor", "good"],
    }
    assert questions["q_noul"] == {
        "type": "noul",
        "instructions": "Escalate?",
    }


def test_decide_and_normalization() -> None:
    """LayaAdapter executes decision and normalizes all primitives."""
    router = FakeLayaRouter()
    adapter = LayaAdapter(router=router)

    req = DecisionRequest(
        state={"ticket": "T-100"},
        questions={
            "dept": ChoiceQuestion(
                instructions="Select dept",
                criteria={"billing": "Billing", "tech": "Technical"},
            ),
            "severity": ScoreQuestion(
                instructions="Severity",
                criteria=["low", "high"],
            ),
            "alert": NoulQuestion(instructions="Send alert?"),
        },
    )

    res = adapter.decide(req)

    assert isinstance(res, DecisionResponse)
    assert res.provider == "laya"
    assert "english" in res.model
    assert isinstance(res.answers["dept"], ChoiceAnswer)
    assert res.answers["dept"].choice == "billing"
    assert res.answers["dept"].confidence == 0.94
    assert isinstance(res.answers["severity"], ScoreAnswer)
    assert res.answers["severity"].score == "low"
    assert isinstance(res.answers["alert"], NoulAnswer)
    assert res.answers["alert"].noul == 0.72


def test_missing_laya_package_raises_provider_unavailable() -> None:
    """When router is not injected and package cannot be imported, raise ProviderUnavailable."""
    adapter = LayaAdapter(router=None)

    # In test environment, laya is not installed
    with pytest.raises(ProviderUnavailable, match="The 'laya' package is required"):
        _ = adapter.router


def test_router_prediction_failure_raises_provider_response_error() -> None:
    """Underlying router exceptions during predict are mapped to ProviderResponseError."""

    class BrokenRouter:
        def predict(self, state: Any, questions: Any) -> Any:
            raise RuntimeError("CUDA out of memory")

    adapter = LayaAdapter(router=BrokenRouter())
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(ProviderResponseError, match="CUDA out of memory"):
        adapter.decide(req)


def test_corrupt_router_output_raises_normalization_error() -> None:
    """Missing answers dictionary or non-dict output triggers NormalizationError."""
    bad_router = FakeLayaRouter(canned_response={"unexpected": 123})
    adapter = LayaAdapter(router=bad_router)
    req = DecisionRequest(questions={"q": NoulQuestion(instructions="Test")})

    with pytest.raises(NormalizationError, match="missing valid 'answers' dictionary"):
        adapter.decide(req)


def test_gateway_with_laya_adapter() -> None:
    """DecisionGateway operates transparently with LayaAdapter."""
    router = FakeLayaRouter()
    adapter = LayaAdapter(router=router)
    gateway = DecisionGateway(provider=adapter)

    req = DecisionRequest(
        questions={
            "esc": NoulQuestion(instructions="Escalate?"),
        }
    )

    res = gateway.decide(req)
    assert res.provider == "laya"
    assert isinstance(res.answers["esc"], NoulAnswer)
    assert res.latency_ms >= 0.0
