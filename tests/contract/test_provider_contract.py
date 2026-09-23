"""Reusable provider contract test suite."""

import pytest

from jevlaya.protocol import (
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulQuestion,
    ScoreQuestion,
)
from jevlaya.providers import DecisionProvider, MockProvider


@pytest.fixture
def canonical_request() -> DecisionRequest:
    """Fixture providing a standard decision request exercising all primitives."""
    return DecisionRequest(
        state={"context_id": "c-100"},
        questions={
            "choice_q": ChoiceQuestion(
                instructions="Select topic",
                criteria={"tech": "Technology", "finance": "Finance", "hr": "Human Resources"},
            ),
            "score_q": ScoreQuestion(
                instructions="Assess confidence",
                criteria=["low", "medium", "high"],
            ),
            "noul_q": NoulQuestion(
                instructions="Requires immediate supervisor notification?",
            ),
        },
    )


def test_provider_protocol_conformance() -> None:
    """Ensure MockProvider satisfies DecisionProvider protocol."""
    provider = MockProvider()
    assert isinstance(provider, DecisionProvider)
    assert isinstance(provider.name, str)
    assert len(provider.name) > 0


def test_provider_contract_answers_all_primitives(canonical_request: DecisionRequest) -> None:
    """Every compliant provider must return valid answers for choice, score, and noul."""
    provider = MockProvider()
    response = provider.decide(canonical_request)

    assert isinstance(response, DecisionResponse)
    assert response.provider == provider.name
    assert isinstance(response.model, str)
    assert response.latency_ms >= 0.0

    # Ensure each question has a corresponding valid answer
    for q_id in canonical_request.questions:
        assert q_id in response.answers
        answer = response.answers[q_id]
        question = canonical_request.questions[q_id]
        assert answer.type == question.type
