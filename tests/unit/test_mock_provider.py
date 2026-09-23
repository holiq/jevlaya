"""Unit tests for MockProvider behavior and inspection capabilities."""

import pytest

from jevlaya.errors import ProviderTimeout
from jevlaya.protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from jevlaya.providers import MockProvider


def test_mock_provider_default_answers() -> None:
    """MockProvider generates canonical default answers for all question primitives."""
    provider = MockProvider()
    request = DecisionRequest(
        state={"session_id": "s-1"},
        questions={
            "dept": ChoiceQuestion(
                instructions="Select dept",
                criteria={"sales": "Sales", "tech": "Tech support"},
            ),
            "priority": ScoreQuestion(
                instructions="Priority",
                criteria=["low", "medium", "high"],
            ),
            "flag": NoulQuestion(instructions="Needs review?"),
        },
    )

    response = provider.decide(request)
    assert response.provider == "mock"
    assert provider.call_count == 1
    assert provider.last_request == request

    assert isinstance(response.answers["dept"], ChoiceAnswer)
    assert response.answers["dept"].choice == "sales"
    assert response.answers["dept"].confidence == 1.0

    assert isinstance(response.answers["priority"], ScoreAnswer)
    assert response.answers["priority"].score == "low"
    assert response.answers["priority"].confidence == 1.0

    assert isinstance(response.answers["flag"], NoulAnswer)
    assert response.answers["flag"].noul == 0.5


def test_mock_provider_preset_answers() -> None:
    """MockProvider returns user-defined canned answers when provided."""
    preset = {
        "flag": NoulAnswer(type="noul", noul=0.99),
    }
    provider = MockProvider(answers=preset)
    request = DecisionRequest(
        questions={"flag": NoulQuestion(instructions="Escalate?")}
    )

    response = provider.decide(request)
    assert response.answers["flag"].noul == 0.99


def test_mock_provider_error_simulation() -> None:
    """MockProvider raises simulated exceptions when configured."""
    provider = MockProvider(simulate_error=ProviderTimeout("Connection timed out"))
    request = DecisionRequest(
        questions={"q": NoulQuestion(instructions="Check")}
    )

    with pytest.raises(ProviderTimeout, match="Connection timed out"):
        provider.decide(request)
