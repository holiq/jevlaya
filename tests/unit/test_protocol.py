"""Unit tests for canonical decision protocol models and primitives."""

import pytest
from pydantic import ValidationError

from jevlaya.protocol import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    UsageInfo,
)


def test_canonical_request_parsing() -> None:
    """Verify parsing the exact canonical request from specs/jevlaya-v0.1.md."""
    raw = {
        "state": {"ticket_id": "T-123"},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {
                    "sales": "pricing and contracts",
                    "support": "bugs and issues",
                    "billing": "invoices and payments",
                    "other": "everything else",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["low", "medium", "high"],
            },
            "escalate": {
                "type": "noul",
                "instructions": "Should this be escalated?",
            },
        },
    }

    req = DecisionRequest.model_validate(raw)
    assert req.state == {"ticket_id": "T-123"}
    assert isinstance(req.questions["department"], ChoiceQuestion)
    assert req.questions["department"].criteria["sales"] == "pricing and contracts"
    assert isinstance(req.questions["urgency"], ScoreQuestion)
    assert req.questions["urgency"].criteria == ["low", "medium", "high"]
    assert isinstance(req.questions["escalate"], NoulQuestion)


def test_canonical_response_parsing() -> None:
    """Verify parsing the exact canonical response from specs/jevlaya-v0.1.md."""
    raw = {
        "provider": "example",
        "model": "example-model",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "support",
                "probabilities": {
                    "sales": 0.03,
                    "support": 0.92,
                    "billing": 0.03,
                    "other": 0.02,
                },
                "confidence": 0.92,
            },
            "urgency": {
                "type": "score",
                "score": "medium",
                "probabilities": [0.10, 0.85, 0.05],
                "confidence": 0.85,
            },
            "escalate": {
                "type": "noul",
                "noul": 0.12,
            },
        },
        "usage": {
            "input_tokens": 120,
            "output_tokens": 15,
            "cost_usd": 0.0004,
        },
        "latency_ms": 31.2,
    }

    res = DecisionResponse.model_validate(raw)
    assert res.provider == "example"
    assert res.latency_ms == 31.2
    assert isinstance(res.usage, UsageInfo)
    assert res.usage.cost_usd == 0.0004
    assert isinstance(res.answers["department"], ChoiceAnswer)
    assert res.answers["department"].choice == "support"
    assert res.answers["department"].confidence == 0.92
    assert isinstance(res.answers["urgency"], ScoreAnswer)
    assert res.answers["urgency"].score == "medium"
    assert isinstance(res.answers["escalate"], NoulAnswer)
    assert res.answers["escalate"].noul == 0.12


def test_choice_validation() -> None:
    """Test choice criteria constraints and probability validation."""
    # Criteria < 2 options must fail
    with pytest.raises(ValidationError, match="at least 2 options"):
        ChoiceQuestion(
            instructions="Pick one",
            criteria={"single": "Only one option"},
        )

    # Empty key must fail
    with pytest.raises(ValidationError, match="non-empty strings"):
        ChoiceQuestion(
            instructions="Pick one",
            criteria={"": "Empty key", "b": "Valid option"},
        )

    # Choice answer confidence > 1.0 must fail
    with pytest.raises(ValidationError):
        ChoiceAnswer(
            choice="a",
            probabilities={"a": 1.2, "b": 0.0},
            confidence=1.2,
        )

    # Choice answer when choice is not in probabilities must fail
    with pytest.raises(ValidationError, match="Selected choice 'unknown' must be present"):
        ChoiceAnswer(
            choice="unknown",
            probabilities={"a": 0.6, "b": 0.4},
            confidence=0.6,
        )


def test_score_validation() -> None:
    """Test score criteria ordering and uniqueness constraints."""
    # Criteria < 2 levels must fail
    with pytest.raises(ValidationError, match="at least 2 ordered levels"):
        ScoreQuestion(
            instructions="Rate severity",
            criteria=["only_one"],
        )

    # Duplicate criteria levels must fail
    with pytest.raises(ValidationError, match="levels must be unique"):
        ScoreQuestion(
            instructions="Rate severity",
            criteria=["low", "low"],
        )

    # Probability out of bounds must fail
    with pytest.raises(ValidationError):
        ScoreAnswer(
            score="low",
            probabilities=[-0.1, 1.1],
            confidence=0.5,
        )


def test_noul_validation() -> None:
    """Test binary noul probability bounds."""
    # Valid noul
    ans = NoulAnswer(noul=0.75)
    assert ans.noul == 0.75

    # Out of [0, 1] range must fail
    with pytest.raises(ValidationError):
        NoulAnswer(noul=1.05)

    with pytest.raises(ValidationError):
        NoulAnswer(noul=-0.01)


def test_empty_questions_validation() -> None:
    """DecisionRequest must have at least one question."""
    with pytest.raises(ValidationError, match="at least one question"):
        DecisionRequest(questions={})


def test_unknown_question_primitive_validation() -> None:
    """Unsupported question primitives must fail discriminator validation."""
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({
            "questions": {
                "q1": {
                    "type": "unsupported_primitive",
                    "instructions": "Do something",
                }
            }
        })


def test_json_roundtrip() -> None:
    """Test serialize to JSON and deserialize back."""
    req = DecisionRequest(
        state={"user": "alice"},
        questions={
            "q1": NoulQuestion(instructions="Escalate?"),
        },
    )
    json_str = req.model_dump_json()
    req_loaded = DecisionRequest.model_validate_json(json_str)
    assert req_loaded == req
