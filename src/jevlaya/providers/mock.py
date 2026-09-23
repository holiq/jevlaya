"""Mock provider implementation for testing and contract verification."""

from __future__ import annotations

import time
from typing import Any

from jevlaya.protocol.models import (
    Answer,
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


class MockProvider:
    """Configurable mock provider that generates canonical responses."""

    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-v1",
        answers: dict[str, Answer] | None = None,
        preset_answers: dict[str, Answer] | None = None,
        simulate_error: Exception | list[Exception | None] | None = None,
        simulate_latency_ms: float = 0.0,
    ) -> None:
        self.name = name
        self.model = model
        self.preset_answers = preset_answers if preset_answers is not None else answers
        self._simulate_errors: list[Exception | None] | None = (
            list(simulate_error) if isinstance(simulate_error, list) else None
        )
        self.simulate_error = simulate_error if not isinstance(simulate_error, list) else None
        self.simulate_latency_ms = simulate_latency_ms
        self.calls: list[DecisionRequest] = []

    def reset(self) -> None:
        """Reset processed request history and error sequences."""
        self.calls.clear()

    @property
    def call_count(self) -> int:
        """Return the number of requests processed."""
        return len(self.calls)

    @property
    def last_request(self) -> DecisionRequest | None:
        """Return the most recently processed request."""
        return self.calls[-1] if self.calls else None

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Process a request, generating mock answers or raising simulated errors."""
        self.calls.append(request)

        # 1. Sequential error simulation
        if self._simulate_errors:
            err = self._simulate_errors.pop(0)
            if err is not None:
                raise err
        elif self.simulate_error is not None:
            raise self.simulate_error

        if self.simulate_latency_ms > 0:
            time.sleep(self.simulate_latency_ms / 1000.0)

        answers: dict[str, Answer] = {}

        if self.preset_answers is not None:
            answers = dict(self.preset_answers)
        else:
            for q_id, question in request.questions.items():
                answers[q_id] = self._generate_default_answer(question)

        return DecisionResponse(
            provider=self.name,
            model=self.model,
            answers=answers,
            usage=UsageInfo(input_tokens=10, output_tokens=5, cost_usd=0.0),
            latency_ms=max(self.simulate_latency_ms, 1.0),
        )

    def _generate_default_answer(self, question: Any) -> Answer:
        """Generate a valid canonical answer for a question primitive."""
        if isinstance(question, ChoiceQuestion):
            first_key = next(iter(question.criteria.keys()))
            probs = {k: 1.0 if k == first_key else 0.0 for k in question.criteria}
            return ChoiceAnswer(
                type="choice",
                choice=first_key,
                probabilities=probs,
                confidence=1.0,
            )

        if isinstance(question, ScoreQuestion):
            first_level = question.criteria[0]
            probs = [1.0] + [0.0] * (len(question.criteria) - 1)
            return ScoreAnswer(
                type="score",
                score=first_level,
                probabilities=probs,
                confidence=1.0,
            )

        if isinstance(question, NoulQuestion):
            return NoulAnswer(type="noul", noul=0.5)

        raise ValueError(f"Unknown question primitive: {type(question)}")
