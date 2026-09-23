"""Canonical request and response models for Jevlaya decision protocol."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BaseProtocolModel(BaseModel):
    """Base model with shared configuration."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# --- Questions (Request Primitives) ---


class ChoiceQuestion(BaseProtocolModel):
    """Question primitive for selecting exactly one option from criteria."""

    type: Literal["choice"] = "choice"
    instructions: str = Field(..., min_length=1)
    criteria: dict[str, str]

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) < 2:
            raise ValueError("Choice criteria must contain at least 2 options")
        for key in v:
            if not key or not key.strip():
                raise ValueError("Choice criteria keys must be non-empty strings")
        return v


class ScoreQuestion(BaseProtocolModel):
    """Question primitive for an ordered rubric scale."""

    type: Literal["score"] = "score"
    instructions: str = Field(..., min_length=1)
    criteria: list[str]

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("Score criteria must contain at least 2 ordered levels")
        if len(set(v)) != len(v):
            raise ValueError("Score criteria levels must be unique")
        for item in v:
            if not item or not item.strip():
                raise ValueError("Score criteria items must be non-empty strings")
        return v


class NoulQuestion(BaseProtocolModel):
    """Question primitive for binary conditions: P(yes)."""

    type: Literal["noul"] = "noul"
    instructions: str = Field(..., min_length=1)


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion,
    Field(discriminator="type"),
]


class DecisionRequest(BaseProtocolModel):
    """Canonical request payload sent to the decision gateway."""

    state: dict[str, Any] = Field(default_factory=dict)
    questions: dict[str, Question]

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v: dict[str, Question]) -> dict[str, Question]:
        if not v:
            raise ValueError("DecisionRequest must contain at least one question")
        for key in v:
            if not key or not key.strip():
                raise ValueError("Question keys must be non-empty strings")
        return v


# --- Answers (Response Primitives) ---


class ChoiceAnswer(BaseProtocolModel):
    """Normalized response for a choice question."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: dict[str, float]) -> dict[str, float]:
        for key, prob in v.items():
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability for '{key}' must be between 0.0 and 1.0, got {prob}")
        return v

    @model_validator(mode="after")
    def validate_choice_in_probabilities(self) -> ChoiceAnswer:
        if self.probabilities and self.choice not in self.probabilities:
            raise ValueError(
                f"Selected choice '{self.choice}' must be present in probabilities distribution"
            )
        return self


class ScoreAnswer(BaseProtocolModel):
    """Normalized response for an ordered score question."""

    type: Literal["score"] = "score"
    score: str
    probabilities: list[float]
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: list[float]) -> list[float]:
        for i, prob in enumerate(v):
            if not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"Probability at index {i} must be between 0.0 and 1.0, got {prob}"
                )
        return v


class NoulAnswer(BaseProtocolModel):
    """Normalized response for a binary condition: model-estimated P(yes)."""

    type: Literal["noul"] = "noul"
    noul: float = Field(..., ge=0.0, le=1.0)


Answer = Annotated[
    ChoiceAnswer | ScoreAnswer | NoulAnswer,
    Field(discriminator="type"),
]


class UsageInfo(BaseProtocolModel):
    """Resource and token usage metadata."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class DecisionResponse(BaseProtocolModel):
    """Canonical response payload emitted by the Decision Gateway."""

    provider: str
    model: str
    answers: dict[str, Answer]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    latency_ms: float = Field(..., ge=0.0)
