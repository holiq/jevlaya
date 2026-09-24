"""Canonical decision telemetry event models."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevlaya.errors import redact_secrets
from jevlaya.protocol.models import DecisionRequest, DecisionResponse


class DecisionEvent(BaseModel):
    """Structured telemetry record emitted for every decision lifecycle event."""

    model_config = ConfigDict(extra="ignore")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    request_id: str | None = None
    provider: str
    model: str
    status: Literal["success", "fallback", "error"] = "success"
    latency_ms: float = Field(..., ge=0.0)
    question_count: int = 0
    question_types: list[str] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    routing: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None

    @classmethod
    def from_response(
        cls,
        response: DecisionResponse,
        request: DecisionRequest | None = None,
        elapsed_ms: float | None = None,
    ) -> DecisionEvent:
        """Create a telemetry event from a successful or fallback DecisionResponse."""
        status: Literal["success", "fallback", "error"] = "success"
        if response.routing and response.routing.get("fallback_used"):
            status = "fallback"

        q_types: list[str] = []
        q_count = len(response.answers)
        if request and request.questions:
            q_types = [q.type for q in request.questions.values()]
            q_count = len(request.questions)
        else:
            q_types = [ans.type for ans in response.answers.values()]

        latency = elapsed_ms if elapsed_ms is not None else response.latency_ms

        return cls(
            request_id=response.request_id or (request.id if request else None),
            provider=response.provider,
            model=response.model,
            status=status,
            latency_ms=round(latency, 2),
            question_count=q_count,
            question_types=q_types,
            input_tokens=response.usage.input_tokens if response.usage else None,
            output_tokens=response.usage.output_tokens if response.usage else None,
            cost_usd=response.usage.cost_usd if response.usage else None,
            routing=response.routing,
        )

    @classmethod
    def from_error(
        cls,
        provider: str,
        error: Exception,
        elapsed_ms: float,
        request: DecisionRequest | None = None,
        model: str = "unknown",
    ) -> DecisionEvent:
        """Create a telemetry event from a failed decision invocation."""
        raw_msg = str(error)
        clean_msg = redact_secrets(raw_msg)

        q_types: list[str] = []
        q_count = 0
        req_id = None
        if request:
            req_id = request.id
            q_count = len(request.questions)
            q_types = [q.type for q in request.questions.values()]

        return cls(
            request_id=req_id,
            provider=provider,
            model=model,
            status="error",
            latency_ms=round(elapsed_ms, 2),
            question_count=q_count,
            question_types=q_types,
            error_type=type(error).__name__,
            error_message=clean_msg,
        )
