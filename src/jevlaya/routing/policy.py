"""Routing policy and metadata definitions for intelligent decision orchestration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RoutingPolicy(BaseModel):
    """Configuration governing provider selection, confidence gating, and fallback."""

    model_config = ConfigDict(extra="ignore")

    primary: str
    fallback: str | None = None
    min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    on_uncertain: Literal["fallback", "escalate", "accept"] = "fallback"
    on_provider_error: Literal["fallback", "raise"] = "fallback"
    policy_version: str = "1.0"
    max_primary_cardinality: int = 20
    max_primary_context_chars: int = 2000
    confidence_scaling: dict[str, float] = Field(default_factory=dict)


class RoutingMetadata(BaseModel):
    """Auditable metadata capturing execution routing and fallback path."""

    model_config = ConfigDict(extra="ignore")

    initial_provider: str
    final_provider: str
    fallback_used: bool = False
    fallback_provider: str | None = None
    fallback_reason: str | None = None
    pre_routing_bypass: bool = False
    bypass_reason: str | None = None
    min_confidence_observed: float | None = None
    policy_version: str = "1.0"
    fallback_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert routing metadata to dictionary."""
        return self.model_dump(exclude_none=True)
