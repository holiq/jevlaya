"""Policy router executing intelligent multi-provider routing, confidence gating, and fallbacks."""

from __future__ import annotations

import time
from typing import Any

from jevlaya.gateway.gateway import DecisionGateway
from jevlaya.protocol.models import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
)
from jevlaya.routing.policy import RoutingMetadata, RoutingPolicy


class PolicyRouter:
    """Orchestrates decision requests according to a declarative RoutingPolicy."""

    def __init__(self, gateway: DecisionGateway, policy: RoutingPolicy) -> None:
        self.gateway = gateway
        self.policy = policy

    def _extract_min_confidence(self, response: DecisionResponse) -> float:
        """Extract the lowest confidence score across typed question answers."""
        confidences: list[float] = [
            ans.confidence
            for ans in response.answers.values()
            if isinstance(ans, (ChoiceAnswer, ScoreAnswer))
        ]
        return min(confidences) if confidences else 1.0

    def decide(self, request: DecisionRequest | dict[str, Any]) -> DecisionResponse:
        """Execute request under policy rules, invoking fallback on errors or low confidence."""
        primary = self.policy.primary
        fallback = self.policy.fallback
        policy_ver = self.policy.policy_version

        # 1. Attempt primary provider
        primary_response: DecisionResponse | None = None
        primary_error: Exception | None = None

        try:
            primary_response = self.gateway.decide(request, provider_name=primary)
        except Exception as exc:
            primary_error = exc
            if self.policy.on_provider_error != "fallback" or not fallback:
                raise

        # Handle provider error fallback
        if primary_error is not None and fallback:
            fb_start = time.perf_counter()
            fb_resp = self.gateway.decide(request, provider_name=fallback)
            fb_latency = (time.perf_counter() - fb_start) * 1000.0

            metadata = RoutingMetadata(
                initial_provider=primary,
                final_provider=fallback,
                fallback_used=True,
                fallback_provider=fallback,
                fallback_reason=f"provider_error: {type(primary_error).__name__}",
                policy_version=policy_ver,
                fallback_latency_ms=round(fb_latency, 2),
            )
            return fb_resp.model_copy(update={"routing": metadata.to_dict()})

        assert primary_response is not None

        # 2. Check confidence thresholds
        min_conf = self._extract_min_confidence(primary_response)
        low_confidence = (
            min_conf < self.policy.min_confidence
            and bool(fallback)
            and self.policy.on_uncertain == "fallback"
        )

        if low_confidence and fallback:
            fb_start = time.perf_counter()
            fb_resp = self.gateway.decide(request, provider_name=fallback)
            fb_latency = (time.perf_counter() - fb_start) * 1000.0

            metadata = RoutingMetadata(
                initial_provider=primary,
                final_provider=fallback,
                fallback_used=True,
                fallback_provider=fallback,
                fallback_reason=(
                    f"low_confidence: min_observed ({round(min_conf, 3)}) "
                    f"< min_required ({self.policy.min_confidence})"
                ),
                min_confidence_observed=round(min_conf, 3),
                policy_version=policy_ver,
                fallback_latency_ms=round(fb_latency, 2),
            )
            return fb_resp.model_copy(update={"routing": metadata.to_dict()})

        # 3. Primary accepted
        metadata = RoutingMetadata(
            initial_provider=primary,
            final_provider=primary,
            fallback_used=False,
            min_confidence_observed=round(min_conf, 3),
            policy_version=policy_ver,
        )
        return primary_response.model_copy(update={"routing": metadata.to_dict()})

    async def adecide(self, request: DecisionRequest | dict[str, Any]) -> DecisionResponse:
        """Asynchronously execute request under policy rules."""
        primary = self.policy.primary
        fallback = self.policy.fallback
        policy_ver = self.policy.policy_version

        # 1. Attempt primary provider
        primary_response: DecisionResponse | None = None
        primary_error: Exception | None = None

        try:
            primary_response = await self.gateway.adecide(request, provider_name=primary)
        except Exception as exc:
            primary_error = exc
            if self.policy.on_provider_error != "fallback" or not fallback:
                raise

        # Handle provider error fallback
        if primary_error is not None and fallback:
            fb_start = time.perf_counter()
            fb_resp = await self.gateway.adecide(request, provider_name=fallback)
            fb_latency = (time.perf_counter() - fb_start) * 1000.0

            metadata = RoutingMetadata(
                initial_provider=primary,
                final_provider=fallback,
                fallback_used=True,
                fallback_provider=fallback,
                fallback_reason=f"provider_error: {type(primary_error).__name__}",
                policy_version=policy_ver,
                fallback_latency_ms=round(fb_latency, 2),
            )
            return fb_resp.model_copy(update={"routing": metadata.to_dict()})

        assert primary_response is not None

        # 2. Check confidence thresholds
        min_conf = self._extract_min_confidence(primary_response)
        low_confidence = (
            min_conf < self.policy.min_confidence
            and bool(fallback)
            and self.policy.on_uncertain == "fallback"
        )

        if low_confidence and fallback:
            fb_start = time.perf_counter()
            fb_resp = await self.gateway.adecide(request, provider_name=fallback)
            fb_latency = (time.perf_counter() - fb_start) * 1000.0

            metadata = RoutingMetadata(
                initial_provider=primary,
                final_provider=fallback,
                fallback_used=True,
                fallback_provider=fallback,
                fallback_reason=(
                    f"low_confidence: min_observed ({round(min_conf, 3)}) "
                    f"< min_required ({self.policy.min_confidence})"
                ),
                min_confidence_observed=round(min_conf, 3),
                policy_version=policy_ver,
                fallback_latency_ms=round(fb_latency, 2),
            )
            return fb_resp.model_copy(update={"routing": metadata.to_dict()})

        # 3. Primary accepted
        metadata = RoutingMetadata(
            initial_provider=primary,
            final_provider=primary,
            fallback_used=False,
            min_confidence_observed=round(min_conf, 3),
            policy_version=policy_ver,
        )
        return primary_response.model_copy(update={"routing": metadata.to_dict()})
