"""Decision Gateway: Provider-agnostic entry point for AI agent decisions."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import ValidationError

from jevlaya.errors import (
    InvalidRequest,
    JevlayaError,
    NormalizationError,
    ProviderResponseError,
    ProviderUnavailable,
)
from jevlaya.protocol.models import DecisionRequest, DecisionResponse
from jevlaya.providers.base import DecisionProvider


class DecisionGateway:
    """Gateway orchestrating validation, provider delegation, and latency tracking."""

    def __init__(
        self,
        provider: DecisionProvider | None = None,
        providers: dict[str, DecisionProvider] | None = None,
        default_provider: str | None = None,
    ) -> None:
        self.providers: dict[str, DecisionProvider] = dict(providers or {})

        if provider is not None:
            self.providers[provider.name] = provider
            if default_provider is None:
                default_provider = provider.name

        self.default_provider = default_provider

    def register_provider(self, provider: DecisionProvider, set_default: bool = False) -> None:
        """Register a new provider adapter."""
        self.providers[provider.name] = provider
        if set_default or self.default_provider is None:
            self.default_provider = provider.name

    def get_provider(self, name: str | None = None) -> DecisionProvider:
        """Retrieve a registered provider by name or return the default provider."""
        provider_name = name or self.default_provider

        if not provider_name:
            raise ProviderUnavailable("No decision provider is configured in the gateway")

        if provider_name not in self.providers:
            avail = list(self.providers.keys())
            raise ProviderUnavailable(
                f"Provider '{provider_name}' is not registered. Available: {avail}"
            )

        return self.providers[provider_name]

    def _validate_request(self, request: DecisionRequest | dict[str, Any]) -> DecisionRequest:
        """Validate input into a canonical DecisionRequest."""
        if isinstance(request, dict):
            try:
                return DecisionRequest.model_validate(request)
            except ValidationError as err:
                raise InvalidRequest(f"Invalid decision request: {err}") from err
        elif isinstance(request, DecisionRequest):
            return request
        else:
            raise InvalidRequest(
                f"Expected DecisionRequest or dict, received: {type(request).__name__}"
            )

    def _normalize_response(
        self,
        raw_response: Any,
        elapsed_ms: float,
        request_id: str | None = None,
    ) -> DecisionResponse:
        """Normalize raw provider output into a canonical DecisionResponse."""
        if isinstance(raw_response, DecisionResponse):
            update_data: dict[str, Any] = {"latency_ms": round(elapsed_ms, 2)}
            if request_id and not raw_response.request_id:
                update_data["request_id"] = request_id
            return raw_response.model_copy(update=update_data)
        elif isinstance(raw_response, dict):
            try:
                raw_response["latency_ms"] = round(elapsed_ms, 2)
                if request_id and not raw_response.get("request_id"):
                    raw_response["request_id"] = request_id
                return DecisionResponse.model_validate(raw_response)
            except ValidationError as err:
                raise NormalizationError(
                    f"Failed to normalize provider dictionary response: {err}"
                ) from err
        else:
            raise NormalizationError(
                f"Provider returned invalid response type: {type(raw_response).__name__}"
            )

    def decide(
        self,
        request: DecisionRequest | dict[str, Any],
        provider_name: str | None = None,
    ) -> DecisionResponse:
        """Validate request, invoke provider, track latency, and return normalized response."""
        valid_request = self._validate_request(request)
        provider = self.get_provider(provider_name)

        start_time = time.perf_counter()
        try:
            raw_response = provider.decide(valid_request)
        except JevlayaError:
            raise
        except Exception as exc:
            raise ProviderResponseError(
                f"Provider '{provider.name}' encountered an unexpected error: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return self._normalize_response(raw_response, elapsed_ms, request_id=valid_request.id)

    async def adecide(
        self,
        request: DecisionRequest | dict[str, Any],
        provider_name: str | None = None,
    ) -> DecisionResponse:
        """Asynchronously validate request, invoke provider, and return normalized response."""
        valid_request = self._validate_request(request)
        provider = self.get_provider(provider_name)

        start_time = time.perf_counter()
        try:
            if hasattr(provider, "adecide"):
                raw_response = await provider.adecide(valid_request)
            else:
                raw_response = await asyncio.to_thread(provider.decide, valid_request)
        except JevlayaError:
            raise
        except Exception as exc:
            raise ProviderResponseError(
                f"Provider '{provider.name}' encountered an unexpected error: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return self._normalize_response(raw_response, elapsed_ms, request_id=valid_request.id)
