"""Decision Gateway: Provider-agnostic entry point for AI agent decisions."""

from __future__ import annotations

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

    def decide(
        self,
        request: DecisionRequest | dict[str, Any],
        provider_name: str | None = None,
    ) -> DecisionResponse:
        """Validate request, invoke provider, track latency, and return normalized response."""
        # 1. Validation
        if isinstance(request, dict):
            try:
                valid_request = DecisionRequest.model_validate(request)
            except ValidationError as err:
                raise InvalidRequest(f"Invalid decision request: {err}") from err
        elif isinstance(request, DecisionRequest):
            valid_request = request
        else:
            raise InvalidRequest(
                f"Expected DecisionRequest or dict, received: {type(request).__name__}"
            )

        # 2. Resolve provider
        provider = self.get_provider(provider_name)

        # 3. Invocation with precision latency tracking
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

        # 4. Response normalization
        if isinstance(raw_response, DecisionResponse):
            response = raw_response.model_copy(update={"latency_ms": round(elapsed_ms, 2)})
        elif isinstance(raw_response, dict):
            try:
                raw_response["latency_ms"] = round(elapsed_ms, 2)
                response = DecisionResponse.model_validate(raw_response)
            except ValidationError as err:
                raise NormalizationError(
                    f"Failed to normalize provider dictionary response: {err}"
                ) from err
        else:
            raise NormalizationError(
                f"Provider returned invalid response type: {type(raw_response).__name__}"
            )

        return response
