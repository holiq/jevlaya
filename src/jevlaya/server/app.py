"""FastAPI application factory for self-hosted Jevlaya Decision Gateway server."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from jevlaya._version import __version__
from jevlaya.errors import (
    InvalidRequest,
    JevlayaError,
    NormalizationError,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderTimeout,
    ProviderUnavailable,
)
from jevlaya.gateway.gateway import DecisionGateway
from jevlaya.protocol.models import DecisionRequest, DecisionResponse
from jevlaya.providers.mock import MockProvider


def create_app(gateway: DecisionGateway | None = None) -> FastAPI:
    """Create and configure a FastAPI application instance for the decision gateway."""
    gw = gateway or DecisionGateway(provider=MockProvider())

    app = FastAPI(
        title="Jevlaya Decision Gateway",
        description="Self-hosted open-source decision-intelligence API for AI agents",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Exception Handlers ---

    @app.exception_handler(InvalidRequest)
    async def invalid_request_handler(_: Request, exc: InvalidRequest) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "InvalidRequest", "message": exc.message},
        )

    @app.exception_handler(ProviderAuthenticationError)
    async def auth_error_handler(_: Request, exc: ProviderAuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"error": "ProviderAuthenticationError", "message": exc.message},
        )

    @app.exception_handler(ProviderUnavailable)
    async def unavailable_handler(_: Request, exc: ProviderUnavailable) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": "ProviderUnavailable", "message": exc.message},
        )

    @app.exception_handler(ProviderTimeout)
    async def timeout_handler(_: Request, exc: ProviderTimeout) -> JSONResponse:
        return JSONResponse(
            status_code=504,
            content={"error": "ProviderTimeout", "message": exc.message},
        )

    @app.exception_handler(ProviderResponseError)
    async def response_error_handler(_: Request, exc: ProviderResponseError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"error": "ProviderResponseError", "message": exc.message},
        )

    @app.exception_handler(NormalizationError)
    async def normalization_handler(_: Request, exc: NormalizationError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "NormalizationError", "message": exc.message},
        )

    @app.exception_handler(JevlayaError)
    async def general_jev_handler(_: Request, exc: JevlayaError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "JevlayaError", "message": exc.message},
        )

    # --- Endpoints ---

    @app.get("/health", summary="Health check")
    async def health_check() -> dict[str, str]:
        """Verify server health and active Jevlaya version."""
        return {"status": "ok", "version": __version__}

    @app.get("/v1/providers", summary="List providers")
    async def list_providers() -> dict[str, Any]:
        """List all registered decision providers and default provider."""
        return {
            "default_provider": gw.default_provider,
            "available_providers": list(gw.providers.keys()),
        }

    @app.post(
        "/v1/decisions",
        response_model=DecisionResponse,
        summary="Submit decision request",
        description=(
            "Process structured questions across state using the specified decision engine."
        ),
    )
    async def submit_decision(
        request: DecisionRequest,
        provider: str | None = Query(
            default=None,
            description="Optional provider identifier (e.g. 'laya', 'jev', 'mock')",
        ),
    ) -> DecisionResponse:
        """Process canonical decision request and return normalized response with probabilities."""
        return await gw.adecide(request, provider_name=provider)

    return app
