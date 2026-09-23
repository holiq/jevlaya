"""Explicit typed errors for Jevlaya."""

from __future__ import annotations

import re

# Patterns to mask sensitive tokens, bearer headers, api keys
_SECRET_PATTERN = re.compile(
    r"(bearer\s+[a-zA-Z0-9_\-\.]{8,}|key[_-]?[a-zA-Z0-9_\-]{16,}|sk-[a-zA-Z0-9]{20,})",
    re.IGNORECASE,
)


def redact_secrets(message: str) -> str:
    """Redact sensitive API tokens or secrets from error messages."""
    return _SECRET_PATTERN.sub("[REDACTED]", message)


class JevlayaError(Exception):
    """Base class for all Jevlaya exceptions."""

    def __init__(self, message: str) -> None:
        self.raw_message = message
        self.message = redact_secrets(message)
        super().__init__(self.message)


class InvalidRequest(JevlayaError):
    """Raised when the request format, criteria, or question definitions are invalid."""


class UnsupportedPrimitive(JevlayaError):
    """Raised when an unrecognized or unsupported decision primitive is requested."""


class ProviderUnavailable(JevlayaError):
    """Raised when a decision provider service or backend is unreachable."""


class ProviderAuthenticationError(JevlayaError):
    """Raised when provider authentication fails (e.g. invalid API key)."""


class ProviderTimeout(JevlayaError):
    """Raised when a provider request times out."""


class ProviderResponseError(JevlayaError):
    """Raised when a provider returns an unparseable or unexpected status/response."""


class NormalizationError(JevlayaError):
    """Raised when a provider response cannot be normalized into canonical schema."""
