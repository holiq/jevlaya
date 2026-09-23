"""Unit tests for Jevlaya typed error hierarchy and secret redaction."""

from jevlaya.errors import (
    InvalidRequest,
    JevlayaError,
    NormalizationError,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderTimeout,
    ProviderUnavailable,
    UnsupportedPrimitive,
    redact_secrets,
)


def test_error_hierarchy() -> None:
    """Verify that all error classes inherit from JevlayaError."""
    assert issubclass(InvalidRequest, JevlayaError)
    assert issubclass(UnsupportedPrimitive, JevlayaError)
    assert issubclass(ProviderUnavailable, JevlayaError)
    assert issubclass(ProviderAuthenticationError, JevlayaError)
    assert issubclass(ProviderTimeout, JevlayaError)
    assert issubclass(ProviderResponseError, JevlayaError)
    assert issubclass(NormalizationError, JevlayaError)


def test_secret_redaction_in_messages() -> None:
    """Ensure sensitive tokens and bearer keys are masked in exception strings."""
    raw_msg = "Auth failed: Bearer secret-token-12345678 and sk-1234567890abcdef123456"
    err = ProviderAuthenticationError(raw_msg)

    assert "Bearer" not in str(err)
    assert "sk-" not in str(err)
    assert "[REDACTED]" in str(err)
    assert err.raw_message == raw_msg


def test_redact_secrets_utility() -> None:
    """Test the standalone secret redactor function."""
    assert redact_secrets("normal error message") == "normal error message"
    assert redact_secrets("Using key-abcdef1234567890") == "Using [REDACTED]"
