from typing import Any

from jevlaya._version import __version__
from jevlaya.bench import (
    BenchmarkComparisonReport,
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkReport,
    DecisionBench,
    MetricResult,
    get_sample_dataset,
)
from jevlaya.calibration import (
    BaseCalibrator,
    CalibrationReport,
    Calibrator,
    ConfidenceScalingCalibrator,
    PlattCalibrator,
    TemperatureScalingCalibrator,
    fit_from_provider_and_dataset,
    fit_temperature_scaling,
)
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
from jevlaya.gateway import DecisionGateway
from jevlaya.protocol import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    UsageInfo,
)
from jevlaya.providers import (
    LAYA_CHECKPOINTS,
    DecisionProvider,
    JevAdapter,
    JevCapabilities,
    LayaAdapter,
    LayaCapabilities,
    MockProvider,
)
from jevlaya.routing import PolicyRouter, RoutingMetadata, RoutingPolicy
from jevlaya.telemetry import (
    CallbackTelemetrySink,
    CompositeTelemetrySink,
    DecisionEvent,
    InMemoryTelemetrySink,
    JsonLinesTelemetrySink,
    TelemetrySink,
    TelemetrySummary,
)


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Create a FastAPI application instance for the decision gateway (requires jevlaya[server])."""
    try:
        from jevlaya.server.app import create_app as _create_app

        return _create_app(*args, **kwargs)
    except ImportError as err:
        raise ImportError(
            "FastAPI server dependencies required. Install with: uv sync --extra server"
        ) from err

__all__ = [
    "__version__",
    # Gateway & Server
    "DecisionGateway",
    "create_app",
    # Providers
    "DecisionProvider",
    "MockProvider",
    "LayaAdapter",
    "LayaCapabilities",
    "LAYA_CHECKPOINTS",
    "JevAdapter",
    "JevCapabilities",
    # DecisionBench
    "DecisionBench",
    "BenchmarkDataset",
    "BenchmarkExample",
    "BenchmarkReport",
    "BenchmarkComparisonReport",
    "MetricResult",
    "get_sample_dataset",
    # Routing
    "PolicyRouter",
    "RoutingPolicy",
    "RoutingMetadata",
    # Calibration
    "Calibrator",
    "BaseCalibrator",
    "TemperatureScalingCalibrator",
    "PlattCalibrator",
    "ConfidenceScalingCalibrator",
    "CalibrationReport",
    "fit_temperature_scaling",
    "fit_from_provider_and_dataset",
    # Telemetry
    "DecisionEvent",
    "TelemetrySink",
    "InMemoryTelemetrySink",
    "JsonLinesTelemetrySink",
    "CallbackTelemetrySink",
    "CompositeTelemetrySink",
    "TelemetrySummary",
    # Errors
    "JevlayaError",
    "InvalidRequest",
    "UnsupportedPrimitive",
    "ProviderUnavailable",
    "ProviderAuthenticationError",
    "ProviderTimeout",
    "ProviderResponseError",
    "NormalizationError",
    "redact_secrets",
    # Protocol
    "DecisionRequest",
    "DecisionResponse",
    "Question",
    "ChoiceQuestion",
    "ScoreQuestion",
    "NoulQuestion",
    "Answer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NoulAnswer",
    "UsageInfo",
]
