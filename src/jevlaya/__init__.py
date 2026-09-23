"""Jevlaya: Open-source decision-intelligence stack for AI agents."""

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
from jevlaya.errors import (
    InvalidRequest,
    JevlayaError,
    NormalizationError,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderTimeout,
    ProviderUnavailable,
    UnsupportedPrimitive,
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
    DecisionProvider,
    JevAdapter,
    JevCapabilities,
    LayaAdapter,
    LayaCapabilities,
    MockProvider,
)

try:
    from jevlaya.server import create_app
except ImportError:
    create_app = None  # type: ignore[assignment]

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
    # Errors
    "JevlayaError",
    "InvalidRequest",
    "UnsupportedPrimitive",
    "ProviderUnavailable",
    "ProviderAuthenticationError",
    "ProviderTimeout",
    "ProviderResponseError",
    "NormalizationError",
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
