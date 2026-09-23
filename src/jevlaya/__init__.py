"""Jevlaya: Open-source decision-intelligence stack for AI agents."""

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

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    # Gateway
    "DecisionGateway",
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
