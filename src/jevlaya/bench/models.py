"""Data models for DecisionBench datasets, metrics, and evaluation reports."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jevlaya.protocol.models import DecisionRequest


class BenchmarkExample(BaseModel):
    """A single evaluation sample pairing a canonical request with ground truth."""

    model_config = ConfigDict(extra="ignore")

    id: str
    request: DecisionRequest
    ground_truth: dict[str, Any]
    soft_ground_truth: dict[str, dict[str, float]] | None = None


class BenchmarkDataset(BaseModel):
    """A versioned collection of evaluation examples."""

    model_config = ConfigDict(extra="ignore")

    name: str
    version: str
    description: str = ""
    examples: list[BenchmarkExample] = Field(default_factory=list)

    def to_json_file(self, file_path: str) -> None:
        """Export dataset to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, file_path: str) -> BenchmarkDataset:
        """Load dataset from a JSON file."""
        with open(file_path, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


class MetricResult(BaseModel):
    """Computed evaluation and calibration metrics."""

    model_config = ConfigDict(frozen=True)

    accuracy: float
    soft_accuracy: float
    brier_score: float
    ece: float
    score_mae: float | None = None
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_decisions_per_sec: float
    total_cost_usd: float | None = None
    sample_count: int


class BenchmarkReport(BaseModel):
    """Comprehensive benchmark report for a single provider."""

    model_config = ConfigDict(extra="ignore")

    dataset_name: str
    dataset_version: str
    provider: str
    model: str
    metrics: MetricResult
    per_primitive_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self, indent: int = 2) -> str:
        """Serialize benchmark report to JSON."""
        return self.model_dump_json(indent=indent)

    def to_markdown(self) -> str:
        """Format report as a clean Markdown summary table."""
        m = self.metrics
        cost_str = f"${m.total_cost_usd:.6f}" if m.total_cost_usd is not None else "N/A"
        mae_str = f"{m.score_mae:.4f}" if m.score_mae is not None else "N/A"

        lines = [
            f"# DecisionBench Report: {self.provider} ({self.model})",
            "",
            f"- **Dataset:** `{self.dataset_name}` (v{self.dataset_version})",
            f"- **Samples Evaluated:** {m.sample_count}",
            f"- **Evaluated At:** {self.timestamp}",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Accuracy** | {m.accuracy:.4f} |",
            f"| **Soft Accuracy** | {m.soft_accuracy:.4f} |",
            f"| **Brier Score** | {m.brier_score:.4f} |",
            f"| **ECE (Calibration)** | {m.ece:.4f} |",
            f"| **Score MAE** | {mae_str} |",
            f"| **Latency P50** | {m.latency_p50_ms:.2f} ms |",
            f"| **Latency P95** | {m.latency_p95_ms:.2f} ms |",
            f"| **Throughput** | {m.throughput_decisions_per_sec:.2f} decisions/sec |",
            f"| **Total Cost** | {cost_str} |",
        ]

        if self.per_primitive_metrics:
            lines.extend(
                [
                    "",
                    "## Per-Primitive Metrics",
                    "",
                    "| Primitive | Accuracy | Brier Score | ECE |",
                    "| :--- | :--- | :--- | :--- |",
                ]
            )
            for prim, p_met in self.per_primitive_metrics.items():
                acc = p_met.get("accuracy", 0.0)
                brier = p_met.get("brier", 0.0)
                ece = p_met.get("ece", 0.0)
                lines.append(f"| `{prim}` | {acc:.4f} | {brier:.4f} | {ece:.4f} |")

        return "\n".join(lines)


class BenchmarkComparisonReport(BaseModel):
    """Comparative report across multiple decision providers."""

    model_config = ConfigDict(extra="ignore")

    dataset_name: str
    dataset_version: str
    reports: list[BenchmarkReport]
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self, indent: int = 2) -> str:
        """Serialize comparison report to JSON."""
        return self.model_dump_json(indent=indent)

    def to_markdown(self) -> str:
        """Render side-by-side comparison table in Markdown."""
        header = (
            "| Provider / Model | Accuracy | Soft Acc | Brier | ECE | "
            "P50 (ms) | P95 (ms) | Throughput (q/s) | Cost ($) |"
        )
        separator = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        lines = [
            f"# DecisionBench Comparison: {self.dataset_name} (v{self.dataset_version})",
            "",
            f"- **Evaluated At:** {self.timestamp}",
            f"- **Providers Compared:** {len(self.reports)}",
            "",
            header,
            separator,
        ]

        for r in self.reports:
            m = r.metrics
            cost_str = f"${m.total_cost_usd:.6f}" if m.total_cost_usd is not None else "$0"
            row = (
                f"| **{r.provider}** ({r.model}) | {m.accuracy:.3f} | {m.soft_accuracy:.3f} | "
                f"{m.brier_score:.3f} | {m.ece:.3f} | {m.latency_p50_ms:.1f} | "
                f"{m.latency_p95_ms:.1f} | {m.throughput_decisions_per_sec:.1f} | {cost_str} |"
            )
            lines.append(row)

        return "\n".join(lines)
