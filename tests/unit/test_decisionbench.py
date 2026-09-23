"""Unit tests for DecisionBench metrics, dataset management, and benchmark execution."""

from jevlaya.bench.metrics import (
    compute_accuracy,
    compute_brier_score,
    compute_ece,
    compute_percentile,
    compute_score_mae,
    compute_soft_accuracy,
)
from jevlaya.bench.models import (
    BenchmarkComparisonReport,
    BenchmarkDataset,
    BenchmarkReport,
)
from jevlaya.bench.runner import DecisionBench
from jevlaya.bench.sample_data import get_sample_dataset
from jevlaya.providers import MockProvider


def test_percentile_calculation() -> None:
    """Verify statistical percentile calculations for P50 and P95."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50 = compute_percentile(data, 50.0)
    assert p50 == 30.0

    empty_p = compute_percentile([], 50.0)
    assert empty_p == 0.0


def test_accuracy_calculation() -> None:
    """Verify categorical classification accuracy."""
    preds = ["a", "b", "c", "d"]
    targets = ["a", "b", "x", "y"]
    assert compute_accuracy(preds, targets) == 0.5


def test_soft_accuracy_calculation() -> None:
    """Verify mean probability assigned to ground truth."""
    masses = [0.9, 0.8, 0.7]
    assert compute_soft_accuracy(masses) == 0.8


def test_brier_score_calculation() -> None:
    """Verify Brier score for binary and categorical predictions."""
    # Perfect binary prediction
    assert compute_brier_score([1.0], [True]) == 0.0
    # Worst binary prediction
    assert compute_brier_score([0.0], [True]) == 1.0

    # Categorical prediction
    cat_preds = [{"a": 0.8, "b": 0.2}]
    # target is 'a': loss is (0.8 - 1)^2 + (0.2 - 0)^2 = 0.04 + 0.04 = 0.08
    assert compute_brier_score(cat_preds, ["a"]) == 0.08


def test_ece_calculation() -> None:
    """Verify Expected Calibration Error (ECE) across confidence bins."""
    # 2 predictions: both confidence 0.8, both correct -> acc=1.0, conf=0.8, ECE = |1.0 - 0.8| = 0.2
    confs = [0.8, 0.8]
    corrects = [True, True]
    assert compute_ece(confs, corrects, num_bins=5) == 0.2


def test_score_mae_calculation() -> None:
    """Verify Mean Absolute Error on ordinal rating scales."""
    preds = [0, 2, 1]
    targets = [0, 1, 2]
    # diffs: |0-0| + |2-1| + |1-2| = 0 + 1 + 1 = 2 -> 2/3 = 0.6667
    assert compute_score_mae(preds, targets) == 0.6667


def test_sample_dataset_integrity() -> None:
    """Verify bundled sample dataset structure and serialization."""
    ds = get_sample_dataset()
    assert ds.name == "jevlaya-standard-triage"
    assert ds.version == "1.0.0"
    assert len(ds.examples) == 3

    # JSON roundtrip
    json_str = ds.model_dump_json()
    loaded = BenchmarkDataset.model_validate_json(json_str)
    assert loaded.name == ds.name
    assert len(loaded.examples) == len(ds.examples)


def test_decisionbench_run_with_mock_provider() -> None:
    """DecisionBench runs on MockProvider, computing full report metrics."""
    ds = get_sample_dataset()
    bench = DecisionBench(ds)
    provider = MockProvider(name="eval-mock")

    report = bench.run(provider, warmup_count=1)

    assert isinstance(report, BenchmarkReport)
    assert report.provider == "eval-mock"
    assert report.dataset_name == "jevlaya-standard-triage"
    assert report.metrics.sample_count == 3
    assert report.metrics.latency_p50_ms >= 0.0
    assert report.metrics.throughput_decisions_per_sec > 0.0

    # Test Markdown rendering
    md = report.to_markdown()
    assert "# DecisionBench Report: eval-mock" in md
    assert "Summary Metrics" in md
    assert "Accuracy" in md

    # Test JSON rendering
    json_rep = report.to_json()
    assert "eval-mock" in json_rep


def test_decisionbench_multi_provider_comparison() -> None:
    """DecisionBench compares multiple providers and renders comparison tables."""
    ds = get_sample_dataset()
    bench = DecisionBench(ds)

    p1 = MockProvider(name="mock-fast")
    p2 = MockProvider(name="mock-slow", simulate_latency_ms=5.0)

    comparison = bench.compare([p1, p2])

    assert isinstance(comparison, BenchmarkComparisonReport)
    assert len(comparison.reports) == 2
    assert comparison.reports[0].provider == "mock-fast"
    assert comparison.reports[1].provider == "mock-slow"

    md = comparison.to_markdown()
    assert "# DecisionBench Comparison" in md
    assert "mock-fast" in md
    assert "mock-slow" in md
