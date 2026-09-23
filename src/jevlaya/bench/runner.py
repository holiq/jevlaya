"""Benchmark runner executing standardized evaluation suites across decision providers."""

from __future__ import annotations

import time
from typing import Any

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
    MetricResult,
)
from jevlaya.protocol.models import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
)
from jevlaya.providers.base import DecisionProvider


class DecisionBench:
    """Benchmark runner executing versioned datasets on decision providers."""

    def __init__(self, dataset: BenchmarkDataset) -> None:
        self.dataset = dataset

    def run(self, provider: DecisionProvider, warmup_count: int = 0) -> BenchmarkReport:
        """Run benchmark dataset on provider and compute full performance metrics."""
        # 1. Warm-up
        if warmup_count > 0 and self.dataset.examples:
            warmup_samples = self.dataset.examples[: min(warmup_count, len(self.dataset.examples))]
            for ex in warmup_samples:
                try:
                    provider.decide(ex.request)
                except Exception:
                    pass

        # 2. Benchmark execution
        latencies_ms: list[float] = []
        all_predictions: list[Any] = []
        all_targets: list[Any] = []
        all_prob_on_target: list[float] = []
        all_confidences: list[float] = []
        all_correct_flags: list[bool] = []
        all_distributions: list[Any] = []

        score_pred_indices: list[int] = []
        score_target_indices: list[int] = []

        total_cost_usd: float | None = None
        has_cost = False
        sample_model = "unknown"

        # Primitive-specific collectors
        prim_preds: dict[str, list[Any]] = {"choice": [], "score": [], "noul": []}
        prim_targets: dict[str, list[Any]] = {"choice": [], "score": [], "noul": []}
        prim_confs: dict[str, list[float]] = {"choice": [], "score": [], "noul": []}
        prim_correct: dict[str, list[bool]] = {"choice": [], "score": [], "noul": []}
        prim_dists: dict[str, list[Any]] = {"choice": [], "score": [], "noul": []}

        bench_start = time.perf_counter()

        for example in self.dataset.examples:
            t0 = time.perf_counter()
            response = provider.decide(example.request)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            latencies_ms.append(elapsed_ms)
            sample_model = response.model

            if response.usage.cost_usd is not None:
                has_cost = True
                total_cost_usd = (total_cost_usd or 0.0) + response.usage.cost_usd

            # Evaluate each question
            for q_id, gt in example.ground_truth.items():
                if q_id not in response.answers:
                    continue

                ans = response.answers[q_id]
                q_spec = example.request.questions.get(q_id)

                if isinstance(ans, ChoiceAnswer):
                    target_choice = str(gt)
                    is_correct = ans.choice == target_choice
                    prob_on_target = ans.probabilities.get(target_choice, 0.0)
                    conf = ans.confidence

                    all_predictions.append(ans.choice)
                    all_targets.append(target_choice)
                    all_prob_on_target.append(prob_on_target)
                    all_confidences.append(conf)
                    all_correct_flags.append(is_correct)
                    all_distributions.append(ans.probabilities)

                    prim_preds["choice"].append(ans.choice)
                    prim_targets["choice"].append(target_choice)
                    prim_confs["choice"].append(conf)
                    prim_correct["choice"].append(is_correct)
                    prim_dists["choice"].append(ans.probabilities)

                elif isinstance(ans, ScoreAnswer):
                    target_score = str(gt)
                    is_correct = ans.score == target_score
                    conf = ans.confidence

                    target_idx = 0
                    pred_idx = 0
                    prob_on_target = 0.0

                    if isinstance(q_spec, ScoreQuestion) and target_score in q_spec.criteria:
                        target_idx = q_spec.criteria.index(target_score)
                        if ans.score in q_spec.criteria:
                            pred_idx = q_spec.criteria.index(ans.score)
                        if target_idx < len(ans.probabilities):
                            prob_on_target = ans.probabilities[target_idx]

                        score_pred_indices.append(pred_idx)
                        score_target_indices.append(target_idx)

                    all_predictions.append(ans.score)
                    all_targets.append(target_score)
                    all_prob_on_target.append(prob_on_target)
                    all_confidences.append(conf)
                    all_correct_flags.append(is_correct)
                    all_distributions.append(ans.probabilities)

                    prim_preds["score"].append(ans.score)
                    prim_targets["score"].append(target_score)
                    prim_confs["score"].append(conf)
                    prim_correct["score"].append(is_correct)
                    prim_dists["score"].append(ans.probabilities)

                elif isinstance(ans, NoulAnswer):
                    target_bool = bool(gt)
                    pred_bool = ans.noul >= 0.5
                    is_correct = pred_bool == target_bool
                    prob_on_target = ans.noul if target_bool else (1.0 - ans.noul)
                    conf = max(ans.noul, 1.0 - ans.noul)

                    all_predictions.append(pred_bool)
                    all_targets.append(target_bool)
                    all_prob_on_target.append(prob_on_target)
                    all_confidences.append(conf)
                    all_correct_flags.append(is_correct)
                    all_distributions.append(ans.noul)

                    prim_preds["noul"].append(pred_bool)
                    prim_targets["noul"].append(target_bool)
                    prim_confs["noul"].append(conf)
                    prim_correct["noul"].append(is_correct)
                    prim_dists["noul"].append(ans.noul)

        total_wall_s = max(time.perf_counter() - bench_start, 0.0001)
        total_decisions = len(all_predictions)

        # 3. Metric aggregations
        accuracy = compute_accuracy(all_predictions, all_targets)
        soft_accuracy = compute_soft_accuracy(all_prob_on_target)
        brier = compute_brier_score(all_distributions, all_targets)
        ece = compute_ece(all_confidences, all_correct_flags)
        score_mae = (
            compute_score_mae(score_pred_indices, score_target_indices)
            if score_pred_indices
            else None
        )

        p50 = compute_percentile(latencies_ms, 50.0)
        p95 = compute_percentile(latencies_ms, 95.0)
        throughput = round(total_decisions / total_wall_s, 2)

        # Per-primitive summaries
        per_prim: dict[str, dict[str, float]] = {}
        for p_name in ("choice", "score", "noul"):
            if prim_preds[p_name]:
                per_prim[p_name] = {
                    "accuracy": compute_accuracy(prim_preds[p_name], prim_targets[p_name]),
                    "brier": compute_brier_score(prim_dists[p_name], prim_targets[p_name]),
                    "ece": compute_ece(prim_confs[p_name], prim_correct[p_name]),
                }

        metrics = MetricResult(
            accuracy=accuracy,
            soft_accuracy=soft_accuracy,
            brier_score=brier,
            ece=ece,
            score_mae=score_mae,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            throughput_decisions_per_sec=throughput,
            total_cost_usd=round(total_cost_usd, 6) if has_cost and total_cost_usd else None,
            sample_count=len(self.dataset.examples),
        )

        return BenchmarkReport(
            dataset_name=self.dataset.name,
            dataset_version=self.dataset.version,
            provider=provider.name,
            model=sample_model,
            metrics=metrics,
            per_primitive_metrics=per_prim,
        )

    def compare(
        self,
        providers: list[DecisionProvider],
        warmup_count: int = 0,
    ) -> BenchmarkComparisonReport:
        """Run identical benchmark dataset across multiple providers for comparison."""
        reports: list[BenchmarkReport] = []
        for p in providers:
            report = self.run(p, warmup_count=warmup_count)
            reports.append(report)

        return BenchmarkComparisonReport(
            dataset_name=self.dataset.name,
            dataset_version=self.dataset.version,
            reports=reports,
        )
