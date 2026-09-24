"""Automated calibration fitting algorithms and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from jevlaya.bench.metrics import compute_brier_score, compute_ece
from jevlaya.calibration.base import Calibrator
from jevlaya.calibration.temperature import TemperatureScalingCalibrator
from jevlaya.protocol.models import (
    ChoiceAnswer,
    DecisionRequest,
    NoulAnswer,
    ScoreAnswer,
)
from jevlaya.providers.base import DecisionProvider


@dataclass
class CalibrationReport:
    """Evaluation summary of pre- and post-calibration metrics."""

    method: str
    parameters: dict[str, float]
    pre_ece: float
    post_ece: float
    pre_brier: float
    post_brier: float
    sample_count: int
    calibrator: Calibrator

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "parameters": self.parameters,
            "pre_ece": self.pre_ece,
            "post_ece": self.post_ece,
            "ece_improvement": round(self.pre_ece - self.post_ece, 4),
            "pre_brier": self.pre_brier,
            "post_brier": self.post_brier,
            "brier_improvement": round(self.pre_brier - self.post_brier, 4),
            "sample_count": self.sample_count,
        }

    def summary_markdown(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        ece_diff = round(self.pre_ece - self.post_ece, 4)
        brier_diff = round(self.pre_brier - self.post_brier, 4)

        ece_msg = f"improved by {ece_diff}" if ece_diff >= 0 else f"degraded by {-ece_diff}"
        brier_msg = f"improved by {brier_diff}" if brier_diff >= 0 else f"degraded by {-brier_diff}"

        return (
            f"### Calibration Report ({self.method.replace('_', ' ').title()})\n\n"
            f"- **Fitted Parameters:** `{param_str}`\n"
            f"- **Samples Evaluated:** {self.sample_count}\n"
            f"- **Expected Calibration Error (ECE):** {self.pre_ece:.4f} -> {self.post_ece:.4f} "
            f"({ece_msg})\n"
            f"- **Brier Score:** {self.pre_brier:.4f} -> {self.post_brier:.4f} "
            f"({brier_msg})\n"
        )


def _golden_section_search(
    f: Any,
    a: float,
    b: float,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """1D convex optimization via golden-section search (pure Python)."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = b - resphi * (b - a)
            fd = f(d)

    return (a + b) / 2.0


def _nll_loss_temperature(
    temperature: float,
    predictions: list[dict[str, float] | list[float] | float],
    targets: list[Any],
    epsilon: float = 1e-12,
) -> float:
    """Compute Negative Log-Likelihood (NLL) for a given temperature."""
    calibrator = TemperatureScalingCalibrator(temperature=temperature)
    total_nll = 0.0

    for pred, target in zip(predictions, targets, strict=True):
        if isinstance(pred, (int, float)):
            # Binary probability (noul)
            p_cal = calibrator.calibrate_probability(float(pred))
            is_pos = target is True or target == 1 or str(target).lower() == "true"
            prob_target = p_cal if is_pos else (1.0 - p_cal)
            total_nll -= math.log(max(epsilon, min(1.0, prob_target)))

        elif isinstance(pred, dict):
            # Categorical distribution
            cal_dist = calibrator.calibrate_distribution(pred)
            target_str = str(target)
            prob_target = cal_dist.get(target_str, epsilon)
            total_nll -= math.log(max(epsilon, min(1.0, prob_target)))

        elif isinstance(pred, list):
            # Ordered score distribution
            cal_scores = calibrator.calibrate_scores(pred)
            target_idx = int(target) if isinstance(target, (int, float)) else -1
            if 0 <= target_idx < len(cal_scores):
                prob_target = cal_scores[target_idx]
            else:
                prob_target = epsilon
            total_nll -= math.log(max(epsilon, min(1.0, prob_target)))

    return total_nll / max(1, len(predictions))


def fit_temperature_scaling(
    predictions: list[dict[str, float] | list[float] | float],
    targets: list[Any],
    confidences: list[float] | None = None,
    min_temp: float = 0.05,
    max_temp: float = 8.0,
) -> CalibrationReport:
    """Fit optimal temperature parameter on calibration dataset using NLL minimization."""
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Predictions and targets must be non-empty and of equal length")

    # 1. Compute pre-calibration metrics
    pre_brier = compute_brier_score(predictions, targets)

    # Extract confidences and correct flags for ECE
    extracted_confs: list[float] = []
    correct_flags: list[bool] = []

    for i, (pred, target) in enumerate(zip(predictions, targets, strict=True)):
        if isinstance(pred, (int, float)):
            # For binary: confidence is distance from 0.5 or prob
            p = float(pred)
            conf = max(p, 1.0 - p)
            is_pos = target is True or target == 1 or str(target).lower() == "true"
            is_correct = (p >= 0.5 and is_pos) or (p < 0.5 and not is_pos)
            extracted_confs.append(conf)
            correct_flags.append(is_correct)
        elif isinstance(pred, dict):
            target_str = str(target)
            best_opt = max(pred, key=lambda k: pred[k]) if pred else ""
            best_prob = pred.get(best_opt, 0.0)
            extracted_confs.append(confidences[i] if confidences else best_prob)
            correct_flags.append(best_opt == target_str)
        elif isinstance(pred, list):
            target_idx = int(target) if isinstance(target, (int, float)) else -1
            best_idx = max(range(len(pred)), key=lambda idx: pred[idx]) if pred else -1
            best_prob = pred[best_idx] if 0 <= best_idx < len(pred) else 0.0
            extracted_confs.append(confidences[i] if confidences else best_prob)
            correct_flags.append(best_idx == target_idx)

    pre_ece = compute_ece(extracted_confs, correct_flags)

    # 2. Optimize temperature parameter
    def loss_fn(t: float) -> float:
        return _nll_loss_temperature(t, predictions, targets)

    best_temp = _golden_section_search(loss_fn, a=min_temp, b=max_temp, tol=1e-4)
    best_temp = round(max(min_temp, min(max_temp, best_temp)), 3)

    calibrator = TemperatureScalingCalibrator(temperature=best_temp)

    # 3. Compute post-calibration metrics
    post_predictions: list[dict[str, float] | list[float] | float] = []
    post_confs: list[float] = []

    for pred in predictions:
        if isinstance(pred, (int, float)):
            cal_p = calibrator.calibrate_probability(float(pred))
            post_predictions.append(cal_p)
            post_confs.append(max(cal_p, 1.0 - cal_p))
        elif isinstance(pred, dict):
            cal_d = calibrator.calibrate_distribution(pred)
            post_predictions.append(cal_d)
            best_key = max(cal_d, key=lambda k: cal_d[k]) if cal_d else ""
            post_confs.append(cal_d.get(best_key, 0.0))
        elif isinstance(pred, list):
            cal_s = calibrator.calibrate_scores(pred)
            post_predictions.append(cal_s)
            best_i = max(range(len(cal_s)), key=lambda idx: cal_s[idx]) if cal_s else 0
            post_confs.append(cal_s[best_i] if cal_s else 0.0)

    post_brier = compute_brier_score(post_predictions, targets)
    post_ece = compute_ece(post_confs, correct_flags)

    return CalibrationReport(
        method="temperature_scaling",
        parameters={"temperature": best_temp},
        pre_ece=pre_ece,
        post_ece=post_ece,
        pre_brier=pre_brier,
        post_brier=post_brier,
        sample_count=len(predictions),
        calibrator=calibrator,
    )


def fit_from_provider_and_dataset(
    provider: DecisionProvider,
    examples: list[tuple[DecisionRequest, dict[str, Any]]],
    method: str = "temperature",
) -> dict[str, CalibrationReport]:
    """Execute provider on dataset examples, fit calibrator per question ID, and return reports."""
    # Group predictions and targets by question ID
    q_predictions: dict[str, list[Any]] = {}
    q_targets: dict[str, list[Any]] = {}
    q_confidences: dict[str, list[float]] = {}

    for req, ground_truth in examples:
        resp = provider.decide(req)
        for q_id, ans in resp.answers.items():
            if q_id not in ground_truth:
                continue

            target = ground_truth[q_id]
            q_predictions.setdefault(q_id, [])
            q_targets.setdefault(q_id, [])
            q_confidences.setdefault(q_id, [])

            if isinstance(ans, ChoiceAnswer):
                q_predictions[q_id].append(ans.probabilities)
                q_targets[q_id].append(target)
                q_confidences[q_id].append(ans.confidence)
            elif isinstance(ans, ScoreAnswer):
                q_predictions[q_id].append(ans.probabilities)
                q_targets[q_id].append(target)
                q_confidences[q_id].append(ans.confidence)
            elif isinstance(ans, NoulAnswer):
                q_predictions[q_id].append(ans.noul)
                q_targets[q_id].append(target)
                q_confidences[q_id].append(ans.noul)

    reports: dict[str, CalibrationReport] = {}
    for q_id in q_predictions:
        preds = q_predictions[q_id]
        targs = q_targets[q_id]
        confs = q_confidences[q_id]
        if preds and len(preds) == len(targs):
            report = fit_temperature_scaling(preds, targs, confidences=confs)
            reports[q_id] = report

    return reports
