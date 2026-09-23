"""Pure mathematical and statistical metric calculations for DecisionBench."""

from __future__ import annotations

import math
from typing import Any


def compute_percentile(values: list[float], percentile: float) -> float:
    """Compute the specified percentile (0.0 to 100.0) from a list of numbers."""
    if not values:
        return 0.0

    sorted_vals = sorted(values)
    if percentile <= 0.0:
        return sorted_vals[0]
    if percentile >= 100.0:
        return sorted_vals[-1]

    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    floor_k = math.floor(k)
    ceil_k = math.ceil(k)

    if floor_k == ceil_k:
        return sorted_vals[int(k)]

    d0 = sorted_vals[floor_k] * (ceil_k - k)
    d1 = sorted_vals[ceil_k] * (k - floor_k)
    return round(d0 + d1, 4)


def compute_accuracy(predictions: list[Any], targets: list[Any]) -> float:
    """Compute argmax classification accuracy in range [0.0, 1.0]."""
    if not predictions or len(predictions) != len(targets):
        return 0.0

    correct = sum(1 for p, t in zip(predictions, targets, strict=True) if p == t)
    return round(correct / len(predictions), 4)


def compute_soft_accuracy(prob_masses_on_target: list[float]) -> float:
    """Compute average probability mass assigned directly to the ground truth label."""
    if not prob_masses_on_target:
        return 0.0

    return round(sum(prob_masses_on_target) / len(prob_masses_on_target), 4)


def compute_brier_score(
    predictions: list[dict[str, float] | list[float] | float],
    targets: list[Any],
) -> float:
    """Compute mean squared probability error (Brier score). Lower is better."""
    if not predictions or len(predictions) != len(targets):
        return 0.0

    total_squared_loss = 0.0

    for pred, target in zip(predictions, targets, strict=True):
        if isinstance(pred, (int, float)):
            # Binary probability (noul)
            y = 1.0 if (target is True or target == 1 or str(target).lower() == "true") else 0.0
            total_squared_loss += (float(pred) - y) ** 2

        elif isinstance(pred, dict):
            # Categorical distribution over named labels
            target_str = str(target)
            loss_item = 0.0
            target_found = False
            for label, prob in pred.items():
                y = 1.0 if label == target_str else 0.0
                if label == target_str:
                    target_found = True
                loss_item += (prob - y) ** 2
            if not target_found:
                loss_item += 1.0
            total_squared_loss += loss_item

        elif isinstance(pred, list):
            # Ordered scale distribution
            target_idx = int(target) if isinstance(target, (int, float)) else -1
            loss_item = 0.0
            target_found = False
            for idx, prob in enumerate(pred):
                y = 1.0 if idx == target_idx else 0.0
                if idx == target_idx:
                    target_found = True
                loss_item += (prob - y) ** 2
            if not target_found and target_idx >= 0:
                loss_item += 1.0
            total_squared_loss += loss_item

    return round(total_squared_loss / len(predictions), 4)


def compute_ece(
    confidences: list[float],
    correct_flags: list[bool],
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins in range [0, 1]."""
    if not confidences or len(confidences) != len(correct_flags):
        return 0.0

    total_samples = len(confidences)
    bin_size = 1.0 / num_bins
    ece = 0.0

    for b in range(num_bins):
        bin_lower = b * bin_size
        bin_upper = (b + 1) * bin_size

        # Find items in [bin_lower, bin_upper) - include 1.0 in last bin
        bin_indices = [
            i
            for i, conf in enumerate(confidences)
            if bin_lower <= conf < bin_upper or (b == num_bins - 1 and conf == bin_upper)
        ]

        if not bin_indices:
            continue

        bin_confs = [confidences[i] for i in bin_indices]
        bin_corrects = [correct_flags[i] for i in bin_indices]

        avg_confidence = sum(bin_confs) / len(bin_confs)
        accuracy = sum(1 for c in bin_corrects if c) / len(bin_corrects)

        bin_weight = len(bin_indices) / total_samples
        ece += bin_weight * abs(accuracy - avg_confidence)

    return round(ece, 4)


def compute_score_mae(
    predicted_indices: list[int],
    target_indices: list[int],
) -> float:
    """Compute Mean Absolute Error (MAE) between predicted and true ordinal rank indices."""
    if not predicted_indices or len(predicted_indices) != len(target_indices):
        return 0.0

    total_abs_diff = sum(abs(p - t) for p, t in zip(predicted_indices, target_indices, strict=True))
    return round(total_abs_diff / len(predicted_indices), 4)
