"""Platt and linear confidence calibrators."""

from __future__ import annotations

import math

from jevlaya.calibration.base import BaseCalibrator


class ConfidenceScalingCalibrator(BaseCalibrator):
    """Linear scaling calibrator matching Jevlaya RoutingPolicy confidence_scaling."""

    def __init__(self, scale: float = 1.0) -> None:
        if scale < 0.0:
            raise ValueError(f"Scale must be non-negative, got {scale}")
        self.scale = float(scale)

    def calibrate_probability(self, prob: float) -> float:
        """Scale probability directly by scale factor, clamped to [0.0, 1.0]."""
        return round(max(0.0, min(1.0, prob * self.scale)), 4)

    def calibrate_distribution(self, probs: dict[str, float]) -> dict[str, float]:
        """Categorical distribution preserves relative ratios while scaling confidence."""
        return dict(probs)

    def calibrate_scores(self, probs: list[float]) -> list[float]:
        """Score distribution preserves relative ratios."""
        return list(probs)


class PlattCalibrator(BaseCalibrator):
    """Platt scaling (logistic calibration) mapping logits through a sigmoid:

    p' = 1 / (1 + exp(-(a * logit(p) + b)))
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, epsilon: float = 1e-12) -> None:
        self.a = float(a)
        self.b = float(b)
        self.epsilon = epsilon

    def calibrate_probability(self, prob: float) -> float:
        """Calibrate a binary probability using logistic scaling."""
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Probability must be between 0.0 and 1.0, got {prob}")

        if prob <= self.epsilon:
            return 0.0
        if prob >= 1.0 - self.epsilon:
            return 1.0

        logit = math.log(prob / (1.0 - prob))
        scaled_logit = self.a * logit + self.b
        # Numerical protection for exp
        scaled_logit = max(-50.0, min(50.0, scaled_logit))
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        return round(max(0.0, min(1.0, calibrated)), 4)

    def calibrate_distribution(self, probs: dict[str, float]) -> dict[str, float]:
        """Calibrate distribution using calibrated pseudo-logits."""
        if not probs:
            return {}

        terms: dict[str, float] = {}
        for k, p in probs.items():
            clamped = max(self.epsilon, min(1.0 - self.epsilon, p))
            logit = math.log(clamped / (1.0 - clamped))
            scaled = max(-50.0, min(50.0, self.a * logit + self.b))
            terms[k] = math.exp(scaled)

        total = sum(terms.values())
        if total <= 0.0:
            uniform = round(1.0 / len(probs), 4)
            return {k: uniform for k in probs}

        normalized = {k: v / total for k, v in terms.items()}
        rounded = {k: round(v, 4) for k, v in normalized.items()}
        diff = round(1.0 - sum(rounded.values()), 4)
        if diff != 0.0:
            max_key = max(rounded, key=lambda k: rounded[k])
            rounded[max_key] = round(rounded[max_key] + diff, 4)
        return rounded

    def calibrate_scores(self, probs: list[float]) -> list[float]:
        """Calibrate scores using calibrated pseudo-logits."""
        if not probs:
            return []

        terms: list[float] = []
        for p in probs:
            clamped = max(self.epsilon, min(1.0 - self.epsilon, p))
            logit = math.log(clamped / (1.0 - clamped))
            scaled = max(-50.0, min(50.0, self.a * logit + self.b))
            terms.append(math.exp(scaled))

        total = sum(terms)
        if total <= 0.0:
            uniform = round(1.0 / len(probs), 4)
            return [uniform for _ in probs]

        normalized = [v / total for v in terms]
        rounded = [round(v, 4) for v in normalized]
        diff = round(1.0 - sum(rounded), 4)
        if diff != 0.0:
            max_idx = max(range(len(rounded)), key=lambda i: rounded[i])
            rounded[max_idx] = round(rounded[max_idx] + diff, 4)
        return rounded
