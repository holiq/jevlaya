"""Temperature scaling calibrator for probability distributions and binary predictions."""

from __future__ import annotations

import math

from jevlaya.calibration.base import BaseCalibrator


class TemperatureScalingCalibrator(BaseCalibrator):
    """Post-processing calibrator that scales probabilities by temperature T.

    - T > 1.0: softens overconfident predictions (moves probabilities towards uniform).
    - T = 1.0: identity transformation.
    - T < 1.0: sharpens underconfident predictions (moves probabilities towards extreme).
    """

    def __init__(self, temperature: float = 1.0, epsilon: float = 1e-12) -> None:
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.epsilon = epsilon

    def calibrate_probability(self, prob: float) -> float:
        """Calibrate a binary probability P(yes) using temperature scaling."""
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Probability must be between 0.0 and 1.0, got {prob}")

        if prob <= self.epsilon:
            return 0.0
        if prob >= 1.0 - self.epsilon:
            return 1.0

        if math.isclose(self.temperature, 1.0, rel_tol=1e-6):
            return round(prob, 4)

        # 2-class softmax: p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
        inv_t = 1.0 / self.temperature
        p_t = math.pow(prob, inv_t)
        q_t = math.pow(1.0 - prob, inv_t)
        calibrated = p_t / (p_t + q_t)
        return round(max(0.0, min(1.0, calibrated)), 4)

    def calibrate_distribution(self, probs: dict[str, float]) -> dict[str, float]:
        """Calibrate a categorical probability distribution using temperature scaling."""
        if not probs:
            return {}

        if math.isclose(self.temperature, 1.0, rel_tol=1e-6):
            return dict(probs)

        inv_t = 1.0 / self.temperature
        scaled_terms: dict[str, float] = {}

        for key, p in probs.items():
            clamped = max(p, self.epsilon)
            scaled_terms[key] = math.pow(clamped, inv_t)

        total_sum = sum(scaled_terms.values())
        if total_sum <= 0.0:
            # Fallback to uniform distribution
            uniform = round(1.0 / len(probs), 4)
            return {k: uniform for k in probs}

        # Normalize so probabilities sum to 1.0
        normalized = {k: v / total_sum for k, v in scaled_terms.items()}

        # Round and adjust largest item to guarantee sum is exactly 1.0
        rounded = {k: round(v, 4) for k, v in normalized.items()}
        diff = round(1.0 - sum(rounded.values()), 4)
        if diff != 0.0:
            max_key = max(rounded, key=lambda k: rounded[k])
            rounded[max_key] = round(rounded[max_key] + diff, 4)

        return rounded

    def calibrate_scores(self, probs: list[float]) -> list[float]:
        """Calibrate an ordered rubric score probability distribution."""
        if not probs:
            return []

        if math.isclose(self.temperature, 1.0, rel_tol=1e-6):
            return list(probs)

        inv_t = 1.0 / self.temperature
        scaled_terms = [math.pow(max(p, self.epsilon), inv_t) for p in probs]
        total_sum = sum(scaled_terms)

        if total_sum <= 0.0:
            uniform = round(1.0 / len(probs), 4)
            return [uniform for _ in probs]

        normalized = [v / total_sum for v in scaled_terms]
        rounded = [round(v, 4) for v in normalized]
        diff = round(1.0 - sum(rounded), 4)
        if diff != 0.0:
            max_idx = max(range(len(rounded)), key=lambda i: rounded[i])
            rounded[max_idx] = round(rounded[max_idx] + diff, 4)

        return rounded
