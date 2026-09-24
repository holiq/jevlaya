"""Base abstractions and protocols for decision calibration."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jevlaya.protocol.models import (
    ChoiceAnswer,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
)


@runtime_checkable
class Calibrator(Protocol):
    """Protocol for decision calibration post-processors."""

    def calibrate_probability(self, prob: float) -> float:
        """Calibrate a single probability value in range [0.0, 1.0]."""
        ...

    def calibrate_distribution(self, probs: dict[str, float]) -> dict[str, float]:
        """Calibrate a discrete categorical probability distribution."""
        ...

    def calibrate_scores(self, probs: list[float]) -> list[float]:
        """Calibrate an ordered rubric score probability distribution."""
        ...

    def calibrate_response(self, response: DecisionResponse) -> DecisionResponse:
        """Apply calibration across all answers in a DecisionResponse."""
        ...


class BaseCalibrator:
    """Base helper providing default response calibration logic."""

    def calibrate_probability(self, prob: float) -> float:
        raise NotImplementedError

    def calibrate_distribution(self, probs: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError

    def calibrate_scores(self, probs: list[float]) -> list[float]:
        raise NotImplementedError

    def calibrate_response(self, response: DecisionResponse) -> DecisionResponse:
        """Calibrate answers in a canonical DecisionResponse."""
        calibrated_answers: dict[str, Any] = {}

        for q_id, ans in response.answers.items():
            if isinstance(ans, ChoiceAnswer):
                cal_probs = self.calibrate_distribution(ans.probabilities)
                cal_conf = cal_probs.get(ans.choice, ans.confidence)
                # Ensure confidence stays within valid bounds [0.0, 1.0]
                cal_conf = max(0.0, min(1.0, cal_conf))
                calibrated_answers[q_id] = ans.model_copy(
                    update={
                        "probabilities": cal_probs,
                        "confidence": cal_conf,
                    }
                )
            elif isinstance(ans, ScoreAnswer):
                cal_probs_list = self.calibrate_scores(ans.probabilities)
                # Find index of winning score to update confidence
                best_idx = 0
                if cal_probs_list:
                    best_idx = max(range(len(cal_probs_list)), key=lambda i: cal_probs_list[i])
                cal_conf = cal_probs_list[best_idx] if cal_probs_list else ans.confidence
                cal_conf = max(0.0, min(1.0, cal_conf))
                calibrated_answers[q_id] = ans.model_copy(
                    update={
                        "probabilities": cal_probs_list,
                        "confidence": cal_conf,
                    }
                )
            elif isinstance(ans, NoulAnswer):
                cal_noul = self.calibrate_probability(ans.noul)
                cal_noul = max(0.0, min(1.0, cal_noul))
                calibrated_answers[q_id] = ans.model_copy(update={"noul": cal_noul})
            else:
                calibrated_answers[q_id] = ans

        return response.model_copy(update={"answers": calibrated_answers})
