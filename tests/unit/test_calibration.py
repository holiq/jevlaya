"""Unit tests for calibration module: Temperature scaling, Platt, confidence scaling, and fitter."""

import math

from jevlaya.calibration import (
    ConfidenceScalingCalibrator,
    PlattCalibrator,
    TemperatureScalingCalibrator,
    fit_temperature_scaling,
)
from jevlaya.protocol.models import (
    ChoiceAnswer,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    UsageInfo,
)


def test_temperature_scaling_categorical() -> None:
    """Test temperature scaling on discrete categorical distribution."""
    # Overconfident distribution
    probs = {"cat": 0.90, "dog": 0.08, "bird": 0.02}

    # T > 1 softens probabilities
    softener = TemperatureScalingCalibrator(temperature=2.0)
    softened = softener.calibrate_distribution(probs)

    assert softened["cat"] < 0.90
    assert softened["dog"] > 0.08
    assert math.isclose(sum(softened.values()), 1.0, abs_tol=0.01)

    # T < 1 sharpens probabilities
    sharpener = TemperatureScalingCalibrator(temperature=0.5)
    sharpened = sharpener.calibrate_distribution(probs)

    assert sharpened["cat"] > 0.90
    assert sharpened["dog"] < 0.08
    assert math.isclose(sum(sharpened.values()), 1.0, abs_tol=0.01)


def test_temperature_scaling_binary() -> None:
    """Test temperature scaling on binary probability (noul)."""
    calibrator = TemperatureScalingCalibrator(temperature=2.0)

    # Overconfident positive prob gets pulled towards 0.5
    cal_prob = calibrator.calibrate_probability(0.9)
    assert 0.5 < cal_prob < 0.9

    # Identity when T=1.0
    identity = TemperatureScalingCalibrator(temperature=1.0)
    assert identity.calibrate_probability(0.85) == 0.85

    # Boundary conditions
    assert calibrator.calibrate_probability(0.0) == 0.0
    assert calibrator.calibrate_probability(1.0) == 1.0


def test_platt_and_confidence_calibrators() -> None:
    """Test Platt and linear confidence calibrators."""
    conf_cal = ConfidenceScalingCalibrator(scale=0.8)
    assert conf_cal.calibrate_probability(0.9) == 0.72

    platt = PlattCalibrator(a=1.5, b=-0.2)
    cal_p = platt.calibrate_probability(0.8)
    assert 0.0 <= cal_p <= 1.0


def test_calibrate_response() -> None:
    """Test applying calibration across a complete DecisionResponse."""
    resp = DecisionResponse(
        provider="mock",
        model="mock-v1",
        answers={
            "q1": ChoiceAnswer(
                choice="a",
                probabilities={"a": 0.80, "b": 0.20},
                confidence=0.80,
            ),
            "q2": ScoreAnswer(
                score="high",
                probabilities=[0.1, 0.2, 0.7],
                confidence=0.70,
            ),
            "q3": NoulAnswer(noul=0.85),
        },
        usage=UsageInfo(),
        latency_ms=12.5,
    )

    calibrator = TemperatureScalingCalibrator(temperature=1.5)
    cal_resp = calibrator.calibrate_response(resp)

    # Choice answer confidence softened
    ans1 = cal_resp.answers["q1"]
    assert isinstance(ans1, ChoiceAnswer)
    assert ans1.confidence < 0.80
    assert math.isclose(sum(ans1.probabilities.values()), 1.0, abs_tol=0.01)

    # Noul answer softened towards 0.5
    ans3 = cal_resp.answers["q3"]
    assert isinstance(ans3, NoulAnswer)
    assert ans3.noul < 0.85


def test_fit_temperature_scaling_reduces_ece() -> None:
    """Test fitting temperature scaling reduces ECE on overconfident synthetic data."""
    # Synthetic overconfident predictions: model predicts 0.95 confidence but accuracy is only 60%
    predictions: list[dict[str, float] | list[float] | float] = []
    targets: list[str] = []

    for i in range(50):
        # Model gives 0.95 to "A"
        predictions.append({"A": 0.95, "B": 0.05})
        # But true label is "A" only 60% of the time
        targets.append("A" if i < 30 else "B")

    report = fit_temperature_scaling(predictions, targets)

    # Temperature should be > 1.0 to soften overconfidence
    assert report.parameters["temperature"] > 1.0
    # ECE should improve
    assert report.post_ece <= report.pre_ece
    # Markdown summary contains expected content
    md = report.summary_markdown()
    assert "Expected Calibration Error" in md
    assert "temperature=" in md
