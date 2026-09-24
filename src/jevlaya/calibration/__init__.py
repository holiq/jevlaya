"""Jevlaya calibration module: post-processing probability and confidence calibration."""

from jevlaya.calibration.base import BaseCalibrator, Calibrator
from jevlaya.calibration.fitter import (
    CalibrationReport,
    fit_from_provider_and_dataset,
    fit_temperature_scaling,
)
from jevlaya.calibration.platt import ConfidenceScalingCalibrator, PlattCalibrator
from jevlaya.calibration.temperature import TemperatureScalingCalibrator

__all__ = [
    "Calibrator",
    "BaseCalibrator",
    "TemperatureScalingCalibrator",
    "PlattCalibrator",
    "ConfidenceScalingCalibrator",
    "CalibrationReport",
    "fit_temperature_scaling",
    "fit_from_provider_and_dataset",
]
