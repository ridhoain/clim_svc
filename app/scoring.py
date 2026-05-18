"""Convert climate deltas to 0-100 risk scores.

Thresholds are calibrated for the 2000-2020 → 2020-2040 window where
expected global-mean warming is ~0.6-1.0 °C above the recent baseline
(SSP2-4.5). Scores use a stepped function rather than linear so that
small but meaningful deltas produce non-trivial risk values.
"""
from dataclasses import dataclass


@dataclass
class RiskResult:
    score: int          # 0-100
    level: str          # low / moderate / high / critical


def _stepped(value: float, breakpoints: list[tuple[float, int, str]]) -> RiskResult:
    """Apply threshold breakpoints: list of (upper_bound, score, level)."""
    for upper, score, level in breakpoints:
        if value < upper:
            return RiskResult(score=score, level=level)
    last = breakpoints[-1]
    return RiskResult(score=last[1], level=last[2])


def score_temperature(delta_c: float) -> RiskResult:
    """Delta in °C (positive = warming)."""
    return _stepped(delta_c, [
        (0.5,  15, "low"),
        (1.0,  35, "moderate"),
        (1.5,  55, "moderate"),
        (2.0,  75, "high"),
        (float("inf"), 95, "critical"),
    ])


def score_precipitation(delta_pct: float) -> RiskResult:
    """
    Delta as signed percentage change in extreme (99th-pct) precipitation.
    Positive = wetter extremes (flood risk).
    Negative = drying (drought risk).
    Score reflects magnitude regardless of sign.
    """
    magnitude = abs(delta_pct)
    return _stepped(magnitude, [
        (2.0,  10, "low"),
        (4.0,  30, "moderate"),
        (7.0,  55, "moderate"),
        (11.0, 75, "high"),
        (float("inf"), 95, "critical"),
    ])


def score_drought(delta_wetness: float) -> RiskResult:
    """
    Delta in NASA POWER GWETROOT (root-zone wetness, 0-1 scale).
    Negative = drying → drought risk.
    Positive = wetting → low risk here (flood risk captured separately).
    """
    drying = -delta_wetness  # positive = more drying
    return _stepped(drying, [
        (0.01, 10, "low"),
        (0.02, 30, "moderate"),
        (0.04, 55, "moderate"),
        (0.06, 75, "high"),
        (float("inf"), 95, "critical"),
    ])


def score_sea_level(delta_cm: float) -> RiskResult:
    """Delta in cm of relative sea level rise from 2020 to 2040."""
    return _stepped(delta_cm, [
        (5.0,  15, "low"),
        (8.0,  35, "moderate"),
        (12.0, 60, "moderate"),
        (16.0, 80, "high"),
        (float("inf"), 95, "critical"),
    ])
