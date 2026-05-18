"""
Unit tests for delta computation and risk scoring.
NASA POWER calls are mocked so tests run offline.
"""
import pytest
from unittest.mock import patch

from app.scoring import (
    score_temperature,
    score_precipitation,
    score_drought,
    score_sea_level,
)
from app.data_sources.cmip6 import get_future_delta
from app.hazards import temperature, precipitation, drought, sea_level


# ---------------------------------------------------------------------------
# Scoring unit tests
# ---------------------------------------------------------------------------

class TestTemperatureScoring:
    def test_low(self):
        assert score_temperature(0.3).level == "low"
        assert score_temperature(0.3).score == 15

    def test_moderate(self):
        assert score_temperature(0.7).level == "moderate"

    def test_high(self):
        r = score_temperature(1.8)
        assert r.level == "high"
        assert r.score == 75

    def test_critical(self):
        r = score_temperature(2.5)
        assert r.level == "critical"
        assert r.score == 95

    def test_boundary_at_one(self):
        # exactly 1.0 falls into the 1.0-1.5 band
        assert score_temperature(1.0).score == 55


class TestPrecipitationScoring:
    def test_low_change(self):
        assert score_precipitation(1.0).level == "low"

    def test_large_increase(self):
        r = score_precipitation(12.0)
        assert r.level == "critical"

    def test_drying_uses_magnitude(self):
        # -8% drying → magnitude 8, falls in 7-11 band
        assert score_precipitation(-8.0).score == 75


class TestDroughtScoring:
    def test_wetting_is_low(self):
        assert score_drought(0.05).level == "low"

    def test_moderate_drying(self):
        r = score_drought(-0.025)
        assert r.level == "moderate"

    def test_critical_drying(self):
        r = score_drought(-0.08)
        assert r.level == "critical"


class TestSeaLevelScoring:
    def test_small_slr(self):
        assert score_sea_level(3.0).level == "low"

    def test_high_slr(self):
        assert score_sea_level(14.0).level == "high"

    def test_critical_slr(self):
        assert score_sea_level(20.0).level == "critical"


# ---------------------------------------------------------------------------
# CMIP6 fallback values
# ---------------------------------------------------------------------------

class TestCmip6Fallback:
    def test_ssp245_temperature_delta(self):
        delta = get_future_delta(0, 110, "temperature", "ssp245")
        assert 0.4 <= delta <= 1.2, f"Unexpected delta: {delta}"

    def test_ssp585_warmer_than_ssp126(self):
        d126 = get_future_delta(0, 110, "temperature", "ssp126")
        d585 = get_future_delta(0, 110, "temperature", "ssp585")
        assert d585 > d126

    def test_soil_wetness_negative(self):
        delta = get_future_delta(0, 110, "soil_wetness", "ssp245")
        assert delta < 0, "Expected drying for SEA under SSP2-4.5"

    def test_slr_positive(self):
        delta = get_future_delta(0, 110, "slr_cm", "ssp245")
        assert delta > 0


# ---------------------------------------------------------------------------
# Hazard assess() functions with mocked NASA POWER
# ---------------------------------------------------------------------------

JAKARTA = {"lat": -6.2088, "lon": 106.8456}

class TestTemperatureAssess:
    def test_delta_applied_correctly(self):
        with patch("app.hazards.temperature.nasa_power.annual_mean", return_value=26.4), \
             patch("app.hazards.temperature.cache.get", return_value=None), \
             patch("app.hazards.temperature.cache.set"):
            result = temperature.assess(**JAKARTA, scenario="ssp245")
        assert result.baseline_mean_c == 26.4
        assert result.delta_c == 0.7  # SSP2-4.5 fallback
        assert result.future_mean_c == pytest.approx(27.1, abs=0.05)
        assert result.risk.score > 0


class TestPrecipitationAssess:
    def test_p99_scaled_by_delta(self):
        with patch("app.hazards.precipitation.nasa_power.percentile_monthly", return_value=85.0), \
             patch("app.hazards.precipitation.cache.get", return_value=None), \
             patch("app.hazards.precipitation.cache.set"):
            result = precipitation.assess(**JAKARTA, scenario="ssp245")
        assert result.baseline_p99_mm_day == 85.0
        # 2% increase → 86.7 mm
        assert result.future_p99_mm_day == pytest.approx(85.0 * 1.02, abs=0.1)


class TestDroughtAssess:
    def test_drying_produces_positive_risk(self):
        with patch("app.hazards.drought.nasa_power.dry_season_mean", return_value=0.45), \
             patch("app.hazards.drought.cache.get", return_value=None), \
             patch("app.hazards.drought.cache.set"):
            result = drought.assess(**JAKARTA, scenario="ssp245")
        assert result.delta_wetness < 0
        assert result.risk.score > 0


class TestSeaLevelAssess:
    def test_returns_expected_structure(self):
        result = sea_level.assess(**JAKARTA, scenario="ssp245")
        assert result.baseline_rate_mm_yr == 3.6
        assert result.projected_delta_cm == 7.0
        assert result.risk.score > 0


# ---------------------------------------------------------------------------
# Delta sign / direction invariants
# ---------------------------------------------------------------------------

class TestDeltaInvariants:
    """Higher-emission scenarios must produce larger deltas."""

    def test_temperature_scenario_ordering(self):
        d126 = get_future_delta(0, 110, "temperature", "ssp126")
        d245 = get_future_delta(0, 110, "temperature", "ssp245")
        d585 = get_future_delta(0, 110, "temperature", "ssp585")
        assert d126 < d245 < d585

    def test_slr_scenario_ordering(self):
        s126 = get_future_delta(0, 110, "slr_cm", "ssp126")
        s245 = get_future_delta(0, 110, "slr_cm", "ssp245")
        s585 = get_future_delta(0, 110, "slr_cm", "ssp585")
        assert s126 < s245 < s585
