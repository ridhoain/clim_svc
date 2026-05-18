from dataclasses import dataclass

from app import cache, config
from app.data_sources import cmip6, nasa_power
from app.scoring import RiskResult, score_precipitation


@dataclass
class PrecipitationAssessment:
    baseline_p99_mm_day: float   # 99th-pct monthly precip in baseline
    future_p99_mm_day: float
    delta_pct: float             # % change in extreme precip
    risk: RiskResult


def assess(lat: float, lon: float, scenario: str = config.DEFAULT_SCENARIO) -> PrecipitationAssessment:
    cache_key = f"precip_baseline_p99_{lat}_{lon}"
    baseline_p99 = cache.get(cache_key)
    if baseline_p99 is None:
        baseline_p99 = nasa_power.percentile_monthly(
            lat, lon, "PRECTOTCORR", config.BASELINE_START, config.BASELINE_END, pct=99.0
        )
        cache.set(cache_key, baseline_p99)

    delta_pct = cmip6.get_future_delta(lat, lon, "precipitation", scenario)
    future_p99 = baseline_p99 * (1 + delta_pct / 100)

    return PrecipitationAssessment(
        baseline_p99_mm_day=round(baseline_p99, 2),
        future_p99_mm_day=round(future_p99, 2),
        delta_pct=round(delta_pct, 2),
        risk=score_precipitation(delta_pct),
    )
