from dataclasses import dataclass

from app import cache, config
from app.data_sources import cmip6, nasa_power
from app.scoring import RiskResult, score_temperature


@dataclass
class TemperatureAssessment:
    baseline_mean_c: float
    future_mean_c: float
    delta_c: float
    risk: RiskResult


def assess(lat: float, lon: float, scenario: str = config.DEFAULT_SCENARIO) -> TemperatureAssessment:
    cache_key = f"temp_baseline_{lat}_{lon}"
    baseline = cache.get(cache_key)
    if baseline is None:
        baseline = nasa_power.annual_mean(
            lat, lon, "T2M", config.BASELINE_START, config.BASELINE_END
        )
        cache.set(cache_key, baseline)

    delta = cmip6.get_future_delta(lat, lon, "temperature", scenario)
    future = baseline + delta

    return TemperatureAssessment(
        baseline_mean_c=round(baseline, 2),
        future_mean_c=round(future, 2),
        delta_c=round(delta, 2),
        risk=score_temperature(delta),
    )
