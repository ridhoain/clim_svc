from dataclasses import dataclass

from app import cache, config
from app.data_sources import cmip6, nasa_power
from app.scoring import RiskResult, score_drought

# Dry season months for the Indonesia / tropical SEA context
_DRY_MONTHS = (6, 7, 8, 9)


@dataclass
class DroughtAssessment:
    baseline_dry_wetness: float   # GWETROOT 0-1 scale, dry-season mean
    future_dry_wetness: float
    delta_wetness: float          # negative = drying
    risk: RiskResult


def assess(lat: float, lon: float, scenario: str = config.DEFAULT_SCENARIO) -> DroughtAssessment:
    cache_key = f"drought_baseline_{lat}_{lon}"
    baseline = cache.get(cache_key)
    if baseline is None:
        baseline = nasa_power.dry_season_mean(
            lat, lon, "GWETROOT",
            config.BASELINE_START, config.BASELINE_END,
            dry_months=_DRY_MONTHS,
        )
        cache.set(cache_key, baseline)

    delta = cmip6.get_future_delta(lat, lon, "soil_wetness", scenario)
    future = baseline + delta

    return DroughtAssessment(
        baseline_dry_wetness=round(baseline, 4),
        future_dry_wetness=round(future, 4),
        delta_wetness=round(delta, 4),
        risk=score_drought(delta),
    )
