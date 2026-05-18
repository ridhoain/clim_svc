"""
Sea level rise assessment for 2020-2040.

Baseline: observed global mean SLR rate over 2000-2020 (~3.6 mm/yr,
slightly faster than the 1993-2020 average of 3.3 mm/yr due to
acceleration). Source: NASA/CNES TOPEX/Poseidon, Jason series.

Future delta: CMIP6/IPCC AR6 projected additional SLR from 2020 to 2040
(global mean, in cm). Local subsidence is NOT included here — add it
separately if known (e.g. Jakarta: ~3.5 mm/yr = +7 cm over 20 years).
"""
from dataclasses import dataclass

from app import config
from app.data_sources import cmip6
from app.scoring import RiskResult, score_sea_level

_OBSERVED_SLR_RATE_MM_YR = 3.6   # global mean, 2000-2020


@dataclass
class SeaLevelAssessment:
    baseline_rate_mm_yr: float    # observed rate during 2000-2020
    projected_delta_cm: float     # additional SLR from 2020 to 2040
    risk: RiskResult


def assess(lat: float, lon: float, scenario: str = config.DEFAULT_SCENARIO) -> SeaLevelAssessment:
    delta_cm = cmip6.get_future_delta(lat, lon, "slr_cm", scenario)

    return SeaLevelAssessment(
        baseline_rate_mm_yr=_OBSERVED_SLR_RATE_MM_YR,
        projected_delta_cm=round(delta_cm, 1),
        risk=score_sea_level(delta_cm),
    )
