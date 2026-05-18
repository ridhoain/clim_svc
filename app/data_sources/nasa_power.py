"""
NASA POWER API client for baseline climate data (2000-2020).
Endpoint: https://power.larc.nasa.gov/api/temporal/monthly/point
No authentication required.
"""
from collections import defaultdict
from typing import Literal

import httpx
import numpy as np

BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
COMMUNITY = "RE"

Parameter = Literal["T2M", "PRECTOTCORR", "GWETROOT"]


def _fetch_raw(
    lat: float,
    lon: float,
    parameter: Parameter,
    start_year: int,
    end_year: int,
) -> dict[str, float]:
    """Return monthly values keyed as YYYYMM strings."""
    params = {
        "parameters": parameter,
        "community": COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": start_year,
        "end": end_year,
        "format": "JSON",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()
    return resp.json()["properties"]["parameter"][parameter]


def annual_mean(
    lat: float,
    lon: float,
    parameter: Parameter,
    start_year: int,
    end_year: int,
) -> float:
    """20-year annual mean for the given parameter."""
    monthly = _fetch_raw(lat, lon, parameter, start_year, end_year)
    # Group by year (key format: YYYYMM), skip fill values (-999)
    by_year: dict[int, list[float]] = defaultdict(list)
    for yyyymm, val in monthly.items():
        if val != -999.0:
            by_year[int(yyyymm[:4])].append(val)
    yearly_means = [np.mean(vals) for vals in by_year.values() if vals]
    return float(np.mean(yearly_means))


def percentile_monthly(
    lat: float,
    lon: float,
    parameter: Parameter,
    start_year: int,
    end_year: int,
    pct: float = 99.0,
) -> float:
    """Return the given percentile across all monthly values in the period."""
    monthly = _fetch_raw(lat, lon, parameter, start_year, end_year)
    values = [v for v in monthly.values() if v != -999.0]
    return float(np.percentile(values, pct))


def dry_season_mean(
    lat: float,
    lon: float,
    parameter: Parameter,
    start_year: int,
    end_year: int,
    dry_months: tuple[int, ...] = (6, 7, 8, 9),
) -> float:
    """Mean over specified months (1-indexed) across all years in the period."""
    monthly = _fetch_raw(lat, lon, parameter, start_year, end_year)
    values = [
        val
        for yyyymm, val in monthly.items()
        if int(yyyymm[4:6]) in dry_months and val != -999.0
    ]
    return float(np.mean(values))
