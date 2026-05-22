"""
ERA5 (Copernicus CDS) client for baseline climate data (2000-2020).

Downloads monthly ERA5 data for Indonesia via the CDS API, then extracts
point values by nearest-neighbour lookup on the 0.25° grid.

Authentication: set environment variable CDS_API_KEY (or rely on
~/.cdsapirc if running locally).  For GitHub Actions store the key as
the repository secret CDS_API_KEY.

ERA5 variables used:
  2m_temperature              → T2M_annual_mean  (°C, converted from K)
  total_precipitation         → PRECTOTCORR_p99  (mm/day, converted from m/s)
  volumetric_soil_water_layer_1 → GWETROOT_dry_season (0-1 scale)
"""
import os
import tempfile
from pathlib import Path

import numpy as np

# Lazy imports so the module loads even when cdsapi / xarray are absent
# (tests mock these out)

CDS_URL = "https://cds.climate.copernicus.eu/api"

# Bounding box for Indonesia (with a small buffer)
_INDONESIA_AREA = [8, 94, -12, 142]   # North, West, South, East

# ERA5 dataset name on CDS
_DATASET = "reanalysis-era5-single-levels-monthly-means"

# Map logical parameter names to CDS variable names
_CDS_VAR = {
    "T2M":         "2m_temperature",
    "PRECTOTCORR": "total_precipitation",
    "GWETROOT":    "volumetric_soil_water_layer_1",
}

# Product type key used by ERA5 monthly-means
_PRODUCT_TYPE = "monthly_averaged_reanalysis"


def _client():
    import cdsapi
    key = os.environ.get("CDS_API_KEY")
    if key:
        return cdsapi.Client(url=CDS_URL, key=key, quiet=True)
    # Fall back to ~/.cdsapirc
    return cdsapi.Client(quiet=True)


def _years_months(start_year: int, end_year: int) -> tuple[list[str], list[str]]:
    years  = [str(y) for y in range(start_year, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    return years, months


def _download_era5(
    variable: str,
    start_year: int,
    end_year: int,
    tmp_path: str,
) -> None:
    """Download monthly ERA5 data for the Indonesia bounding box."""
    years, months = _years_months(start_year, end_year)
    c = _client()
    c.retrieve(
        _DATASET,
        {
            "product_type": _PRODUCT_TYPE,
            "variable": variable,
            "year": years,
            "month": months,
            "time": "00:00",
            "area": _INDONESIA_AREA,
            "format": "netcdf",
        },
        tmp_path,
    )


def _nearest_point(ds, lat: float, lon: float):
    """Return DataArray of the nearest grid point over time."""
    import xarray as xr  # noqa: F401 (type hint only)
    return ds.sel(latitude=lat, longitude=lon, method="nearest")


def _to_celsius(arr: np.ndarray) -> np.ndarray:
    return arr - 273.15


def _to_mm_per_day(arr: np.ndarray) -> np.ndarray:
    # ERA5 total_precipitation is in m per time step (1 month here).
    # Monthly-mean value is m/s; multiply by seconds-in-day.
    return arr * 86400 * 1000  # m/s → mm/day


def annual_mean(
    lat: float,
    lon: float,
    parameter: str,
    start_year: int,
    end_year: int,
) -> float:
    """Annual mean across start_year–end_year for the given parameter."""
    import xarray as xr

    variable = _CDS_VAR[parameter]
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp = f.name
    try:
        _download_era5(variable, start_year, end_year, tmp)
        ds = xr.open_dataset(tmp)
        var_name = list(ds.data_vars)[0]
        point = _nearest_point(ds[var_name], lat, lon).values.astype(float)

        if parameter == "T2M":
            point = _to_celsius(point)
        elif parameter == "PRECTOTCORR":
            point = _to_mm_per_day(point)

        return float(np.nanmean(point))
    finally:
        Path(tmp).unlink(missing_ok=True)


def percentile_monthly(
    lat: float,
    lon: float,
    parameter: str,
    start_year: int,
    end_year: int,
    pct: float = 99.0,
) -> float:
    """pct-th percentile across all monthly values."""
    import xarray as xr

    variable = _CDS_VAR[parameter]
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp = f.name
    try:
        _download_era5(variable, start_year, end_year, tmp)
        ds = xr.open_dataset(tmp)
        var_name = list(ds.data_vars)[0]
        point = _nearest_point(ds[var_name], lat, lon).values.astype(float)

        if parameter == "T2M":
            point = _to_celsius(point)
        elif parameter == "PRECTOTCORR":
            point = _to_mm_per_day(point)
            point = point[point >= 0]  # drop fill values

        return float(np.nanpercentile(point, pct))
    finally:
        Path(tmp).unlink(missing_ok=True)


def dry_season_mean(
    lat: float,
    lon: float,
    parameter: str,
    start_year: int,
    end_year: int,
    dry_months: tuple[int, ...] = (6, 7, 8, 9),
) -> float:
    """Mean over dry-season months across all years."""
    import xarray as xr

    variable = _CDS_VAR[parameter]
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp = f.name
    try:
        _download_era5(variable, start_year, end_year, tmp)
        ds = xr.open_dataset(tmp)
        var_name = list(ds.data_vars)[0]
        da = _nearest_point(ds[var_name], lat, lon)

        # Filter to dry-season months
        da = da.sel(time=da.time.dt.month.isin(list(dry_months)))
        point = da.values.astype(float)

        if parameter == "T2M":
            point = _to_celsius(point)
        elif parameter == "PRECTOTCORR":
            point = _to_mm_per_day(point)

        return float(np.nanmean(point))
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── Grid-based helpers (used by fetch_baselines.py --source era5) ─────────────
# These accept an already-open xarray Dataset so the CDS download happens once
# for all cities, not once per city.

def _extract_point(ds, lat: float, lon: float, parameter: str) -> np.ndarray:
    var_name = list(ds.data_vars)[0]
    point = _nearest_point(ds[var_name], lat, lon).values.astype(float)
    if parameter == "T2M":
        point = _to_celsius(point)
    elif parameter == "PRECTOTCORR":
        point = _to_mm_per_day(point)
    return point


def annual_mean_from_grid(ds, lat: float, lon: float, parameter: str) -> float:
    """Annual mean from an already-downloaded ERA5 Dataset."""
    return float(np.nanmean(_extract_point(ds, lat, lon, parameter)))


def percentile_from_grid(
    ds, lat: float, lon: float, parameter: str, pct: float = 99.0
) -> float:
    """pct-th percentile from an already-downloaded ERA5 Dataset."""
    point = _extract_point(ds, lat, lon, parameter)
    if parameter == "PRECTOTCORR":
        point = point[point >= 0]
    return float(np.nanpercentile(point, pct))


def dry_season_from_grid(
    ds,
    lat: float,
    lon: float,
    parameter: str,
    dry_months: tuple[int, ...] = (6, 7, 8, 9),
) -> float:
    """Dry-season mean from an already-downloaded ERA5 Dataset."""
    var_name = list(ds.data_vars)[0]
    da = _nearest_point(ds[var_name], lat, lon)
    da = da.sel(time=da.time.dt.month.isin(list(dry_months)))
    point = da.values.astype(float)
    if parameter == "T2M":
        point = _to_celsius(point)
    elif parameter == "PRECTOTCORR":
        point = _to_mm_per_day(point)
    return float(np.nanmean(point))
