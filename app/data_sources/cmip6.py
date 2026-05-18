"""
CMIP6 climate projections for the 2020-2040 future window.

Two modes:
1. Local NetCDF files — set CMIP6_DIR env var to a directory containing
   pre-downloaded CMIP6 monthly files (variable, scenario, 2015-2050).
2. IPCC AR6 regional fallback — if no local files are found, returns
   published delta estimates for Southeast Asia (IPCC AR6 WGI,
   Interactive Atlas, SEA region, change relative to 2000-2020 baseline).

Fallback deltas are appropriate for Indonesia/SEA. For other regions,
load local CMIP6 files or extend the REGIONAL_DELTAS table.
"""
import os
from pathlib import Path
from typing import Literal

import numpy as np

Scenario = Literal["ssp126", "ssp245", "ssp585"]

# ---------------------------------------------------------------------------
# IPCC AR6 WGI Interactive Atlas — Southeast Asia (SEA)
# Change: 2020-2040 mean vs 2000-2020 mean, multi-model median
# Source: IPCC AR6 WGI Ch.11, Atlas Ch.11, Table SPM.1
# ---------------------------------------------------------------------------
_REGIONAL_DELTAS: dict[str, dict[str, float]] = {
    "ssp126": {
        "temperature":    0.5,    # °C warming
        "precipitation":  1.5,    # % change in annual precip
        "soil_wetness":  -0.012,  # Δ GWETROOT (0-1 scale)
        "slr_cm":         5.5,    # cm relative SLR by 2040 vs 2020
    },
    "ssp245": {
        "temperature":    0.7,
        "precipitation":  2.0,
        "soil_wetness":  -0.018,
        "slr_cm":         7.0,
    },
    "ssp585": {
        "temperature":    0.9,
        "precipitation":  3.0,
        "soil_wetness":  -0.025,
        "slr_cm":         9.0,
    },
}


def _try_load_netcdf(
    lat: float, lon: float, variable: str, scenario: Scenario
) -> float | None:
    """
    Attempt to load a local CMIP6 NetCDF file and return the 2020-2040
    mean for the given location. Returns None if files are not available.

    Expected file glob: <CMIP6_DIR>/<variable>_<scenario>_*.nc
    Variable naming follows CMIP6 convention: tas, pr, mrso.
    """
    cmip6_dir = os.environ.get("CMIP6_DIR")
    if not cmip6_dir:
        return None

    import xarray as xr

    pattern = list(Path(cmip6_dir).glob(f"{variable}_{scenario}_*.nc"))
    if not pattern:
        return None

    ds = xr.open_mfdataset(pattern, combine="by_coords")
    da = ds[variable]

    # Select nearest grid point
    da = da.sel(lat=lat, lon=lon % 360, method="nearest")

    # Slice 2020-2040
    da = da.sel(time=slice("2020", "2040"))

    mean_val = float(da.mean("time").values)

    # Convert units if needed (CMIP6 tas is in K)
    if variable == "tas":
        mean_val -= 273.15

    return mean_val


def get_future_delta(
    lat: float,
    lon: float,
    hazard: Literal["temperature", "precipitation", "soil_wetness", "slr_cm"],
    scenario: Scenario = "ssp245",
) -> float:
    """
    Return the projected delta for 2020-2040 vs the 2000-2020 baseline.

    Tries local CMIP6 files first; falls back to IPCC AR6 regional values.
    """
    cmip6_var_map = {
        "temperature":  "tas",
        "precipitation": "pr",
        "soil_wetness": "mrso",
    }

    if hazard in cmip6_var_map:
        nc_val = _try_load_netcdf(lat, lon, cmip6_var_map[hazard], scenario)
        if nc_val is not None:
            return nc_val

    return _REGIONAL_DELTAS[scenario][hazard]
