"""
Fetch baseline climate data for Indonesian cities and write to data/baselines.json.

Supports two data sources:
  --source nasa   NASA POWER API (default, no API key required)
  --source era5   ERA5 via Copernicus CDS (requires CDS_API_KEY env var or ~/.cdsapirc)

ERA5 downloads the entire Indonesia bounding box per parameter per run, then
extracts individual city values — much more efficient than per-city point queries.

Usage:
    python scripts/fetch_baselines.py                            # NASA POWER, all cities
    python scripts/fetch_baselines.py --source era5              # ERA5, all cities
    python scripts/fetch_baselines.py --cities Jakarta Bali      # filter by city name
    python scripts/fetch_baselines.py --province "Jawa Barat"
    python scripts/fetch_baselines.py --lat -6.21 --lon 106.85 --name MyCity
    python scripts/fetch_baselines.py --workers 4                # parallel workers (NASA only)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import BASELINE_START, BASELINE_END
from app.data_sources import nasa_power

BASELINES_FILE = Path(__file__).parent.parent / "data" / "baselines.json"
CITIES_FILE    = Path(__file__).parent.parent / "data" / "cities.csv"

# NASA POWER informal rate limit: ~30 requests/min per IP
_REQUEST_DELAY_S = 2.2


def key(lat: float, lon: float) -> str:
    return f"{lat:.2f},{lon:.2f}"


# ── NASA POWER fetching ────────────────────────────────────────────────────────

def _fetch_city_nasa(name: str, lat: float, lon: float) -> tuple[str, dict | None]:
    try:
        time.sleep(_REQUEST_DELAY_S)
        t2m = nasa_power.annual_mean(lat, lon, "T2M", BASELINE_START, BASELINE_END)

        time.sleep(_REQUEST_DELAY_S)
        p99 = nasa_power.percentile_monthly(lat, lon, "PRECTOTCORR", BASELINE_START, BASELINE_END, pct=99.0)

        time.sleep(_REQUEST_DELAY_S)
        gwet = nasa_power.dry_season_mean(lat, lon, "GWETROOT", BASELINE_START, BASELINE_END)

        return key(lat, lon), {
            "city": name,
            "T2M_annual_mean":      round(t2m,  2),
            "PRECTOTCORR_p99":      round(p99,  2),
            "GWETROOT_dry_season":  round(gwet, 4),
        }
    except Exception as exc:
        print(f"  ERROR [{name}]: {exc}", file=sys.stderr)
        return key(lat, lon), None


def fetch_all_nasa(targets: list[dict], workers: int) -> dict[str, dict]:
    print(f"Source: NASA POWER  |  workers={workers}  |  "
          f"~{len(targets) * 3 * _REQUEST_DELAY_S / workers / 60:.0f} min estimated\n")
    locations: dict[str, dict] = {}
    success = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_city_nasa, t["name"], t["lat"], t["lon"]): t["name"]
            for t in targets
        }
        for future in as_completed(futures):
            city_name = futures[future]
            k, record = future.result()
            if record:
                locations[k] = record
                print(f"  ✓ {city_name}: {record['T2M_annual_mean']}°C | "
                      f"{record['PRECTOTCORR_p99']} mm/d | {record['GWETROOT_dry_season']}")
                success += 1
            else:
                failed += 1
    print(f"\n  NASA POWER: {success} ok, {failed} failed")
    return locations


# ── ERA5 fetching ──────────────────────────────────────────────────────────────

def fetch_all_era5(targets: list[dict]) -> dict[str, dict]:
    """
    Download one NetCDF per parameter (covering all of Indonesia), then extract
    values for every target city.  Much faster than per-city CDS requests.
    """
    from app.data_sources import era5

    print(f"Source: ERA5 (Copernicus CDS)  |  {len(targets)} cities\n"
          f"Downloading 3 NetCDF files (~20 years × monthly)…\n")

    # --- temperature ---
    print("  [1/3] 2m_temperature …", flush=True)
    t2m_grid = _era5_load_grid("T2M")

    # --- precipitation ---
    print("  [2/3] total_precipitation …", flush=True)
    pcp_grid = _era5_load_grid("PRECTOTCORR")

    # --- soil wetness ---
    print("  [3/3] volumetric_soil_water_layer_1 …", flush=True)
    gwet_grid = _era5_load_grid("GWETROOT")

    locations: dict[str, dict] = {}
    success = failed = 0

    for t in targets:
        name, lat, lon = t["name"], t["lat"], t["lon"]
        try:
            t2m  = era5.annual_mean_from_grid(t2m_grid,  lat, lon, "T2M")
            p99  = era5.percentile_from_grid(pcp_grid,   lat, lon, "PRECTOTCORR", pct=99.0)
            gwet = era5.dry_season_from_grid(gwet_grid,  lat, lon, "GWETROOT")

            k = key(lat, lon)
            locations[k] = {
                "city": name,
                "T2M_annual_mean":      round(t2m,  2),
                "PRECTOTCORR_p99":      round(p99,  2),
                "GWETROOT_dry_season":  round(gwet, 4),
            }
            print(f"  ✓ {name}: {t2m:.2f}°C | {p99:.2f} mm/d | {gwet:.4f}")
            success += 1
        except Exception as exc:
            print(f"  ERROR [{name}]: {exc}", file=sys.stderr)
            failed += 1

    print(f"\n  ERA5: {success} ok, {failed} failed")
    return locations


def _era5_load_grid(parameter: str):
    """Download Indonesia-wide ERA5 NetCDF and return open xarray Dataset."""
    import tempfile
    from app.data_sources import era5

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp = f.name
    era5._download_era5(era5._CDS_VAR[parameter], BASELINE_START, BASELINE_END, tmp)
    import xarray as xr
    return xr.open_dataset(tmp)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_targets(args) -> list[dict]:
    if args.lat is not None and args.lon is not None:
        return [{"name": args.name, "lat": args.lat, "lon": args.lon}]

    if not CITIES_FILE.exists():
        print(f"Cities file not found: {CITIES_FILE}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CITIES_FILE)

    if args.cities:
        df = df[df["city"].isin(args.cities)]
    if args.province:
        df = df[df["province"] == args.province]

    if df.empty:
        print("No cities matched the filter.", file=sys.stderr)
        sys.exit(1)

    return df[["city", "lat", "lon"]].rename(columns={"city": "name"}).to_dict("records")


def main():
    parser = argparse.ArgumentParser(description="Fetch climate baselines for Indonesian cities")
    parser.add_argument("--source",   choices=["nasa", "era5"], default="nasa",
                        help="Data source: 'nasa' (default) or 'era5'")
    parser.add_argument("--cities",   nargs="+", help="Filter by city name")
    parser.add_argument("--province", type=str,  help="Filter by province name")
    parser.add_argument("--lat",      type=float, help="Custom latitude")
    parser.add_argument("--lon",      type=float, help="Custom longitude")
    parser.add_argument("--name",     type=str,   default="Custom")
    parser.add_argument("--workers",  type=int,   default=4,
                        help="Parallel workers for NASA POWER (default: 4, ignored for ERA5)")
    args = parser.parse_args()

    targets = load_targets(args)

    if args.source == "era5":
        new_locations = fetch_all_era5(targets)
        source_label = "ERA5 (Copernicus CDS, reanalysis-era5-single-levels-monthly-means)"
    else:
        new_locations = fetch_all_nasa(targets, args.workers)
        source_label = "NASA POWER API (https://power.larc.nasa.gov/api/temporal/monthly/point)"

    # Merge into existing fixture (keeps cities not in current run)
    if BASELINES_FILE.exists():
        existing = json.loads(BASELINES_FILE.read_text())
    else:
        existing = {"metadata": {}, "locations": {}}
    locations = existing.get("locations", {})
    locations.update(new_locations)

    output = {
        "metadata": {
            "baseline_period": f"{BASELINE_START}-{BASELINE_END}",
            "source": source_label,
            "parameters": {
                "T2M_annual_mean":     "Annual mean 2m air temperature (°C)",
                "PRECTOTCORR_p99":     "99th-percentile monthly precipitation (mm/day)",
                "GWETROOT_dry_season": "Mean root-zone wetness Jun-Sep (0-1 scale)",
            },
            "city_count": len(locations),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        "locations": locations,
    }

    BASELINES_FILE.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Total in fixture: {len(locations)} cities")
    print(f"✓ Written to {BASELINES_FILE}")


if __name__ == "__main__":
    main()
