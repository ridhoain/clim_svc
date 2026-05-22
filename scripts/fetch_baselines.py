"""
Fetch NASA POWER baseline climate data for Indonesian cities and write
to data/baselines.json. Supports parallel requests with rate limiting.

Usage:
    python scripts/fetch_baselines.py                        # all cities from data/cities.csv
    python scripts/fetch_baselines.py --cities Jakarta Bali  # filter by city name
    python scripts/fetch_baselines.py --province "Jawa Barat"
    python scripts/fetch_baselines.py --lat -6.21 --lon 106.85 --name MyCity
    python scripts/fetch_baselines.py --workers 4            # parallel workers (default: 4)
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
_REQUEST_DELAY_S = 2.2   # seconds between requests per worker


def key(lat: float, lon: float) -> str:
    return f"{lat:.2f},{lon:.2f}"


def fetch_city(name: str, lat: float, lon: float) -> tuple[str, dict | None]:
    """Fetch all three parameters for one city. Returns (key, record) or (key, None) on error."""
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


def load_targets(args) -> list[dict]:
    """Build list of {name, lat, lon} dicts from CLI args."""
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
    parser = argparse.ArgumentParser(description="Fetch NASA POWER baselines for Indonesian cities")
    parser.add_argument("--cities",   nargs="+", help="Filter by city name")
    parser.add_argument("--province", type=str,  help="Filter by province name")
    parser.add_argument("--lat",      type=float, help="Custom latitude")
    parser.add_argument("--lon",      type=float, help="Custom longitude")
    parser.add_argument("--name",     type=str,   default="Custom")
    parser.add_argument("--workers",  type=int,   default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()

    targets = load_targets(args)
    print(f"Fetching baselines for {len(targets)} cities "
          f"(~{len(targets) * 3 * _REQUEST_DELAY_S / args.workers / 60:.0f} min estimated)…\n")

    # Load existing fixture to merge results
    if BASELINES_FILE.exists():
        existing = json.loads(BASELINES_FILE.read_text())
    else:
        existing = {"metadata": {}, "locations": {}}
    locations = existing.get("locations", {})

    success, failed = 0, 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_city, t["name"], t["lat"], t["lon"]): t["name"]
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

    output = {
        "metadata": {
            "baseline_period": f"{BASELINE_START}-{BASELINE_END}",
            "source": "NASA POWER API (https://power.larc.nasa.gov/api/temporal/monthly/point)",
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
    print(f"\n✓ {success} cities fetched, {failed} failed")
    print(f"✓ Total in fixture: {len(locations)} cities")
    print(f"✓ Written to {BASELINES_FILE}")


if __name__ == "__main__":
    main()
