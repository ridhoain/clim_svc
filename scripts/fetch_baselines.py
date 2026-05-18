"""
Fetch NASA POWER baseline climate data for a set of cities and write to
data/baselines.json. Run this locally (or in CI) where outbound network
access to power.larc.nasa.gov is available.

Usage:
    python scripts/fetch_baselines.py
    python scripts/fetch_baselines.py --cities Jakarta Surabaya
    python scripts/fetch_baselines.py --lat -6.2088 --lon 106.8456 --name MyCity
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import BASELINE_START, BASELINE_END
from app.data_sources import nasa_power

BASELINES_FILE = Path(__file__).parent.parent / "data" / "baselines.json"

CITIES = {
    "Jakarta":   {"lat": -6.2088, "lon": 106.8456},
    "Surabaya":  {"lat": -7.2575, "lon": 112.7521},
    "Medan":     {"lat":  3.5952, "lon":  98.6722},
    "Makassar":  {"lat": -5.1477, "lon": 119.4327},
    "Bandung":   {"lat": -6.9175, "lon": 107.6191},
    "Semarang":  {"lat": -6.9667, "lon": 110.4167},
    "Palembang": {"lat": -2.9761, "lon": 104.7754},
    "Denpasar":  {"lat": -8.6500, "lon": 115.2167},
}


def fetch_city(name: str, lat: float, lon: float) -> dict:
    print(f"  Fetching T2M        for {name}...", end=" ", flush=True)
    t2m = nasa_power.annual_mean(lat, lon, "T2M", BASELINE_START, BASELINE_END)
    print(f"{t2m:.2f}°C")

    print(f"  Fetching PRECTOTCORR for {name}...", end=" ", flush=True)
    p99 = nasa_power.percentile_monthly(lat, lon, "PRECTOTCORR", BASELINE_START, BASELINE_END, pct=99.0)
    print(f"{p99:.2f} mm/day")

    print(f"  Fetching GWETROOT   for {name}...", end=" ", flush=True)
    gwet = nasa_power.dry_season_mean(lat, lon, "GWETROOT", BASELINE_START, BASELINE_END)
    print(f"{gwet:.4f}")

    return {
        "city": name,
        "T2M_annual_mean": round(t2m, 2),
        "PRECTOTCORR_p99": round(p99, 2),
        "GWETROOT_dry_season": round(gwet, 4),
    }


def key(lat: float, lon: float) -> str:
    return f"{lat:.2f},{lon:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Fetch NASA POWER baselines")
    parser.add_argument("--cities", nargs="+", choices=list(CITIES), help="Subset of cities to fetch")
    parser.add_argument("--lat", type=float, help="Custom latitude")
    parser.add_argument("--lon", type=float, help="Custom longitude")
    parser.add_argument("--name", type=str, default="Custom", help="Name for custom location")
    args = parser.parse_args()

    # Load existing file to merge results
    BASELINES_FILE.parent.mkdir(exist_ok=True)
    if BASELINES_FILE.exists():
        existing = json.loads(BASELINES_FILE.read_text())
    else:
        existing = {"metadata": {}, "locations": {}}

    targets = {}
    if args.lat is not None and args.lon is not None:
        targets[args.name] = {"lat": args.lat, "lon": args.lon}
    else:
        subset = args.cities or list(CITIES)
        targets = {name: CITIES[name] for name in subset}

    locations = existing.get("locations", {})

    for name, coords in targets.items():
        lat, lon = coords["lat"], coords["lon"]
        print(f"\n[{name}] lat={lat}, lon={lon}")
        try:
            locations[key(lat, lon)] = fetch_city(name, lat, lon)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    output = {
        "metadata": {
            "baseline_period": f"{BASELINE_START}-{BASELINE_END}",
            "source": "NASA POWER API (https://power.larc.nasa.gov/api/temporal/monthly/point)",
            "parameters": {
                "T2M_annual_mean": "Annual mean 2m air temperature (°C)",
                "PRECTOTCORR_p99": "99th-percentile monthly precipitation (mm/day)",
                "GWETROOT_dry_season": "Mean root-zone wetness Jun-Sep (0-1 scale)",
            },
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        "locations": locations,
    }

    BASELINES_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(locations)} location(s) to {BASELINES_FILE}")


if __name__ == "__main__":
    main()
