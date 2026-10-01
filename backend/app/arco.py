"""Seed ERA5 single-level grids from the public ARCO-ERA5 zarr.

This is the same Copernicus dataset as
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
(DOI 10.24381/cds.adbb2d47, CC-BY). The zarr is the Google public copy
(gs://gcp-public-data-arco-era5), used because this machine has no CDS token.
Each hourly chunk is the full globe; only the configured box is kept.

    python -m app.arco scan --start 2025-05-01 --end 2025-05-31
    python -m app.arco seed --start 2025-05-08 --end 2025-05-21

Not downloaded, and not invented:
    ERA5 pressure levels (shear) — one global 37-level chunk per hour
    IMD radar                     — viewer, no archive grid
    INSAT-3D / 3DS                — MOSDAC login
    GPM IMERG                     — PPS login
    LIS VHRMC                     — Earthdata login, and it is a monthly climatology
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import numpy as np

from .config import ERA5_AREA, ERA5_DEFAULT_END, ERA5_DEFAULT_START, ERA5_DOMAIN_NAME, HISTORICAL_DIR
from .era5 import _rh_percent

STORE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
EPOCH = datetime(1940, 1, 1)
SOURCE_NAME = (
    "Copernicus ERA5 hourly single levels (reanalysis-era5-single-levels, DOI 10.24381/cds.adbb2d47) "
    "via the public ARCO-ERA5 zarr"
)

# Short name -> ARCO variable. Units match the CDS GRIB parameters.
VARIABLES = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "cape": "convective_available_potential_energy",
    "cin": "convective_inhibition",
    "kx": "k_index",
    "totalx": "total_totals_index",
    "cp": "convective_precipitation",
}


def _curl(url: str) -> bytes:
    last: Exception | None = None
    for _ in range(4):
        try:
            return subprocess.check_output(
                ["curl", "-fsS", "--retry", "2", "--max-time", "180", url],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            last = exc
    raise RuntimeError(f"download failed: {url}") from last


def _coords() -> tuple[np.ndarray, np.ndarray]:
    import numcodecs

    lat = np.frombuffer(numcodecs.blosc.decompress(_curl(f"{STORE}/latitude/0")), dtype="<f4")
    lon = np.frombuffer(numcodecs.blosc.decompress(_curl(f"{STORE}/longitude/0")), dtype="<f4")
    return lat, lon


def _window(lat: np.ndarray, lon: np.ndarray, area: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_i = np.where((lat >= area["south"] - 1e-6) & (lat <= area["north"] + 1e-6))[0]
    lon_i = np.where((lon >= area["west"] - 1e-6) & (lon <= area["east"] + 1e-6))[0]
    if lat_i.size == 0 or lon_i.size == 0:
        raise SystemExit(f"No ERA5 cells inside {area}.")
    return lat_i, lon_i, lat[lat_i], lon[lon_i]


def _hour_index(moment: datetime) -> int:
    return int((moment - EPOCH).total_seconds() // 3600)


def _slice(variable: str, hour_index: int, lat_i: np.ndarray, lon_i: np.ndarray) -> np.ndarray:
    import numcodecs

    raw = _curl(f"{STORE}/{variable}/{hour_index}.0.0")
    grid = np.frombuffer(numcodecs.blosc.decompress(raw), dtype="<f4").reshape(721, 1440)
    return np.asarray(grid[np.ix_(lat_i, lon_i)], dtype=np.float64)


def _days(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def _fetch_day(day: date, names: list[str], lat_i: np.ndarray, lon_i: np.ndarray, workers: int) -> dict[str, np.ndarray]:
    hours = [datetime(day.year, day.month, day.day) + timedelta(hours=h) for h in range(24)]
    jobs = [(short, _hour_index(moment), ti) for ti, moment in enumerate(hours) for short in names]
    stacked = {short: np.empty((24, lat_i.size, lon_i.size), dtype=np.float64) for short in names}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_slice, VARIABLES[short], hour_index, lat_i, lon_i): (short, ti)
            for short, hour_index, ti in jobs
        }
        done = 0
        for future in as_completed(futures):
            short, ti = futures[future]
            stacked[short][ti] = future.result()
            done += 1
            if done % 48 == 0 or done == len(futures):
                print(f"  {day} {done}/{len(futures)} chunks", flush=True)
    return stacked


def _orient(lat: np.ndarray, fields: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if lat.size > 1 and lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        fields = {key: np.flip(value, axis=1) for key, value in fields.items()}
    return lat, fields


def scan(start: date, end: date, area: dict[str, float], workers: int) -> None:
    """Sample convective precipitation every 3 hours so a wet window can be chosen before a full seed."""
    lat, lon = _coords()
    lat_i, lon_i, lat_box, _lon_box = _window(lat, lon, area)
    print(f"grid {lat_box.size} x {lon_i.size}  south={float(lat_box.min()):.2f} north={float(lat_box.max()):.2f}", flush=True)
    sample_hours = (0, 3, 6, 9, 12, 15, 18, 21)
    for day in _days(start, end):
        jobs = []
        for hour in sample_hours:
            moment = datetime(day.year, day.month, day.day, hour)
            jobs.append((hour, _hour_index(moment)))
        samples = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_slice, VARIABLES["cp"], index, lat_i, lon_i): hour for hour, index in jobs}
            for future in as_completed(futures):
                samples.append(future.result())
        cp = np.stack(samples) * 1000.0
        cells = int(np.nansum(cp >= 1.0))
        print(f"{day}  sampled cp>=1mm/h cells {cells}  max {float(np.nanmax(cp)):.2f} mm/h", flush=True)


def seed(start: date, end: date, area: dict[str, float], domain_name: str, workers: int) -> list:
    lat, lon = _coords()
    lat_i, lon_i, lat_box, lon_box = _window(lat, lon, area)
    if (end - start).days + 1 > 40:
        raise SystemExit("Keep a seed under 40 days. Each day downloads about 300 MB of global chunks.")
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    names = list(VARIABLES)
    for day in _days(start, end):
        target = HISTORICAL_DIR / f"era5-{day.isoformat()}.npz"
        if target.exists():
            print(f"exists {target.name}", flush=True)
            written.append(target)
            continue
        print(f"seeding {day}", flush=True)
        raw = _fetch_day(day, names, lat_i, lon_i, workers)
        lat_asc, raw = _orient(lat_box, raw)
        t2m = raw["t2m"]
        payload: dict[str, np.ndarray] = {
            "lat": lat_asc.astype(np.float64),
            "lon": lon_box.astype(np.float64),
            "dt_min": np.int64(60),
            "times": np.array([f"{day.isoformat()}T{hour:02d}:00Z" for hour in range(24)]),
            "domain_name": np.array(domain_name),
            "source_name": np.array(SOURCE_NAME),
            "temperature": t2m - 273.15,
            "rh": _rh_percent(t2m, raw["d2m"]),
            "pressure": raw["sp"] / 100.0,
            "wind_u": raw["u10"],
            "wind_v": raw["v10"],
            "cape": raw["cape"],
            "k_index": raw["kx"],
            "total_totals": raw["totalx"],
        }
        cin = raw["cin"]
        cin = np.where(np.isfinite(cin), cin, 0.0)
        payload["cin"] = -np.abs(cin)
        cp = raw["cp"] * 1000.0
        payload["convective_precip"] = np.clip(np.where(np.isfinite(cp), cp, 0.0), 0.0, None)
        np.savez_compressed(target, **payload)
        hot = int((payload["convective_precip"] >= 1.0).sum())
        print(
            f"wrote {target.name}  frames=24  cells={lat_asc.size}x{lon_box.size}  "
            f"cp>=1mm/h={hot}  T={np.nanmin(payload['temperature']):.1f}..{np.nanmax(payload['temperature']):.1f}C  "
            f"K={np.nanmin(payload['k_index']):.1f}..{np.nanmax(payload['k_index']):.1f}",
            flush=True,
        )
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["scan", "seed"])
    parser.add_argument("--start", default=ERA5_DEFAULT_START)
    parser.add_argument("--end", default=ERA5_DEFAULT_END)
    parser.add_argument("--north", type=float, default=ERA5_AREA["north"])
    parser.add_argument("--south", type=float, default=ERA5_AREA["south"])
    parser.add_argument("--west", type=float, default=ERA5_AREA["west"])
    parser.add_argument("--east", type=float, default=ERA5_AREA["east"])
    parser.add_argument("--domain-name", default=ERA5_DOMAIN_NAME)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        raise SystemExit("--end is before --start")
    area = {"north": args.north, "south": args.south, "west": args.west, "east": args.east}
    if args.command == "scan":
        scan(start, end, area, args.workers)
    else:
        written = seed(start, end, area, args.domain_name, args.workers)
        print(f"{len(written)} cases in {HISTORICAL_DIR}. Now run: python -m app.train")


if __name__ == "__main__":
    main(sys.argv[1:])
