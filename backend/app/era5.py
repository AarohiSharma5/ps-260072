"""Fetch ERA5 hourly single-level fields from the Copernicus Climate Data Store
and convert them into daily NPZ cases for the trainer.

    python -m app.era5 fetch   [--start 2025-04-01 --end 2025-05-31]
    python -m app.era5 convert
    python -m app.era5 all     # fetch + convert

Requires a free CDS account and ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <your personal access token>

Dataset: reanalysis-era5-single-levels (DOI 10.24381/cds.adbb2d47), CC-BY.
Variables used and what they become:
    2m_temperature                          -> temperature (°C)
    2m_dewpoint_temperature + 2m_temperature -> rh (%)   Magnus formula
    surface_pressure                        -> pressure (hPa)
    10m_u/v_component_of_wind               -> wind_u, wind_v (m/s)
    convective_available_potential_energy   -> cape (J/kg)
    convective_inhibition                   -> cin (J/kg, negative = inhibition)
    k_index                                 -> k_index (°C)
    total_totals_index                      -> total_totals (°C)
    convective_precipitation                -> convective_precip (mm/h)  proxy label

Not in this dataset and therefore left absent: radar reflectivity, satellite
brightness temperature, lightning, bulk wind shear.
"""

from __future__ import annotations

import argparse
import calendar
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from .config import ERA5_AREA, ERA5_DEFAULT_END, ERA5_DEFAULT_START, ERA5_DOMAIN_NAME, ERA5_RAW_DIR, HISTORICAL_DIR

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "convective_available_potential_energy",
    "convective_inhibition",
    "k_index",
    "total_totals_index",
    "convective_precipitation",
]

# NetCDF short names as delivered by the CDS.
SHORT = {
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


def _month_ranges(start: date, end: date):
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        last_day = calendar.monthrange(cursor.year, cursor.month)[1]
        month_start = max(start, cursor)
        month_end = min(end, date(cursor.year, cursor.month, last_day))
        yield month_start, month_end
        cursor = date(cursor.year + (cursor.month // 12), (cursor.month % 12) + 1, 1)


def fetch(start: date, end: date, area: dict[str, float]) -> list[Path]:
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install cdsapi") from exc
    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        raise SystemExit(
            "No CDS credentials. Create ~/.cdsapirc with two lines:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <token from https://cds.climate.copernicus.eu/profile>\n"
            "You also have to accept the ERA5 licence once on the dataset page."
        )
    ERA5_RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    outputs = []
    for month_start, month_end in _month_ranges(start, end):
        target = ERA5_RAW_DIR / f"era5_{month_start:%Y%m%d}_{month_end:%Y%m%d}.nc"
        if target.exists():
            print(f"exists  {target.name}")
            outputs.append(target)
            continue
        days = [f"{d:02d}" for d in range(month_start.day, month_end.day + 1)]
        request = {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [f"{month_start.year}"],
            "month": [f"{month_start.month:02d}"],
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [area["north"], area["west"], area["south"], area["east"]],
        }
        print(f"request {target.name} ({len(days)} days)…", flush=True)
        client.retrieve("reanalysis-era5-single-levels", request, str(target))
        outputs.append(target)
    return outputs


def _rh_percent(t_k: np.ndarray, td_k: np.ndarray) -> np.ndarray:
    t_c = t_k - 273.15
    td_c = td_k - 273.15
    a, b = 17.625, 243.04
    es = np.exp(a * t_c / (b + t_c))
    e = np.exp(a * td_c / (b + td_c))
    return np.clip(100.0 * e / es, 0.0, 100.0)


def convert(raw_dir: Path = ERA5_RAW_DIR, out_dir: Path = HISTORICAL_DIR, domain_name: str = ERA5_DOMAIN_NAME) -> list[Path]:
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install xarray netCDF4") from exc
    files = sorted(raw_dir.glob("*.nc"))
    if not files:
        raise SystemExit(f"No NetCDF files in {raw_dir}. Run: python -m app.era5 fetch")
    ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords") if len(files) > 1 else xr.open_dataset(files[0])
    time_name = "valid_time" if "valid_time" in ds.dims else "time"
    for drop in ("number", "expver"):
        if drop in ds.coords and drop not in ds.dims:
            ds = ds.reset_coords(drop, drop=True)
        elif drop in ds.dims:
            ds = ds.isel({drop: 0}, drop=True)
    ds = ds.sortby("latitude")  # ascending south -> north
    ds = ds.sortby(time_name)
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    times = ds[time_name].values.astype("datetime64[m]")
    present = {short for short in SHORT if short in ds.data_vars}
    missing = sorted(set(SHORT) - present)
    if missing:
        print(f"note: not in files, left absent: {', '.join(SHORT[m] for m in missing)}")
    if "cp" not in present:
        raise SystemExit("convective_precipitation is missing; there is no label source.")

    def arr(short: str) -> np.ndarray:
        return np.asarray(ds[short].values, dtype=np.float64)

    step = int(np.median(np.diff(times)).astype("timedelta64[m]").astype(int)) if times.size > 1 else 60
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    days = np.unique(times.astype("datetime64[D]"))
    for day in days:
        sel = np.where(times.astype("datetime64[D]") == day)[0]
        if sel.size < 8:
            print(f"skip {day}: only {sel.size} frames")
            continue
        payload: dict[str, np.ndarray] = {
            "lat": lat,
            "lon": lon,
            "dt_min": np.int64(step),
            "times": np.array([str(t) + "Z" for t in times[sel]]),
            "domain_name": np.array(domain_name),
            "source_name": np.array("Copernicus ERA5 hourly single levels (reanalysis-era5-single-levels)"),
        }
        if "t2m" in present:
            t2m = arr("t2m")[sel]
            payload["temperature"] = t2m - 273.15
            if "d2m" in present:
                payload["rh"] = _rh_percent(t2m, arr("d2m")[sel])
        if "sp" in present:
            payload["pressure"] = arr("sp")[sel] / 100.0
        if "u10" in present and "v10" in present:
            payload["wind_u"] = arr("u10")[sel]
            payload["wind_v"] = arr("v10")[sel]
        if "cape" in present:
            payload["cape"] = arr("cape")[sel]
        if "cin" in present:
            cin = arr("cin")[sel]
            cin = np.where(np.isfinite(cin), cin, 0.0)  # ERA5 masks CIN where CAPE is zero
            payload["cin"] = -np.abs(cin)  # store as negative = inhibition, matching the simulator sign
        if "kx" in present:
            payload["k_index"] = arr("kx")[sel]
        if "totalx" in present:
            payload["total_totals"] = arr("totalx")[sel]
        cp = arr("cp")[sel] * 1000.0  # m accumulated over the hour -> mm/h
        payload["convective_precip"] = np.clip(np.where(np.isfinite(cp), cp, 0.0), 0.0, None)
        for key, value in list(payload.items()):
            if isinstance(value, np.ndarray) and value.dtype.kind == "f" and not np.isfinite(value).all():
                payload[key] = np.where(np.isfinite(value), value, np.nan)
                bad = int((~np.isfinite(value)).sum())
                print(f"warn {day} {key}: {bad} non-finite values kept as NaN")
        target = out_dir / f"era5-{str(day)}.npz"
        np.savez_compressed(target, **payload)
        written.append(target)
        print(f"wrote {target.name}  frames={sel.size}  cells={lat.size}x{lon.size}  cp>=1mm/h cells={(payload['convective_precip'] >= 1.0).sum()}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "convert", "all"])
    parser.add_argument("--start", default=ERA5_DEFAULT_START)
    parser.add_argument("--end", default=ERA5_DEFAULT_END)
    parser.add_argument("--north", type=float, default=ERA5_AREA["north"])
    parser.add_argument("--south", type=float, default=ERA5_AREA["south"])
    parser.add_argument("--west", type=float, default=ERA5_AREA["west"])
    parser.add_argument("--east", type=float, default=ERA5_AREA["east"])
    parser.add_argument("--domain-name", default=ERA5_DOMAIN_NAME)
    args = parser.parse_args(argv)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        raise SystemExit("--end is before --start")
    if (end - start) > timedelta(days=200):
        raise SystemExit("Keep the period under 200 days for this prototype.")
    area = {"north": args.north, "south": args.south, "west": args.west, "east": args.east}
    if args.command in ("fetch", "all"):
        fetch(start, end, area)
    if args.command in ("convert", "all"):
        written = convert(domain_name=args.domain_name)
        print(f"{len(written)} cases in {HISTORICAL_DIR}. Now run: python -m app.train")


if __name__ == "__main__":
    main(sys.argv[1:])
