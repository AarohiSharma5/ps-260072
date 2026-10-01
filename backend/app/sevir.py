"""Turn the shared SEVIR Drive archive into daily-style NPZ cases.

The archive is Storm EVent ImageryR (SEVIR): United States storm patches.
The Mumbai-analogue CSV only selects which of those US storms to keep.
Patches are not moved to Mumbai.

Downloaded fields, and only when both exist for an event:
    processed/radar/vil/{id}_vil_phys.npy   VIL, kg/m², shape (384, 384, 49)
    processed/lightning/{id}_lght_grid.npy  flash counts, same shape

Each 8×8 block becomes one 8 km cell (block mean for VIL, block sum for flashes).
Native 1 km files are deleted after the NPZ is written.

    python -m app.sevir build

Infrared H5 files and events that lack VIL are left unused.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .config import SEVIR_DIR, SEVIR_META_DIR, VIL_KG_M2

RADAR_FOLDER = "1ecGkoIkeysSYZpMXIwTD3sx8VQi-jQtV"
LIGHTNING_FOLDER = "1mBqZVpCmZj58NP81MPbNUmBmqIIh0xD6"
FACTOR = 8
SOURCE_NAME = (
    "SEVIR storm patches (Veillette et al., NeurIPS 2020), VIL and lightning, "
    "selected by the Mumbai-analogue catalog. US storms, not Mumbai observations."
)


def _folder_files(folder_id: str) -> dict[str, str]:
    html = subprocess.check_output(
        ["curl", "-fsSL", "--max-time", "60", f"https://drive.google.com/embeddedfolderview?id={folder_id}"],
        text=True,
    )
    titles = re.findall(r'flip-entry-title\">([^<]+)', html)
    ids = re.findall(r"/file/d/([A-Za-z0-9_-]+)", html)
    if len(titles) != len(ids):
        raise SystemExit(f"Drive listing for {folder_id} did not pair names and ids.")
    return dict(zip(titles, ids))


def _download(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    last: Exception | None = None
    for attempt in range(2):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "gdown", file_id, "-O", str(dest)],
                stdout=subprocess.DEVNULL,
            )
            return
        except subprocess.CalledProcessError as exc:
            last = exc
            dest.unlink(missing_ok=True)
            wait = 25
            print(f"retry {file_id} in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"could not download {file_id}") from last


def _coarse(array: np.ndarray, how: str) -> np.ndarray:
    """(y, x, time) native image -> (time, y, x) with row 0 at the south edge."""
    y, x, t = array.shape
    y2, x2 = y // FACTOR, x // FACTOR
    view = array[: y2 * FACTOR, : x2 * FACTOR].reshape(y2, FACTOR, x2, FACTOR, t)
    reduced = view.mean(axis=(1, 3)) if how == "mean" else view.sum(axis=(1, 3))
    reduced = np.flip(reduced, axis=0)  # image row 0 is north
    return np.moveaxis(reduced, -1, 0).astype(np.float64)


def _catalog() -> dict[str, dict[str, str]]:
    path = SEVIR_META_DIR / "mumbai_analogue_100_events.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {row["id"]: row for row in rows}


def _write_case(event_id: str, row: dict[str, str], vil_native: np.ndarray, lightning_native: np.ndarray) -> Path:
    vil = _coarse(vil_native, "mean")
    flashes = _coarse(lightning_native, "sum")
    n_time, n_lat, n_lon = vil.shape
    south, north = float(row["llcrnrlat"]), float(row["urcrnrlat"])
    west, east = float(row["llcrnrlon"]), float(row["urcrnrlon"])
    lat = np.linspace(south, north, n_lat)
    lon = np.linspace(west, east, n_lon)
    center = datetime.strptime(row["time_utc"], "%Y-%m-%d %H:%M:%S")
    times = [(center + timedelta(minutes=offset)).strftime("%Y-%m-%dT%H:%MZ") for offset in range(-120, 125, 5)]
    if len(times) != n_time:
        raise ValueError(f"{event_id} has {n_time} frames, catalog step implies {len(times)}.")
    stamp = center.strftime("%Y%m%dT%H%MZ")
    target = SEVIR_DIR / f"{stamp}-{event_id}.npz"
    np.savez_compressed(
        target,
        lat=lat,
        lon=lon,
        dt_min=np.int64(5),
        times=np.array(times),
        domain_name=np.array(f"SEVIR {event_id} (US patch, Mumbai-analogue selection)"),
        source_name=np.array(SOURCE_NAME),
        vil=vil,
        lightning_count=flashes,
    )
    hot = int((vil >= VIL_KG_M2).sum())
    flashes_n = int((flashes >= 1).sum())
    print(f"wrote {target.name}  vil>={VIL_KG_M2:g} {hot}  flashes {flashes_n}", flush=True)
    return target


def build(workers: int = 4) -> list[Path]:
    catalog = _catalog()
    radar = _folder_files(RADAR_FOLDER)
    lightning = _folder_files(LIGHTNING_FOLDER)
    paired = sorted(
        event_id for event_id in catalog
        if f"{event_id}_vil_phys.npy" in radar and f"{event_id}_lght_grid.npy" in lightning
    )
    missing_vil = sorted(event_id for event_id in catalog if f"{event_id}_vil_phys.npy" not in radar)
    print(f"{len(paired)} events have VIL and lightning. {len(missing_vil)} catalogue events have no VIL file and are skipped.", flush=True)
    SEVIR_DIR.mkdir(parents=True, exist_ok=True)
    raw = SEVIR_DIR / "_raw"
    raw.mkdir(exist_ok=True)

    def one(event_id: str) -> Path:
        row = catalog[event_id]
        stamp = datetime.strptime(row["time_utc"], "%Y-%m-%d %H:%M:%S").strftime("%Y%m%dT%H%MZ")
        target = SEVIR_DIR / f"{stamp}-{event_id}.npz"
        if target.exists():
            print(f"exists {target.name}", flush=True)
            return target
        vil_path = raw / f"{event_id}_vil_phys.npy"
        lght_path = raw / f"{event_id}_lght_grid.npy"
        try:
            _download(radar[f"{event_id}_vil_phys.npy"], vil_path)
            _download(lightning[f"{event_id}_lght_grid.npy"], lght_path)
        except RuntimeError as exc:
            print(f"skip {event_id}: {exc}", flush=True)
            return target
        try:
            if not vil_path.exists() or not lght_path.exists():
                return target
            written_path = _write_case(event_id, row, np.load(vil_path), np.load(lght_path))
            return written_path
        finally:
            vil_path.unlink(missing_ok=True)
            lght_path.unlink(missing_ok=True)

    written: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, event_id): event_id for event_id in paired}
        for future in as_completed(futures):
            written.append(future.result())
    print(f"{len(written)} cases in {SEVIR_DIR}. Now run: python -m app.train")
    return written


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    build(args.workers)


if __name__ == "__main__":
    main(sys.argv[1:])
