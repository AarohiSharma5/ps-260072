"""INSAT-3DS cloud-top fields around Chennai, fetched from MOSDAC.

Uses the same public endpoints as MOSDAC's own mdapi.py client, but keeps one
login, downloads files in parallel in memory, and stores only a 4 x 4 degree
window per day. Raw ~2 MB files are never written to disk.

    export MOSDAC_USER=...  MOSDAC_PASS=...      # or keep backend/tools/mosdac/config.json
    python -m app.mosdac fetch --start 2025-04-01 --end 2025-06-30

Product 3SIMG_L2B_CTP (INSAT-3DS imager, one scene every 30 min):
    CTT  cloud-top temperature (K)       CTP  cloud-top pressure (hPa)
Both are a 9x9-pixel-box product (~0.35 degree spacing). Negative values are
fill (no cloud-top retrieval) and become NaN.

Output: data/mosdac_ctp_india/YYYY-MM-DD.npz with times (UTC ISO), ctt, ctp (T, N),
and pixels.npz with pixel lat/lon (N,). Region 'chennai' is the original small window.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import requests

from .config import DATA_DIR

DATASET = "3SIMG_L2B_CTP"

# name -> (output dir, lat_min, lat_max, lon_min, lon_max). "chennai" is the original 4x4 degree
# window; "india" covers Chennai, Kolkata, Mumbai, Delhi, Hyderabad and Bhubaneswar in one pass.
REGIONS = {
    "chennai": (DATA_DIR / "mosdac_ctp", 11.11, 15.11, 78.25, 82.25),
    "india": (DATA_DIR / "mosdac_ctp_india", 8.0, 30.0, 70.0, 92.0),
}
OUT_DIR = REGIONS["india"][0]
BOUNDS = REGIONS["india"][1:]


def use_region(name: str) -> None:
    global OUT_DIR, BOUNDS
    OUT_DIR, *bounds = REGIONS[name]
    BOUNDS = tuple(bounds)

TOKEN_URL = "https://mosdac.gov.in/download_api/gettoken"
SEARCH_URL = "https://mosdac.gov.in/apios/datasets.json"
DOWNLOAD_URL = "https://mosdac.gov.in/download_api/download"
REFRESH_URL = "https://mosdac.gov.in/download_api/refresh-token"
LOGOUT_URL = "https://mosdac.gov.in/download_api/logout"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "tools" / "mosdac" / "config.json"


class QuotaReached(RuntimeError):
    pass


def _credentials() -> tuple[str, str]:
    user, pw = os.environ.get("MOSDAC_USER"), os.environ.get("MOSDAC_PASS")
    if user and pw:
        return user, pw
    cfg = json.loads(CONFIG_PATH.read_text())["user_credentials"]
    return cfg["username/email"], cfg["password"]


class Session:
    def __init__(self) -> None:
        self.user, self.password = _credentials()
        self.lock = threading.Lock()
        self.access = ""
        self.refresh = ""
        self.login()

    def login(self) -> None:
        r = requests.post(TOKEN_URL, json={"username": self.user, "password": self.password}, timeout=30)
        if r.status_code != 200:
            raise SystemExit(f"MOSDAC login failed ({r.status_code}): {r.text[:200]}")
        body = r.json()
        self.access, self.refresh = body["access_token"], body["refresh_token"]

    def renew(self, stale: str) -> None:
        with self.lock:
            if self.access != stale:
                return
            r = requests.post(REFRESH_URL, json={"refresh_token": self.refresh}, timeout=30)
            if r.status_code == 200:
                body = r.json()
                self.access, self.refresh = body["access_token"], body["refresh_token"]
            else:
                self.login()

    def logout(self) -> None:
        try:
            requests.post(LOGOUT_URL, json={"username": self.user}, timeout=10)
        except requests.RequestException:
            pass


def search_day(day: date) -> list[dict]:
    entries: list[dict] = []
    start = 1
    while True:
        r = requests.get(
            SEARCH_URL,
            params={"datasetId": DATASET, "startTime": day.isoformat(), "endTime": day.isoformat(), "startIndex": start},
            timeout=60,
        )
        r.raise_for_status()
        body = r.json()
        got = body.get("entries", [])
        entries += got
        if len(entries) >= int(body.get("totalResults", 0)) or not got:
            return entries
        start += 100


def _download(session: Session, record_id: str) -> bytes:
    for attempt in range(8):
        token = session.access
        r = requests.get(
            DOWNLOAD_URL,
            headers={"Authorization": f"Bearer {token}"},
            params={"id": record_id},
            timeout=120,
        )
        if r.status_code == 401:
            session.renew(token)
            continue
        if r.status_code == 429:
            info = r.json()
            if info.get("type") == "daily_limit":
                raise QuotaReached(info.get("message", "daily limit"))
            time.sleep(20)
            continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.content
    raise RuntimeError(f"download failed for {record_id}")


def _scene_time(identifier: str) -> datetime:
    m = re.search(r"_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", identifier)
    if not m:
        raise ValueError(identifier)
    return datetime.strptime(m.group(1) + m.group(2), "%d%b%Y%H%M")


def _pixel_mask(h5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = h5["Latitude"][:].astype(np.float64) / 100.0
    lon = h5["Longitude"][:].astype(np.float64) / 100.0
    lat0, lat1, lon0, lon1 = BOUNDS
    mask = (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)
    return mask, lat[mask], lon[mask]


def _extract(blob: bytes, mask: np.ndarray | None):
    import h5py

    with h5py.File(io.BytesIO(blob), "r") as h5:
        if mask is None:
            mask, lat, lon = _pixel_mask(h5)
        else:
            lat = lon = None
        ctt = h5["CTT"][0][mask].astype(np.float32)
        ctp = h5["CTP"][0][mask].astype(np.float32)
    ctt[ctt <= 0] = np.nan
    ctp[ctp <= 0] = np.nan
    return mask, lat, lon, ctt, ctp


def _fetch_one(session: Session, entry: dict, mask: np.ndarray | None):
    blob = _download(session, entry["id"])
    return _scene_time(entry["identifier"]), _extract(blob, mask)


def fetch_day(session: Session, day: date, workers: int = 6) -> int:
    out = OUT_DIR / f"{day.isoformat()}.npz"
    if out.exists():
        return 0
    entries = search_day(day)
    if not entries:
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pix = OUT_DIR / "pixels.npz"
    mask = None
    first = None
    if pix.exists():
        mask_file = OUT_DIR / "mask.npy"
        mask = np.load(mask_file) if mask_file.exists() else None
    rows = []
    todo = list(entries)
    if mask is None:
        t0, (mask, lat, lon, ctt, ctp) = _fetch_one(session, todo.pop(0), None)
        np.save(OUT_DIR / "mask.npy", mask)
        np.savez(pix, lat=lat, lon=lon)
        rows.append((t0, ctt, ctp))
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(_fetch_one, session, e, mask) for e in todo]
        for f in as_completed(futs):
            t, (_, _, _, ctt, ctp) = f.result()
            rows.append((t, ctt, ctp))
    rows.sort(key=lambda r: r[0])
    tmp = OUT_DIR / f"part-{day.isoformat()}.npz"  # written whole, then renamed: readers never see half a file
    np.savez_compressed(
        tmp,
        times=np.array([r[0].strftime("%Y-%m-%dT%H:%MZ") for r in rows]),
        ctt=np.stack([r[1] for r in rows]),
        ctp=np.stack([r[2] for r in rows]),
    )
    tmp.replace(out)
    return len(rows)


def cmd_fetch(args: argparse.Namespace) -> None:
    use_region(args.region)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    days = [start + timedelta(days=i) for i in range(0, (end - start).days + 1, args.step)]
    session = Session()
    total = 0
    try:
        for day in days:
            failures = 0
            while True:
                try:
                    n = fetch_day(session, day, args.workers)
                    break
                except QuotaReached as exc:
                    if not args.wait_on_quota:
                        print(f"STOP: {exc}. Rerun later; finished days are kept.", flush=True)
                        return
                    print(f"{datetime.now():%H:%M} quota reached; sleeping {args.wait_min} min ({exc})", flush=True)
                    time.sleep(args.wait_min * 60)
                    session.login()
                except Exception as exc:  # network hiccup: retry the day, then skip it
                    failures += 1
                    print(f"{day} error {exc!r} (attempt {failures})", flush=True)
                    if failures >= 3:
                        n = 0
                        break
                    time.sleep(30)
            total += n
            print(f"{day} scenes={n} total={total}", flush=True)
    finally:
        session.logout()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--step", type=int, default=1, help="take every Nth day")
    f.add_argument("--region", choices=sorted(REGIONS), default="india")
    f.add_argument("--wait-on-quota", action="store_true", help="sleep and retry when the daily limit is hit")
    f.add_argument("--wait-min", type=int, default=30)
    f.set_defaults(fn=cmd_fetch)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
