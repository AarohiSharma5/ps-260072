"""Live nowcast for a city: newest INSAT-3DS scene + current weather forecast + latest airport reports.

    python -m app.nowcast_live --city chennai

Uses the same feature code as training (app.nowcast_train), so a live row is built exactly like a
training row. Needs MOSDAC credentials on the server (MOSDAC_USER / MOSDAC_PASS). A new scene
arrives every 30 minutes; only scenes not seen before are downloaded (about 48 files a day).

Experimental. The models were trained on 2025 April-December data for one location; this is a
research prototype, not a warning.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

from . import mosdac
from .cities import CITIES, DEFAULT_CITY, City, get_city
from .config import ARTIFACT_DIR
from .nowcast_data import HOURLY, fetch_metar_range
from .nowcast_train import (
    LEADS,
    Geometry,
    add_sat_trends,
    env_features,
    interpolate_env,
    obs_features,
    sat_row,
    time_features,
    to_ns,
)

CACHE_SECONDS = 300
SCENE_TOLERANCE_MIN = 10


class LiveUnavailable(RuntimeError):
    pass


_lock = threading.Lock()
_session: mosdac.Session | None = None
_geometry: dict[str, Geometry] = {}
_scene_cache: dict[tuple[str, str], dict] = {}
_result_cache: dict[str, tuple[float, dict]] = {}
_failures: dict[str, tuple[float, str]] = {}
FAILURE_SECONDS = 90
_models: dict[tuple[str, str, int], XGBClassifier] = {}


def _get_session() -> mosdac.Session:
    global _session
    if _session is None:
        try:
            _session = mosdac.Session()
        except SystemExit as exc:  # Session() exits on a failed login
            raise LiveUnavailable(str(exc)) from exc
        except (FileNotFoundError, KeyError) as exc:
            raise LiveUnavailable(
                "Live mode needs MOSDAC credentials on the server (set MOSDAC_USER and MOSDAC_PASS)."
            ) from exc
    return _session


# ------------------------------------------------------------------------------ satellite

def list_recent_scenes(now: datetime) -> list[tuple[datetime, dict]]:
    """Newest-first list of (scene time UTC, MOSDAC entry) for yesterday and today."""
    entries: list[dict] = []
    for d in (now.date() - timedelta(days=1), now.date()):
        try:
            entries += mosdac.search_day(d)
        except requests.RequestException as exc:
            raise LiveUnavailable(f"MOSDAC search failed: {exc}") from exc
    scenes = [(mosdac._scene_time(e["identifier"]), e) for e in entries]
    scenes.sort(key=lambda s: s[0], reverse=True)
    return scenes


def scene_features(blob: bytes, city: City) -> dict:
    """Cloud-top features of one downloaded scene (same maths as the training table)."""
    import h5py

    with h5py.File(io.BytesIO(blob), "r") as h5:
        geo = _geometry.get(city.key)
        if geo is None:
            lat = h5["Latitude"][:].astype(np.float64) / 100.0
            lon = h5["Longitude"][:].astype(np.float64) / 100.0
            geo = _geometry[city.key] = Geometry(lat, lon, city)
        ctt = h5["CTT"][0][geo.sel].astype(np.float32)
        ctp = h5["CTP"][0][geo.sel].astype(np.float32)
    ctt[ctt <= 0] = np.nan
    ctp[ctp <= 0] = np.nan
    return sat_row(ctt, ctp, geo)


def _scene_row(city: City, when: datetime, entry: dict) -> dict:
    key = (city.key, entry["identifier"])
    cached = _scene_cache.get(key)
    if cached is not None:
        return cached
    try:
        blob = mosdac._download(_get_session(), entry["id"])
    except mosdac.QuotaReached as exc:
        raise LiveUnavailable(f"MOSDAC daily download limit reached: {exc}") from exc
    except (requests.RequestException, RuntimeError) as exc:
        raise LiveUnavailable(f"MOSDAC download failed: {exc}") from exc
    row = scene_features(blob, city)
    row["time"] = pd.Timestamp(when, tz="UTC")
    _scene_cache[key] = row
    if len(_scene_cache) > 400:
        for old in list(_scene_cache)[:100]:
            _scene_cache.pop(old, None)
    return row


# --------------------------------------------------------------------------- environment

def live_environment(city: City) -> pd.DataFrame:
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": city.lat, "longitude": city.lon, "hourly": ",".join(HOURLY),
                "past_days": 2, "forecast_days": 2, "timezone": "UTC",
            },
            timeout=60,
        )
        r.raise_for_status()
    except requests.RequestException as exc:
        raise LiveUnavailable(f"Open-Meteo request failed: {exc}") from exc
    wx = pd.DataFrame(r.json()["hourly"])
    wx["time"] = pd.to_datetime(wx.pop("time"), utc=True)
    wx = wx.set_index("time")
    return env_features(wx)


# ------------------------------------------------------------------------------ models

def _model(city: City, key: str, lead: int) -> XGBClassifier:
    k = (city.key, key, lead)
    m = _models.get(k)
    if m is None:
        path = ARTIFACT_DIR / city.key / f"xgb_{key}_{lead}.json"
        if not path.exists():
            raise LiveUnavailable(f"Trained model missing: {path.name}. Run python -m app.nowcast_train --city {city.key}")
        m = XGBClassifier()
        m.load_model(str(path))
        _models[k] = m
    return m


def _final_meta(city: City) -> dict:
    path = ARTIFACT_DIR / city.key / "metrics.json"
    if not path.exists():
        raise LiveUnavailable(f"No trained results for {city.name}.")
    return json.loads(path.read_text())["final_models"]


def build_feature_row(
    city: City, scene: pd.Series, sat_hist: pd.DataFrame, env: pd.DataFrame, ts_times: np.ndarray
) -> pd.DataFrame:
    """One training-style row for the scene time."""
    idx = pd.DatetimeIndex([scene.name])
    env_row = interpolate_env(env, idx)
    sat = add_sat_trends(sat_hist).loc[idx]
    row = pd.concat([env_row, sat, time_features(idx)], axis=1)
    for name, vals in obs_features(ts_times, to_ns(idx)).items():
        row[name] = vals
    return row


# ---------------------------------------------------------------------------------- main

def forecast(city_key: str = DEFAULT_CITY, force: bool = False) -> dict:
    city = get_city(city_key)
    with _lock:
        hit = _result_cache.get(city.key)
        if hit and not force and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
        bad = _failures.get(city.key)
        if bad and not force and time.time() - bad[0] < FAILURE_SECONDS:
            if hit:
                return {**hit[1], "stale": True, "stale_reason": bad[1]}
            raise LiveUnavailable(bad[1])
        try:
            result = _forecast(city)
        except LiveUnavailable as exc:
            _failures[city.key] = (time.time(), str(exc))
            if hit:  # keep showing the last good forecast, clearly marked
                return {**hit[1], "stale": True, "stale_reason": str(exc)}
            raise
        _failures.pop(city.key, None)
        _result_cache[city.key] = (time.time(), result)
        return result


def _forecast(city: City) -> dict:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    final = _final_meta(city)

    scenes = list_recent_scenes(now)
    if not scenes:
        raise LiveUnavailable("MOSDAC lists no recent INSAT-3DS scenes.")
    latest_t, latest_entry = scenes[0]
    # the latest scene plus the two before it (30 and 60 minutes earlier) for the cooling trends
    wanted = [(latest_t, latest_entry)]
    for lag in (30, 60):
        target = latest_t - timedelta(minutes=lag)
        best = min(scenes, key=lambda s: abs((s[0] - target).total_seconds()))
        if abs((best[0] - target).total_seconds()) <= SCENE_TOLERANCE_MIN * 60 and best[0] != latest_t:
            wanted.append(best)
    rows = [_scene_row(city, t, e) for t, e in wanted]
    # trends use exact 30/60 minute lags; align the older scenes onto that grid
    sat_hist = pd.DataFrame(rows).set_index("time").sort_index()
    scene_time = pd.Timestamp(latest_t, tz="UTC")
    for lag in (30, 60):
        want = scene_time - pd.Timedelta(minutes=lag)
        near = [i for i in sat_hist.index if abs((i - want).total_seconds()) <= SCENE_TOLERANCE_MIN * 60]
        if near and near[0] != want:
            sat_hist = sat_hist.rename(index={near[0]: want})
    scene = sat_hist.iloc[-1]

    env = live_environment(city)

    start = pd.Timestamp(now) - pd.Timedelta(hours=14)
    metar = fetch_metar_range(city.station, start, pd.Timestamp(now) + pd.Timedelta(days=1))
    ts_all = to_ns(metar.loc[metar.thunder, "time"].sort_values()) if len(metar) else np.array([], dtype="datetime64[ns]")

    row = build_feature_row(city, scene, sat_hist, env, ts_all)
    nan_share = float(row[final["env_sat"]["columns"]].isna().mean(axis=1).iloc[0])

    models = {}
    for key, label in (("env_sat", "satellite + weather model"), ("env_sat_obs", "satellite + weather model + airport reports")):
        cols = final[key]["columns"]
        per = {}
        for lead in LEADS:
            p = float(_model(city, key, lead).predict_proba(row[cols])[:, 1][0])
            meta = final[key]["leads"][str(lead)]
            per[str(lead)] = {"probability": p, "threshold": meta["threshold"], "base_rate": meta["base_rate"],
                              "alert": p >= meta["threshold"]}
        models[key] = {"label": label, "leads": per}

    after = metar[(metar["time"] > scene_time)] if len(metar) else metar
    reports_since = [{"time": t.strftime("%Y-%m-%dT%H:%MZ"), "metar": m}
                     for t, m, th in zip(after["time"], after["metar"], after["thunder"]) if th] if len(after) else []
    latest_metar = metar.iloc[-1] if len(metar) else None
    r = row.iloc[0]

    def val(name):
        v = r.get(name)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    return {
        "city": city.key, "city_name": city.name, "station": city.station,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "scene_time": scene_time.strftime("%Y-%m-%dT%H:%MZ"),
        "scene_age_min": round((pd.Timestamp(now, tz="UTC") - scene_time).total_seconds() / 60),
        "scenes_used": [t.strftime("%Y-%m-%dT%H:%MZ") for t, _ in wanted],
        "models": models,
        "drivers": {
            "coldest_cloud_top_k": val("ctt_min_m"),
            "very_cold_fraction": val("cold220_m"),
            "cloud_top_change_30min_k": val("ctt_mean_m_d30"),
            "highest_cloud_top_hpa": val("ctp_min_m"),
            "cape": val("cape"),
            "lifted_index": val("lifted_index"),
            "shear_850_500": val("shear_850_500"),
            "humidity_700hpa": val("rh700"),
            "temperature": val("temp"),
        },
        "airport": {
            "storm_reported_last_hour": bool(row["ts_last_hour"].iloc[0]),
            "storm_reported_last_3h": bool(row["ts_last_3h"].iloc[0]),
            "latest_report": None if latest_metar is None else {
                "time": latest_metar["time"].strftime("%Y-%m-%dT%H:%MZ"), "metar": latest_metar["metar"]},
            "storm_reports_since_scene": reports_since,
        },
        "quality": {"missing_feature_share": nan_share},
        "notes": [
            "Experimental. Trained on 2025 data for one location; a research prototype, not a warning.",
            "Probabilities are for a thunderstorm reported at the airport, counted from the satellite scene time.",
            "Environment fields are the Open-Meteo forecast for this hour, not observations.",
        ],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=DEFAULT_CITY, choices=sorted(CITIES))
    print(json.dumps(forecast(ap.parse_args().city, force=True), indent=2))
