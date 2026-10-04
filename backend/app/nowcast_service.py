"""Read-only access to the per-city nowcast results for the web app.

Everything here comes from files written by app.nowcast_train / app.nowcast_report:
    artifacts/<city>/metrics.json     cross-validated skill and feature importance
    artifacts/<city>/timeline.csv     per-scene features and out-of-fold probabilities
    artifacts/<city>/ts_reports.csv   thunderstorm reports at the city airport

The probabilities are out-of-fold: each day was scored by a model that never saw it.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import pandas as pd

from .cities import CITIES, get_city
from .config import ARTIFACT_DIR


def art_dir(city: str):
    return ARTIFACT_DIR / get_city(city).key


SETS = ("env", "env+sat", "env+sat+obs")
LEADS = (30, 60, 90)

SET_LABELS = {
    "env": "Weather model only",
    "env+sat": "Weather model + INSAT-3DS satellite",
    "env+sat+obs": "Weather model + satellite + airport reports (last 3 h)",
}

FEATURE_LABELS = {
    "ctt_min_m": ("Coldest cloud top within 1°", "satellite"),
    "ctt_min_c": ("Coldest cloud top within 0.5°", "satellite"),
    "ctt_min_w": ("Coldest cloud top within 2°", "satellite"),
    "ctt_min_w_": ("Coldest cloud top, west side", "satellite"),
    "ctt_min_e": ("Coldest cloud top, east (sea) side", "satellite"),
    "ctt_min_n": ("Coldest cloud top, north side", "satellite"),
    "ctt_min_s": ("Coldest cloud top, south side", "satellite"),
    "ctt_mean_m": ("Mean cloud-top temperature within 1°", "satellite"),
    "ctt_mean_c": ("Mean cloud-top temperature within 0.5°", "satellite"),
    "ctt_mean_w": ("Mean cloud-top temperature within 2°", "satellite"),
    "ctt_mean_m_d30": ("Cloud-top cooling over 30 min", "satellite"),
    "ctt_mean_m_d60": ("Cloud-top cooling over 60 min", "satellite"),
    "ctt_min_m_d30": ("Coldest cloud-top change, 30 min", "satellite"),
    "ctt_min_m_d60": ("Coldest cloud-top change, 60 min", "satellite"),
    "ctt_min_w_d30": ("Coldest cloud-top change (2°), 30 min", "satellite"),
    "ctt_min_w_d60": ("Coldest cloud-top change (2°), 60 min", "satellite"),
    "cold235_m": ("Cold-cloud fraction (<235 K) within 1°", "satellite"),
    "cold235_c": ("Cold-cloud fraction (<235 K) within 0.5°", "satellite"),
    "cold235_w": ("Cold-cloud fraction (<235 K) within 2°", "satellite"),
    "cold235_m_d30": ("Cold-cloud fraction change, 30 min", "satellite"),
    "cold235_m_d60": ("Cold-cloud fraction change, 60 min", "satellite"),
    "cold220_m": ("Very cold cloud fraction (<220 K) within 1°", "satellite"),
    "cold220_c": ("Very cold cloud fraction (<220 K) within 0.5°", "satellite"),
    "cold220_w": ("Very cold cloud fraction (<220 K) within 2°", "satellite"),
    "ctp_min_m": ("Highest cloud top (lowest pressure) within 1°", "satellite"),
    "ctp_min_c": ("Highest cloud top within 0.5°", "satellite"),
    "ctp_min_w": ("Highest cloud top within 2°", "satellite"),
    "valid_frac": ("Cloudiness (share of pixels with cloud top)", "satellite"),
    "cape": ("CAPE (storm energy)", "environment"),
    "cin": ("CIN (convective inhibition)", "environment"),
    "lifted_index": ("Lifted index", "environment"),
    "rh": ("Relative humidity", "environment"),
    "rh700": ("Humidity at 700 hPa", "environment"),
    "dew_dep": ("Dew-point depression", "environment"),
    "shear_850_500": ("Wind shear 850-500 hPa", "environment"),
    "shear_10_850": ("Wind shear surface-850 hPa", "environment"),
    "lapse_850_500": ("Lapse rate 850-500 hPa", "environment"),
    "temp": ("Temperature", "environment"),
    "pressure": ("Surface pressure", "environment"),
    "cloud": ("Cloud cover (model)", "environment"),
    "cloud_low": ("Low cloud cover (model)", "environment"),
    "cloud_high": ("High cloud cover (model)", "environment"),
    "wind10": ("10 m wind speed", "environment"),
    "gust": ("Wind gust", "environment"),
    "u10": ("10 m wind, eastward", "environment"),
    "v10": ("10 m wind, northward", "environment"),
    "precip": ("Model precipitation", "environment"),
    "hour_sin": ("Time of day", "environment"),
    "hour_cos": ("Time of day", "environment"),
    "doy_sin": ("Season", "environment"),
    "doy_cos": ("Season", "environment"),
    "temp_d3h": ("Temperature change, 3 h", "environment"),
    "cape_d3h": ("CAPE change, 3 h", "environment"),
    "pressure_d3h": ("Pressure change, 3 h", "environment"),
    "rh_d3h": ("Humidity change, 3 h", "environment"),
}


class NowcastUnavailable(RuntimeError):
    pass


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


@lru_cache(maxsize=8)
def _metrics(city: str) -> dict:
    path = art_dir(city) / "metrics.json"
    if not path.exists():
        raise NowcastUnavailable(f"No results for {city}. Run: python -m app.nowcast_train --city {city} && python -m app.nowcast_report --city {city}")
    return json.loads(path.read_text())


@lru_cache(maxsize=8)
def _timeline(city: str) -> pd.DataFrame:
    path = art_dir(city) / "timeline.csv"
    if not path.exists():
        raise NowcastUnavailable(f"Timeline for {city} is missing. Run: python -m app.nowcast_report --city {city}")
    return pd.read_csv(path)


@lru_cache(maxsize=8)
def _reports(city: str) -> pd.DataFrame:
    path = art_dir(city) / "ts_reports.csv"
    if not path.exists():
        return pd.DataFrame(columns=["time", "metar"])
    return pd.read_csv(path)


def available(city: str) -> bool:
    try:
        d = art_dir(city)
    except KeyError:
        return False
    return (d / "metrics.json").exists() and (d / "timeline.csv").exists()


def cities() -> list[dict]:
    out = []
    for key, c in CITIES.items():
        out.append({"key": key, "name": c.name, "station": c.station, "lat": c.lat, "lon": c.lon,
                    "season": c.season, "ready": available(key)})
    return out


def domains() -> list[dict]:
    """Map footprint of every city: window, INSAT pixels, airport, and whether results exist."""
    out = []
    for key, c in CITIES.items():
        path = ARTIFACT_DIR / key / "domain.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        d["ready"] = available(key)
        d["season"] = c.season
        out.append(d)
    return out


def summary(city: str) -> dict:
    m = _metrics(city)
    leads = {}
    for lead, block in m["leads"].items():
        sets = {}
        for name in SETS:
            r = block[name]
            tuned = r["at_tuned"]
            sets[name] = {
                "label": SET_LABELS[name],
                "roc_auc": _clean(r["roc_auc"]),
                "pr_auc": _clean(r["pr_auc"]),
                "brier_skill": _clean(r["brier_skill_vs_climatology"]),
                "pod": _clean(tuned["pod"]),
                "far": _clean(tuned["far"]),
                "csi": _clean(tuned["csi"]),
                "threshold": _clean(tuned["threshold_mean"]),
                "tp": tuned["tp"], "fp": tuned["fp"], "fn": tuned["fn"], "tn": tuned["tn"],
                "reliability": r["reliability"],
            }
        pers = block["persistence"]
        leads[lead] = {
            "events": block["events"],
            "rows": block["rows"],
            "event_rate": block["event_rate"],
            "sets": sets,
            "persistence": {"pod": _clean(pers["pod"]), "far": _clean(pers["far"]), "csi": _clean(pers["csi"])},
        }
    importance = {}
    for lead, gains in m["feature_importance_gain"].items():
        rows = []
        for name, gain in gains.items():
            label, group = FEATURE_LABELS.get(name, (name, "environment"))
            rows.append({"feature": name, "label": label, "group": group, "gain": gain})
        importance[lead] = rows
    meta = m["meta"]
    return {
        "meta": {
            "city": meta.get("city", city),
            "city_name": meta.get("city_name", get_city(city).name),
            "station": meta.get("station", get_city(city).station),
            "season": meta.get("season", ""),
            "scenes": meta["scenes"],
            "days": meta["days"],
            "first": meta["first"],
            "last": meta["last"],
            "folds": meta["folds"],
            "block_days": meta["block_days"],
            "label": meta["label"],
            "satellite": meta["satellite"],
            "environment": meta["environment"],
            "trained_at": meta["trained_at"],
            "feature_sets": meta.get("feature_sets", {}),
            "lat": meta.get("lat", get_city(city).lat),
            "lon": meta.get("lon", get_city(city).lon),
        },
        "set_labels": SET_LABELS,
        "leads": leads,
        "importance": importance,
        "notes": [
            "Scores are cross-validated: each 4-day block was predicted by a model that never saw it.",
            f"The label is a thunderstorm (TS or VCTS) reported at {meta.get('city_name', city)} airport ({meta.get('station', '')}). It is not a lightning-network record.",
            "Satellite input is INSAT-3DS cloud-top temperature and pressure from MOSDAC, at about 35 km pixel spacing.",
            "Environment fields are Open-Meteo model data, not observations.",
            f"{meta['days']} days of 2025 ({meta.get('season', '')}); one location; one year.",
        ],
    }


def days(city: str) -> list[dict]:
    tl = _timeline(city)
    rep = _reports(city)
    rep_day = pd.to_datetime(rep["time"]).dt.strftime("%Y-%m-%d") if len(rep) else pd.Series(dtype=str)
    counts = rep_day.value_counts().to_dict()
    out = []
    for day, g in tl.groupby("day"):
        out.append(
            {
                "day": day,
                "scenes": int(len(g)),
                "storm_scenes": int(g["y60"].sum()),
                "reports": int(counts.get(day, 0)),
                "coldest_k": _clean(float(g["ctt_min_m"].min())) if g["ctt_min_m"].notna().any() else None,
            }
        )
    out.sort(key=lambda d: (-d["storm_scenes"], d["day"]))
    return out


def day(city: str, day_id: str) -> dict:
    tl = _timeline(city)
    g = tl[tl["day"] == day_id]
    if g.empty:
        raise KeyError(day_id)
    rep = _reports(city)
    mask = pd.to_datetime(rep["time"]).dt.strftime("%Y-%m-%d").eq(day_id) if len(rep) else []
    # include reports shortly after midnight UTC that fall inside the last windows of the day
    reports = rep[mask] if len(rep) else rep
    series = {
        "ctt_min": [_clean(v) for v in g["ctt_min_m"].tolist()],
        "cold220": [_clean(v) for v in g["cold220_m"].tolist()],
        "ctp_min": [_clean(v) for v in g["ctp_min_m"].tolist()],
        "cloudiness": [_clean(v) for v in g["valid_frac"].tolist()],
        "cape": [_clean(v) for v in g["cape"].tolist()],
    }
    prob = {name: {str(L): [_clean(v) for v in g[f"{name}_{L}"].tolist()] for L in LEADS} for name in SETS}
    truth = {str(L): g[f"y{L}"].astype(int).tolist() for L in LEADS}
    return {
        "day": day_id,
        "times": g["time"].tolist(),
        "series": series,
        "probability": prob,
        "truth": truth,
        "reports": [{"time": r.time, "metar": r.metar} for r in reports.itertuples()],
    }
