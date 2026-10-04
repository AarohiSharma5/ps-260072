"""City thunderstorm nowcast: INSAT-3DS cloud tops + Open-Meteo environment,
trained against observed thunderstorm reports at the city's airport.

    python -m app.nowcast_train --city chennai
    python -m app.nowcast_train --city kolkata

One row per INSAT scene (every 30 min). Target for lead L in {30, 60, 90} min:
a thunderstorm is reported at the airport in (t, t + L].

Validation: 5-fold cross-validation in blocks of 4 consecutive days, so no
storm is split between training and test. Model settings and the alert
threshold are chosen inside each training fold (nested). Scores are computed on
the pooled out-of-fold predictions. Three feature sets are compared:
    env          Open-Meteo environment only
    env+sat      plus INSAT cloud-top temperature / pressure features
    env+sat+obs  plus the airport's own thunderstorm reports of the last hours
Baselines: climatology by hour of day, and persistence (storm reported in the last hour).

The feature builders (sat_row, env_features, obs_features) are shared with app.nowcast_live.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from .cities import CITIES, DEFAULT_CITY, City, get_city
from .config import ARTIFACT_DIR, DATA_DIR
from .nowcast_data import city_dir

SAT_DIRS = [DATA_DIR / "mosdac_ctp", DATA_DIR / "mosdac_ctp_india"]
HALF_DEG = 2.0
LEADS = (30, 60, 90)
FOLDS = 5
BLOCK_DAYS = 4
SEED = 7
OBS_COLS = ["ts_last_hour", "ts_last_3h", "ts_count_3h", "mins_since_ts"]

BASE = dict(subsample=0.8, eval_metric="logloss", n_jobs=4, random_state=SEED)
# Small, strongly regularised candidates: events are scarce, so shallow trees win.
CONFIGS = [
    dict(max_depth=2, n_estimators=150, learning_rate=0.05, min_child_weight=10, reg_lambda=10.0, colsample_bytree=0.6),
    dict(max_depth=3, n_estimators=200, learning_rate=0.04, min_child_weight=8, reg_lambda=10.0, colsample_bytree=0.6),
    dict(max_depth=3, n_estimators=300, learning_rate=0.03, min_child_weight=5, reg_lambda=5.0, colsample_bytree=0.8),
    dict(max_depth=4, n_estimators=250, learning_rate=0.03, min_child_weight=10, reg_lambda=10.0, colsample_bytree=0.6),
]
INNER = 4


def out_dir(city: City):
    return ARTIFACT_DIR / city.key


# ------------------------------------------------------------------ feature builders (shared)

class Geometry:
    """Pixels of the INSAT cloud-top grid inside the 4x4 degree window of a city."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, city: City):
        self.sel = (np.abs(lat - city.lat) <= HALF_DEG) & (np.abs(lon - city.lon) <= HALF_DEG)
        la, lo = lat[self.sel], lon[self.sel]
        self.dist = np.hypot(la - city.lat, lo - city.lon)
        self.east = lo > city.lon
        self.north = la > city.lat

    @property
    def n(self) -> int:
        return int(self.sel.sum())


def sat_row(ctt: np.ndarray, ctp: np.ndarray, g: Geometry) -> dict:
    """Cloud-top features for one scene. ctt/ctp are the window's pixels (already selected)."""
    r: dict = {}
    for name, rad in (("c", 0.5), ("m", 1.0), ("w", 2.0)):
        sel = g.dist <= rad
        v = ctt[sel]
        v = v[np.isfinite(v)]
        r[f"ctt_min_{name}"] = v.min() if v.size else np.nan
        r[f"ctt_mean_{name}"] = v.mean() if v.size else np.nan
        r[f"cold235_{name}"] = float((v < 235).mean()) if v.size else np.nan
        r[f"cold220_{name}"] = float((v < 220).mean()) if v.size else np.nan
        p = ctp[sel]
        p = p[np.isfinite(p)]
        r[f"ctp_min_{name}"] = p.min() if p.size else np.nan
    for name, sel in (("e", g.east), ("w_", ~g.east), ("n", g.north), ("s", ~g.north)):
        v = ctt[sel & (g.dist <= 2.0)]
        v = v[np.isfinite(v)]
        r[f"ctt_min_{name}"] = v.min() if v.size else np.nan
    r["valid_frac"] = float(np.isfinite(ctt).mean())
    return r


def add_sat_trends(sat: pd.DataFrame) -> pd.DataFrame:
    out = sat.copy()
    for col in ("ctt_min_m", "ctt_mean_m", "ctt_min_w", "cold235_m"):
        for lag in (30, 60):
            prev = sat[col].reindex(sat.index - pd.Timedelta(minutes=lag)).to_numpy()
            out[f"{col}_d{lag}"] = sat[col].to_numpy() - prev
    return out


def env_features(wx: pd.DataFrame) -> pd.DataFrame:
    """Hourly Open-Meteo frame (UTC index, raw variable names) -> model environment features."""
    u10 = -wx.wind_speed_10m * np.sin(np.deg2rad(wx.wind_direction_10m))
    v10 = -wx.wind_speed_10m * np.cos(np.deg2rad(wx.wind_direction_10m))
    u850 = -wx.wind_speed_850hPa * np.sin(np.deg2rad(wx.wind_direction_850hPa))
    v850 = -wx.wind_speed_850hPa * np.cos(np.deg2rad(wx.wind_direction_850hPa))
    u500 = -wx.wind_speed_500hPa * np.sin(np.deg2rad(wx.wind_direction_500hPa))
    v500 = -wx.wind_speed_500hPa * np.cos(np.deg2rad(wx.wind_direction_500hPa))
    env = pd.DataFrame(index=wx.index)
    env["temp"] = wx.temperature_2m
    env["rh"] = wx.relative_humidity_2m
    env["dew_dep"] = wx.temperature_2m - wx.dew_point_2m
    env["pressure"] = wx.surface_pressure
    env["cloud"] = wx.cloud_cover
    env["cloud_low"] = wx.cloud_cover_low
    env["cloud_high"] = wx.cloud_cover_high
    env["wind10"] = wx.wind_speed_10m
    env["gust"] = wx.wind_gusts_10m
    env["u10"], env["v10"] = u10, v10
    env["precip"] = wx.precipitation
    env["cape"] = wx.cape
    env["cin"] = wx.convective_inhibition
    env["lifted_index"] = wx.lifted_index
    env["lapse_850_500"] = wx.temperature_850hPa - wx.temperature_500hPa
    env["rh700"] = wx.relative_humidity_700hPa
    env["shear_850_500"] = np.hypot(u500 - u850, v500 - v850)
    env["shear_10_850"] = np.hypot(u850 - u10, v850 - v10)
    for col in ("temp", "cape", "pressure", "rh"):
        env[f"{col}_d3h"] = env[col] - env[col].shift(3)
    return env


def interpolate_env(env: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Linear interpolation of the hourly environment to the scene times."""
    return env.reindex(env.index.union(idx)).interpolate(method="time", limit=3).reindex(idx)


def time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    hours = idx.hour + idx.minute / 60
    doy = idx.dayofyear
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hours / 24), "hour_cos": np.cos(2 * np.pi * hours / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365), "doy_cos": np.cos(2 * np.pi * doy / 365),
        },
        index=idx,
    )


def obs_features(ts_times: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Recent airport thunderstorm reports as seen at each time t (datetime64[ns] arrays)."""
    hi = np.searchsorted(ts_times, t, side="right")
    lo1 = np.searchsorted(ts_times, t - np.timedelta64(60, "m"), side="right")
    lo3 = np.searchsorted(ts_times, t - np.timedelta64(180, "m"), side="right")
    last = hi - 1
    since = np.where(last >= 0, (t - ts_times[np.maximum(last, 0)]) / np.timedelta64(1, "m"), 360.0) if len(ts_times) else np.full(len(t), 360.0)
    return {
        "ts_last_hour": (hi > lo1).astype(int),
        "ts_last_3h": (hi > lo3).astype(int),
        "ts_count_3h": hi - lo3,
        "mins_since_ts": np.minimum(since, 360.0),
    }


def to_ns(values) -> np.ndarray:
    s = pd.Series(pd.to_datetime(values, utc=True))
    return s.dt.tz_convert(None).to_numpy().astype("datetime64[ns]")


# ------------------------------------------------------------------------------- data tables

def load_satellite(city: City) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict] = {}
    for sat_dir in SAT_DIRS:
        pix = sat_dir / "pixels.npz"
        if not pix.exists():
            continue
        p = np.load(pix)
        geo = Geometry(p["lat"], p["lon"], city)
        if geo.n < 40:  # this source does not cover the city
            continue
        for f in sorted(sat_dir.glob("20*.npz")):
            d = np.load(f)
            for t, ctt, ctp in zip(d["times"], d["ctt"], d["ctp"]):
                ts = pd.Timestamp(str(t)[:-1], tz="UTC")
                if ts in rows:
                    continue
                row = sat_row(ctt[geo.sel], ctp[geo.sel], geo)
                row["time"] = ts
                rows[ts] = row
    if not rows:
        raise SystemExit(f"No satellite scenes cover {city.name}. Run app.mosdac fetch first.")
    df = pd.DataFrame(list(rows.values())).sort_values("time").set_index("time")
    return df


def load_weather(city: City) -> pd.DataFrame:
    wx = pd.read_csv(city_dir(city) / "weather_hourly.csv", parse_dates=["time"]).set_index("time")
    wx.index = pd.to_datetime(wx.index, utc=True)
    return env_features(wx)


def load_observations(city: City) -> pd.DataFrame:
    m = pd.read_csv(city_dir(city) / "metar.csv", parse_dates=["time"])
    m["time"] = pd.to_datetime(m["time"], utc=True)
    return m[["time", "thunder"]]


def build_table(city: City) -> pd.DataFrame:
    sat = add_sat_trends(load_satellite(city))
    env = load_weather(city)
    obs = load_observations(city)
    ts_times = to_ns(obs.loc[obs.thunder, "time"].sort_values())

    # keep only scenes inside the city's training window
    lo, hi = pd.Timestamp(city.start, tz="UTC"), pd.Timestamp(city.end, tz="UTC") + pd.Timedelta(days=1)
    sat = sat[(sat.index >= lo) & (sat.index < hi)]
    idx = sat.index
    df = pd.concat([interpolate_env(env, idx), sat], axis=1)
    df = pd.concat([df, time_features(idx)], axis=1)

    t = to_ns(idx)
    for lead in LEADS:
        a = np.searchsorted(ts_times, t, side="right")
        b = np.searchsorted(ts_times, t + np.timedelta64(lead, "m"), side="right")
        df[f"y{lead}"] = (b > a).astype(int)
    for name, vals in obs_features(ts_times, t).items():
        df[name] = vals
    df["day"] = idx.tz_convert(None).normalize()
    return df


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    env_cols = list(load_weather_columns()) + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    skip = set(env_cols) | set(OBS_COLS) | {"day"} | {f"y{lead}" for lead in LEADS}
    sat_cols = [c for c in df.columns if c not in skip]
    return env_cols, sat_cols


def load_weather_columns() -> list[str]:
    """Environment column names without needing any data (built from a dummy frame)."""
    from .nowcast_data import HOURLY

    dummy = pd.DataFrame(np.ones((4, len(HOURLY))), columns=HOURLY, index=pd.date_range("2025-01-01", periods=4, freq="h"))
    return list(env_features(dummy).columns)


# ------------------------------------------------------------------------ evaluation

def fold_ids(days: pd.Series) -> np.ndarray:
    ordinal = days.map(pd.Timestamp.toordinal).to_numpy()
    blocks = ordinal // BLOCK_DAYS
    uniq = np.unique(blocks)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    assign = {b: i % FOLDS for i, b in enumerate(uniq)}
    return np.array([assign[b] for b in blocks])


def contingency(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    csi = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    return dict(threshold=thr, tp=tp, fp=fp, fn=fn, tn=tn, pod=pod, far=far, csi=csi)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 8) -> list[dict]:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi if hi == edges[-1] else p < hi)
        if m.sum():
            out.append(dict(mean_pred=float(p[m].mean()), obs_freq=float(y[m].mean()), n=int(m.sum())))
    return out


def tuned_contingency(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return dict(threshold_mean=float(thr.mean()), tp=tp, fp=fp, fn=fn, tn=tn,
                pod=tp / (tp + fn) if tp + fn else float("nan"),
                far=fp / (tp + fp) if tp + fp else float("nan"),
                csi=tp / (tp + fp + fn) if tp + fp + fn else float("nan"))


def score(y: np.ndarray, p: np.ndarray, clim: np.ndarray, thr: np.ndarray | None = None) -> dict:
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, clim)
    return dict(
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        brier=float(brier),
        brier_skill_vs_climatology=float(1 - brier / brier_ref),
        at_0_20=contingency(y, p, 0.20),
        at_0_35=contingency(y, p, 0.35),
        at_0_50=contingency(y, p, 0.50),
        at_tuned=tuned_contingency(y, p, thr) if thr is not None else None,
        reliability=reliability(y, p),
    )


def climatology_oof(df: pd.DataFrame, target: str, folds: np.ndarray) -> np.ndarray:
    out = np.zeros(len(df))
    bucket = (df.index.hour // 3).to_numpy()
    for k in range(FOLDS):
        tr, te = folds != k, folds == k
        rate = pd.Series(df[target].to_numpy()[tr]).groupby(bucket[tr]).mean()
        out[te] = pd.Series(bucket[te]).map(rate).fillna(df[target].to_numpy()[tr].mean()).to_numpy()
    return out


def inner_folds(blocks: np.ndarray, salt: int) -> np.ndarray:
    uniq = np.unique(blocks)
    rng = np.random.default_rng(SEED + salt)
    rng.shuffle(uniq)
    assign = {b: i % INNER for i, b in enumerate(uniq)}
    return np.array([assign[b] for b in blocks])


def select_config(X: pd.DataFrame, y: np.ndarray, blocks: np.ndarray, salt: int):
    """Pick the candidate with the lowest inner-CV log loss; also return the
    inner out-of-fold probabilities of that candidate (to choose a threshold)."""
    inner = inner_folds(blocks, salt)
    best = (np.inf, 0, None)
    for ci, cfg in enumerate(CONFIGS):
        oof = np.zeros(len(y))
        for k in range(INNER):
            tr, te = inner != k, inner == k
            if y[tr].sum() == 0:
                oof[te] = y.mean()
                continue
            m = XGBClassifier(**BASE, **cfg).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        loss = log_loss(y, np.clip(oof, 1e-4, 1 - 1e-4))
        if loss < best[0]:
            best = (loss, ci, oof)
    return best[1], best[2]


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.8, 31)
    csi = np.nan_to_num([contingency(y, p, g)["csi"] for g in grid])
    return float(grid[int(np.argmax(csi))])


def oof_predict(df: pd.DataFrame, cols: list[str], target: str, folds: np.ndarray, blocks: np.ndarray):
    """Nested CV. Returns out-of-fold probabilities, the per-row tuned threshold
    and the chosen config index per outer fold."""
    out = np.zeros(len(df))
    thr = np.zeros(len(df))
    chosen = []
    X, y = df[cols], df[target].to_numpy()
    for k in range(FOLDS):
        tr, te = folds != k, folds == k
        ci, inner_oof = select_config(X[tr], y[tr], blocks[tr], salt=k)
        thr[te] = best_threshold(y[tr], inner_oof)
        model = XGBClassifier(**BASE, **CONFIGS[ci]).fit(X[tr], y[tr])
        out[te] = model.predict_proba(X[te])[:, 1]
        chosen.append(ci)
    return out, thr, chosen


def main(city_key: str = DEFAULT_CITY) -> None:
    city = get_city(city_key)
    out = out_dir(city)
    out.mkdir(parents=True, exist_ok=True)
    df = build_table(city)
    env_cols, sat_cols = split_columns(df)
    # Keep clear-sky scenes: no cloud-top retrieval means no cloud, which is informative.
    df = df.dropna(subset=["valid_frac"]).copy()
    folds = fold_ids(df["day"])
    blocks = df["day"].map(pd.Timestamp.toordinal).to_numpy() // BLOCK_DAYS
    print(f"{city.name}: scenes {len(df)}  days {df['day'].nunique()}  {df.index.min()} .. {df.index.max()}")
    print("features: env", len(env_cols), " sat", len(sat_cols), " obs", len(OBS_COLS))

    sets = {
        "env": env_cols,
        "env+sat": env_cols + sat_cols,
        "env+sat+obs": env_cols + sat_cols + OBS_COLS,
    }
    results: dict = {"leads": {}}
    oof_store = pd.DataFrame(index=df.index)
    for lead in LEADS:
        target = f"y{lead}"
        y = df[target].to_numpy()
        clim = climatology_oof(df, target, folds)
        res = {
            "events": int(y.sum()), "rows": int(len(y)), "event_rate": float(y.mean()),
            "climatology": score(y, clim, clim),
            "persistence": contingency(y, df["ts_last_hour"].to_numpy().astype(float), 0.5),
        }
        for name, cols in sets.items():
            p, thr, chosen = oof_predict(df, cols, target, folds, blocks)
            oof_store[f"{name}_{lead}"] = p
            res[name] = score(y, p, clim, thr)
            res[name]["configs_chosen"] = chosen
            t = res[name]["at_tuned"]
            print(f"+{lead:>2} min  {name:12s} AUC {res[name]['roc_auc']:.3f}  PR-AUC {res[name]['pr_auc']:.3f}  "
                  f"BSS {res[name]['brier_skill_vs_climatology']:+.3f}  tuned thr {t['threshold_mean']:.2f}: "
                  f"POD {t['pod']:.2f} FAR {t['far']:.2f} CSI {t['csi']:.3f}")
        results["leads"][str(lead)] = res
        print(f"+{lead:>2} min  events {res['events']}/{res['rows']}  climatology AUC {res['climatology']['roc_auc']:.3f}  "
              f"persistence POD {res['persistence']['pod']:.2f} FAR {res['persistence']['far']:.2f} CSI {res['persistence']['csi']:.2f}")

    # Final models on all days, for the two sets a live forecast can use.
    importance, final = {}, {}
    for key, name in (("env_sat", "env+sat"), ("env_sat_obs", "env+sat+obs")):
        cols = sets[name]
        final[key] = {"columns": cols, "leads": {}}
        for lead in LEADS:
            yy = df[f"y{lead}"].to_numpy()
            ci, inner_oof = select_config(df[cols], yy, blocks, salt=99)
            model = XGBClassifier(**BASE, **CONFIGS[ci]).fit(df[cols], yy)
            model.save_model(out / f"xgb_{key}_{lead}.json")
            final[key]["leads"][str(lead)] = {
                "threshold": best_threshold(yy, inner_oof), "base_rate": float(yy.mean()), "config": ci,
            }
            if key == "env_sat":
                imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
                importance[str(lead)] = (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()
    results["feature_importance_gain"] = importance
    results["final_models"] = final
    results["meta"] = dict(
        city=city.key, city_name=city.name, station=city.station, lat=city.lat, lon=city.lon,
        trained_at=datetime.now(timezone.utc).isoformat(),
        scenes=int(len(df)), days=int(df["day"].nunique()),
        first=str(df.index.min()), last=str(df.index.max()),
        folds=FOLDS, block_days=BLOCK_DAYS, season=city.season,
        feature_sets={k: len(v) for k, v in sets.items()},
        label=f"Thunderstorm (TS or VCTS in current weather) reported at {city.station} within the lead window",
        satellite=f"INSAT-3DS 3SIMG_L2B_CTP (MOSDAC), 4x4 degree window around {city.lat}N {city.lon}E",
        environment="Open-Meteo historical-forecast API (model data)",
    )
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    df.assign(**{c: oof_store[c] for c in oof_store.columns}).to_csv(out / "oof_predictions.csv")
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=DEFAULT_CITY, choices=sorted(CITIES))
    main(ap.parse_args().city)
