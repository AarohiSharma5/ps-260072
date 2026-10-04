"""Charts and compact app files for a city nowcast. Reads artifacts/<city>/{metrics.json,oof_predictions.csv}.

    python -m app.nowcast_report --city chennai
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cities import CITIES, DEFAULT_CITY, get_city
from .config import ARTIFACT_DIR
from .nowcast_data import city_dir

OUT = ARTIFACT_DIR / DEFAULT_CITY  # rebound per city in main()
CITY_NAME = "Chennai"
COLORS = {"env": "#9aa5b1", "env+sat": "#1f77b4", "env+sat+obs": "#d62728"}
LABELS = {
    "env": "Weather model only",
    "env+sat": "+ INSAT-3DS satellite",
    "env+sat+obs": "+ airport reports (last 3 h)",
}


def skill_chart(m: dict) -> None:
    leads = list(m["leads"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (key, title) in zip(
        axes, (("roc_auc", "ROC-AUC (higher is better)"), ("pr_auc", "Precision-recall AUC"), ("csi", "Critical success index"))
    ):
        w = 0.2
        x = np.arange(len(leads))
        for i, name in enumerate(("env", "env+sat", "env+sat+obs")):
            vals = []
            for L in leads:
                r = m["leads"][L][name]
                vals.append(r["at_tuned"]["csi"] if key == "csi" else r[key])
            ax.bar(x + (i - 1) * w, vals, w, color=COLORS[name], label=LABELS[name])
        if key == "pr_auc":
            ax.plot(x, [m["leads"][L]["event_rate"] for L in leads], "k_", ms=28, mew=2, label="Chance (event rate)")
        if key == "roc_auc":
            ax.axhline(0.5, color="k", lw=1, ls=":")
        if key == "csi":
            ax.plot(x, [m["leads"][L]["persistence"]["csi"] for L in leads], "k_", ms=28, mew=2, label="Persistence (storm in last hour)")
        ax.set_xticks(x, [f"+{L} min" for L in leads])
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[2].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{CITY_NAME} thunderstorm nowcast — out-of-fold skill (4-day blocks, 5-fold, nested tuning)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "skill.png", dpi=140)
    plt.close(fig)


def reliability_chart(oof: pd.DataFrame) -> None:
    edges = np.array([0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 1.0001])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, L in zip(axes, (30, 60, 90)):
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        y = oof[f"y{L}"].to_numpy()
        for name in ("env+sat", "env+sat+obs"):
            p = oof[f"{name}_{L}"].to_numpy()
            xs, ys, ns = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (p >= lo) & (p < hi)
                if m.sum() >= 15:
                    xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
            ax.plot(xs, ys, "o-", color=COLORS[name], label=LABELS[name])
        ax.set_title(f"+{L} min", fontsize=11)
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Reliability: do forecast probabilities match how often storms happened?  (bins with >=15 cases)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "reliability.png", dpi=140)
    plt.close(fig)


def importance_chart(m: dict) -> None:
    imp = pd.Series(m["feature_importance_gain"]["60"]).sort_values()
    nice = {
        "ctt_min_m": "Coldest cloud top (1°)", "ctt_min_c": "Coldest cloud top (0.5°)", "ctt_min_w": "Coldest cloud top (2°)",
        "ctt_mean_m": "Mean cloud-top temp (1°)", "ctt_mean_c": "Mean cloud-top temp (0.5°)",
        "cold235_m": "Cold-cloud fraction <235 K (1°)", "cold220_m": "Very cold fraction <220 K (1°)",
        "cold235_w": "Cold-cloud fraction (2°)", "cold220_w": "Very cold fraction (2°)",
        "ctp_min_m": "Highest cloud top, pressure (1°)", "ctp_min_w": "Highest cloud top, pressure (2°)",
        "valid_frac": "Cloudiness (valid pixels)", "cape": "CAPE", "cin": "CIN", "lifted_index": "Lifted index",
        "rh": "Relative humidity", "rh700": "Humidity at 700 hPa", "dew_dep": "Dew-point depression",
        "shear_850_500": "Wind shear 850-500 hPa", "shear_10_850": "Wind shear surface-850 hPa",
        "lapse_850_500": "Lapse rate 850-500 hPa", "temp": "Temperature", "pressure": "Pressure",
        "hour_sin": "Time of day (sin)", "hour_cos": "Time of day (cos)", "doy_sin": "Season (sin)", "doy_cos": "Season (cos)",
    }
    names = [nice.get(i, i) for i in imp.index]
    sat = [i.startswith(("ctt", "ctp", "cold", "valid")) for i in imp.index]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(names, imp.values, color=["#1f77b4" if s else "#9aa5b1" for s in sat])
    ax.set_title("What the +60 min model relies on (blue = satellite)", fontsize=11)
    ax.set_xlabel("Share of model gain")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "importance.png", dpi=140)
    plt.close(fig)


def case_chart(oof: pd.DataFrame, day: str, lead: int = 60) -> None:
    d = oof[oof["day"] == day]
    t = pd.to_datetime(d.index).tz_convert("Asia/Kolkata") if d.index.tz is not None else pd.to_datetime(d.index)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    a1.plot(t, d["ctt_min_m"], color="#1f77b4", lw=1.8)
    a1.axhline(235, color="#1f77b4", ls=":", lw=1)
    a1.set_ylabel("Coldest cloud top (K)")
    a1.invert_yaxis()
    a1.set_title(f"{day}  ·  INSAT-3DS cloud tops and +{lead} min thunderstorm probability (out-of-fold)", fontsize=11)
    a2.plot(t, d[f"env+sat_{lead}"], color=COLORS["env+sat"], lw=1.8, label=LABELS["env+sat"])
    a2.plot(t, d[f"env+sat+obs_{lead}"], color=COLORS["env+sat+obs"], lw=1.4, alpha=0.9, label=LABELS["env+sat+obs"])
    a2.fill_between(t, 0, 1, where=d[f"y{lead}"].to_numpy() == 1, color="#f2c14e", alpha=0.35, step="mid",
                    label=f"Storm reported at airport within next {lead} min")
    a2.set_ylim(0, 1)
    a2.set_ylabel("Probability")
    a2.legend(fontsize=8, loc="upper left", ncol=1)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a2.set_xlabel("Local time (IST)")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"case_{day}.png", dpi=140)
    plt.close(fig)


def export_app_files(oof: pd.DataFrame) -> None:
    """Small files the web app reads: per-scene timeline and the airport thunderstorm reports."""
    keep = ["day", "ctt_min_m", "cold220_m", "ctp_min_m", "valid_frac", "cape", "y30", "y60", "y90"]
    keep += [f"{n}_{L}" for n in ("env", "env+sat", "env+sat+obs") for L in (30, 60, 90)]
    out = oof[keep].copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(3)
    out.index = pd.to_datetime(out.index, utc=True).strftime("%Y-%m-%dT%H:%MZ")
    out.index.name = "time"
    out.to_csv(OUT / "timeline.csv")
    metar = pd.read_csv(city_dir(get_city(OUT.name)) / "metar.csv")
    ts = metar[metar["thunder"]].copy()
    ts["time"] = pd.to_datetime(ts["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%MZ")
    ts[["time", "metar"]].to_csv(OUT / "ts_reports.csv", index=False)


def write_domain(city_key: str) -> None:
    """Footprint of the model: the 4x4 degree window, the INSAT pixels inside it and the airport."""
    from .nowcast_train import HALF_DEG, SAT_DIRS, Geometry

    city = get_city(city_key)
    for sat_dir in SAT_DIRS:
        pix = sat_dir / "pixels.npz"
        if not pix.exists():
            continue
        p = np.load(pix)
        geo = Geometry(p["lat"], p["lon"], city)
        if geo.n >= 40:
            lat, lon = p["lat"][geo.sel], p["lon"][geo.sel]
            break
    else:
        raise SystemExit(f"No satellite pixels cover {city.name}; run app.mosdac fetch first.")
    out = ARTIFACT_DIR / city.key
    out.mkdir(parents=True, exist_ok=True)
    (out / "domain.json").write_text(json.dumps({
        "city": city.key, "name": city.name, "station": city.station, "lat": city.lat, "lon": city.lon,
        "window": [city.lon - HALF_DEG, city.lat - HALF_DEG, city.lon + HALF_DEG, city.lat + HALF_DEG],
        "pixels": [[round(float(a), 2), round(float(b), 2)] for a, b in zip(lon, lat)],
        "pixel_spacing_km": 35,
    }))
    print(f"{city.name}: {geo.n} INSAT pixels in the window")


def main(city_key: str = DEFAULT_CITY) -> None:
    global OUT, CITY_NAME
    city = get_city(city_key)
    OUT, CITY_NAME = ARTIFACT_DIR / city.key, city.name
    m = json.loads((OUT / "metrics.json").read_text())
    skill_chart(m)
    importance_chart(m)
    oof = pd.read_csv(OUT / "oof_predictions.csv", index_col=0, parse_dates=True)
    oof["day"] = pd.to_datetime(oof["day"]).dt.strftime("%Y-%m-%d")
    reliability_chart(oof)
    export_app_files(oof)
    # the day with the most storm-window scenes: the clearest illustration
    day = oof.groupby("day")["y60"].sum().sort_values(ascending=False).index[0]
    case_chart(oof, day)
    (OUT / "case_day.txt").write_text(day)
    print(f"{city.name}: charts written to {OUT} | case day {day}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=DEFAULT_CITY, choices=sorted(CITIES))
    ap.add_argument("--domain-only", action="store_true", help="write only the map footprint (no trained results needed)")
    a = ap.parse_args()
    if a.domain_only:
        write_domain(a.city)
    else:
        write_domain(a.city)
        main(a.city)
