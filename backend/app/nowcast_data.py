"""Observed thunderstorm labels and hourly environment for one city.

    python -m app.nowcast_data fetch --city chennai
    python -m app.nowcast_data fetch --city kolkata

Labels   The city's airport METAR/SPECI via the Iowa Environmental Mesonet.
         A report counts as thunderstorm when its *current weather* group holds
         TS (TSRA, +TSRA, TSGR ...) or VCTS (thunderstorm in the vicinity).
         Forecast trend groups (TEMPO / BECMG / NOSIG) and remarks are cut
         off first, so a forecast "TEMPO TSRA" is never counted as observed.
Weather  Open-Meteo historical-forecast API at the city point, hourly, UTC.
         Model data, not observations. Includes CAPE, CIN, lifted index and
         winds/temperature aloft (850 and 500 hPa) for shear.
"""

from __future__ import annotations

import argparse
import io
import re

import pandas as pd
import requests

from .cities import CITIES, DEFAULT_CITY, City, get_city
from .config import DATA_DIR

HOURLY = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure", "pressure_msl",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "precipitation",
    "cape", "lifted_index", "convective_inhibition",
    "temperature_850hPa", "temperature_500hPa", "wind_speed_850hPa", "wind_direction_850hPa",
    "wind_speed_500hPa", "wind_direction_500hPa", "relative_humidity_700hPa",
]

_CUT = re.compile(r"\b(TEMPO|BECMG|NOSIG|RMK|FM\d{4}|TL\d{4}|AT\d{4})\b")
_TS = re.compile(r"(^|\s)[-+]?(VC)?TS[A-Z]{0,6}(\s|$)")


def city_dir(city: City):
    return DATA_DIR / city.key


def observed_thunder(metar: str) -> bool:
    body = _CUT.split(metar, maxsplit=1)[0]
    # The regex only matches a token that *starts* with TS / VCTS (+ optional intensity),
    # so cloud groups such as FEW025CB or TCU never count.
    return bool(_TS.search(" " + body + " "))


def fetch_metar_range(station: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """METAR/SPECI reports for [start, end] (UTC), thunderstorm flag added."""
    url = (
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        f"?station={station}&data=metar&year1={start.year}&month1={start.month}&day1={start.day}"
        f"&year2={end.year}&month2={end.month}&day2={end.day}&tz=Etc/UTC&format=onlycomma"
        "&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4"
    )
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return pd.DataFrame(columns=["time", "metar", "thunder"])
    df["time"] = pd.to_datetime(df["valid"], utc=True)
    df["thunder"] = df["metar"].astype(str).map(observed_thunder)
    return df[["time", "metar", "thunder"]].sort_values("time").reset_index(drop=True)


def fetch_metar(city: City, start: str, end: str) -> pd.DataFrame:
    return fetch_metar_range(city.station, pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1))


def fetch_weather(city: City, start: str, end: str) -> pd.DataFrame:
    r = requests.get(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params={
            "latitude": city.lat, "longitude": city.lon, "start_date": start, "end_date": end,
            "hourly": ",".join(HOURLY), "timezone": "UTC",
        },
        timeout=180,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df.pop("time"), utc=True)
    return df


def cmd_fetch(args: argparse.Namespace) -> None:
    city = get_city(args.city)
    start, end = args.start or city.start, args.end or city.end
    out = city_dir(city)
    out.mkdir(parents=True, exist_ok=True)
    metar = fetch_metar(city, start, end)
    metar.to_csv(out / "metar.csv", index=False)
    print(f"{city.name} {city.station}: METAR reports {len(metar)}  thunderstorm reports {int(metar.thunder.sum())}")
    wx = fetch_weather(city, start, end)
    wx.to_csv(out / "weather_hourly.csv", index=False)
    print(f"weather rows {len(wx)}  columns {len(wx.columns) - 1}")
    nulls = wx.isna().mean().round(2)
    print("columns with missing values:", {k: v for k, v in nulls.items() if v > 0} or "none")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--city", default=DEFAULT_CITY, choices=sorted(CITIES))
    f.add_argument("--start")
    f.add_argument("--end")
    f.set_defaults(fn=cmd_fetch)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
