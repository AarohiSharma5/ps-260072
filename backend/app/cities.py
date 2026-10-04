"""Cities the nowcast is trained for. One point per city: its airport METAR station is the truth."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    key: str
    name: str
    station: str  # ICAO code of the airport whose METAR reports are the label
    lat: float
    lon: float
    start: str  # first day with satellite data used for training
    end: str
    season: str  # what the training period covers, for the UI


CITIES: dict[str, City] = {
    "chennai": City("chennai", "Chennai", "VOMM", 13.11, 80.25, "2025-04-01", "2025-12-31",
                    "April-December 2025"),
    "kolkata": City("kolkata", "Kolkata", "VECC", 22.65, 88.45, "2025-04-01", "2025-10-31",
                    "April-October 2025 (pre-monsoon Nor'westers and monsoon)"),
}

DEFAULT_CITY = "chennai"


def get_city(key: str) -> City:
    try:
        return CITIES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown city '{key}'. Known: {', '.join(CITIES)}") from exc
