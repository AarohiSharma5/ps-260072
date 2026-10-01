"""Case containers: deterministic simulator and the NPZ reader used for ERA5.

A case is a set of (time, lat, lon) arrays on one grid plus boolean label
arrays per hazard. Missing variables stay absent. Nothing is imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    CONVECTIVE_PRECIP_MM_H,
    DEMO_DATASETS,
    DEMO_DISCLAIMER,
    DOMAIN_NAME,
    DT_MIN,
    ERA5_DATASETS,
    ERA5_DISCLAIMER,
    SEVIR_DATASETS,
    SEVIR_DISCLAIMER,
    VIL_KG_M2,
    LIGHTNING_COUNT_MIN,
    N_TIMES,
    OBSERVATION_FIELDS,
    THUNDERSTORM_DBZ,
)
from .gridutil import GridSpec, default_grid, grid_from_bounds


@dataclass
class Case:
    case_id: str
    seed: int
    grid: GridSpec
    fields: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]  # hazard -> bool (time, lat, lon)
    label_definitions: dict[str, str]
    n_times: int
    dt_min: int
    source: str  # "DEMO" or "HISTORICAL"
    source_name: str
    domain_name: str
    datasets: tuple[dict, ...]
    disclaimer: str
    times: list[str] | None = None  # ISO UTC strings when the data has real time
    extras: dict = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return name in self.fields and self.fields[name] is not None

    @property
    def hazards(self) -> tuple[str, ...]:
        return tuple(self.labels.keys())


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _pulse(age_h: float, t0: float, t_peak: float, t1: float, peak: float) -> float:
    if age_h < t0 or age_h > t1:
        return 0.0
    if age_h <= t_peak:
        span = max(t_peak - t0, 1.0e-3)
        return float(peak * np.sin(0.5 * np.pi * (age_h - t0) / span))
    span = max(t1 - t_peak, 1.0e-3)
    return float(peak * np.sin(0.5 * np.pi * (t1 - age_h) / span))


def _box_smooth(field_: np.ndarray) -> np.ndarray:
    padded = np.pad(field_, 1, mode="edge")
    nlat, nlon = field_.shape
    acc = np.zeros_like(field_)
    for a in range(3):
        for b in range(3):
            acc += padded[a : a + nlat, b : b + nlon]
    return acc / 9.0


def simulate_case(seed: int, grid: GridSpec | None = None) -> Case:
    """One independent convective case from a seed. Still a simulator."""
    grid = grid or default_grid()
    rng = np.random.default_rng(seed)
    nlat, nlon = grid.n_lat, grid.n_lon
    rows = np.arange(nlat, dtype=np.float64)[:, None]
    cols = np.arange(nlon, dtype=np.float64)[None, :]

    phase = float(rng.uniform(0.0, np.pi))
    cx = 0.42 + 0.2 * np.cos(phase)
    cy = 0.48 + 0.15 * np.sin(phase)
    xx = (cols + 0.5) / nlon
    yy = (rows + 0.5) / nlat
    corridor = np.clip(np.exp(-0.5 * (((xx - cx) / 0.30) ** 2 + ((yy - cy) / 0.34) ** 2)), 0.0, 1.0)

    shear_env = np.clip(8.0 + 8.0 * (0.4 + 0.6 * yy) + rng.normal(0.0, 0.6, (nlat, nlon)), 4.0, 28.0)
    cape_env = np.clip(350.0 + 1900.0 * corridor + rng.normal(0.0, 40.0, (nlat, nlon)), 80.0, 3200.0)
    cin_env = np.clip(-280.0 + 250.0 * corridor + rng.normal(0.0, 8.0, (nlat, nlon)), -400.0, -5.0)
    rh_env = 48.0 + 32.0 * corridor
    temp_env = 32.5 - 1.6 * yy + rng.normal(0.0, 0.15, (nlat, nlon))
    pressure_env = 1007.0 - 1.4 * corridor + 0.6 * yy
    u_bg = float(rng.normal(7.0, 1.5))
    v_bg = float(rng.normal(2.5, 1.5))

    systems = _place_systems(rng, corridor, shear_env, nlat, nlon)
    shape = (N_TIMES, nlat, nlon)
    arrays = {name: np.zeros(shape, dtype=np.float64) for name in (
        "reflectivity", "ir_bt", "temperature", "rh", "pressure", "cape", "cin", "shear", "wind_u", "wind_v", "lightning_count"
    )}

    for ti in range(N_TIMES):
        age_h = ti * DT_MIN / 60.0
        diurnal = 180.0 * np.sin(np.pi * ti / (N_TIMES - 1))
        refl_t = np.zeros((nlat, nlon))
        cool = np.zeros((nlat, nlon))
        u_pert = np.zeros((nlat, nlon))
        v_pert = np.zeros((nlat, nlon))

        for system in systems:
            age_s = age_h * 3600.0
            rr = system["row"] + (system["v"] * age_s) / grid.dy_m
            cc = system["col"] + (system["u"] * age_s) / grid.dx_m
            dist_km = np.hypot((rows - rr) * grid.dy_m / 1000.0, (cols - cc) * grid.dx_m / 1000.0)
            intensity = _pulse(age_h, system["t0"], system["t_peak"], system["t1"], system["peak"])
            refl_t = np.maximum(refl_t, intensity * np.exp(-0.5 * (dist_km / system["radius"]) ** 2))
            if system["kind"] == "storm" and system["t0"] - 0.75 <= age_h < system["t0"]:
                frac = (age_h - (system["t0"] - 0.75)) / 0.75
                cool += 12.0 * frac * np.exp(-0.5 * (dist_km / (system["radius"] * 1.15)) ** 2)
            if intensity > 1.0:
                dist_m = np.maximum(dist_km * 1000.0, 250.0)
                inward_c = (cc - cols) * grid.dx_m / dist_m
                inward_r = (rr - rows) * grid.dy_m / dist_m
                sign = 1.0 if age_h <= system["t_peak"] else -0.65
                speed = sign * 5.0 * (intensity / system["peak"]) * np.exp(-0.5 * (dist_km / system["radius"]) ** 2)
                u_pert += speed * inward_c
                v_pert += speed * inward_r

        noise = rng.normal(0.0, 1.0, (nlat, nlon))
        refl_t = np.clip(refl_t + noise * (refl_t > 1.0), 0.0, 72.0)
        smooth_refl = _box_smooth(refl_t)
        ir_t = 292.0 - 1.15 * smooth_refl - 22.0 * _sigmoid((smooth_refl - 40.0) / 4.0) - cool
        ir_t = np.clip(ir_t + rng.normal(0.0, 0.5, (nlat, nlon)), 185.0, 310.0)

        storm_mask = _sigmoid((refl_t - 42.0) / 4.0)
        cape_t = np.clip(cape_env + diurnal - 450.0 * storm_mask, 50.0, 3600.0)
        cin_t = np.clip(cin_env - 50.0 * storm_mask, -450.0, -1.0)
        rh_t = np.clip(rh_env + 22.0 * _sigmoid((refl_t - 20.0) / 6.0), 20.0, 100.0)

        mu = np.zeros((nlat, nlon))
        for system in systems:
            if system["kind"] != "storm":
                continue
            age_s = age_h * 3600.0
            rr = system["row"] + (system["v"] * age_s) / grid.dy_m
            cc = system["col"] + (system["u"] * age_s) / grid.dx_m
            dist_km = np.hypot((rows - rr) * grid.dy_m / 1000.0, (cols - cc) * grid.dx_m / 1000.0)
            footprint = np.exp(-0.5 * (dist_km / system["radius"]) ** 2)
            mature = np.clip((refl_t - 36.0) / 10.0, 0.0, 1.0)
            cold = np.clip((245.0 - ir_t) / 18.0, 0.0, 1.2)
            instability = np.clip((cape_t - 700.0) / 900.0, 0.0, 1.4)
            mu += 7.5 * footprint * mature * cold * instability
        flashes = rng.poisson(np.clip(mu, 0.0, 24.0)).astype(np.float64)
        if rng.random() < 0.2:
            flashes[int(rng.integers(0, nlat)), int(rng.integers(0, nlon))] += 1.0

        arrays["reflectivity"][ti] = refl_t
        arrays["ir_bt"][ti] = ir_t
        arrays["temperature"][ti] = temp_env - 3.8 * storm_mask
        arrays["rh"][ti] = rh_t
        arrays["pressure"][ti] = pressure_env - 1.6 * storm_mask
        arrays["cape"][ti] = cape_t
        arrays["cin"][ti] = cin_t
        arrays["shear"][ti] = shear_env
        arrays["wind_u"][ti] = u_bg + u_pert
        arrays["wind_v"][ti] = v_bg + v_pert
        arrays["lightning_count"][ti] = flashes

    labels = {
        "thunderstorm": arrays["reflectivity"] >= THUNDERSTORM_DBZ,
        "lightning": arrays["lightning_count"] >= LIGHTNING_COUNT_MIN,
    }
    return Case(
        case_id=f"sim-{seed}",
        seed=seed,
        grid=grid,
        fields=arrays,
        labels=labels,
        label_definitions={
            "thunderstorm": f"Simulated reflectivity at the valid time is at least {THUNDERSTORM_DBZ:.0f} dBZ.",
            "lightning": f"Simulated lightning count at the valid time is at least {LIGHTNING_COUNT_MIN:.0f} flash in the {DT_MIN}-minute bin.",
            "status": "Prototype labels on synthetic fields. Not an official thunderstorm or lightning-warning definition.",
        },
        n_times=N_TIMES,
        dt_min=DT_MIN,
        source="DEMO",
        source_name="deterministic-simulator",
        domain_name=DOMAIN_NAME,
        datasets=DEMO_DATASETS,
        disclaimer=DEMO_DISCLAIMER,
    )


def _place_systems(rng, corridor, shear, nlat, nlon) -> list[dict]:
    systems: list[dict] = []
    favorable = np.argwhere(corridor > 0.62)
    if len(favorable) < 8:
        favorable = np.argwhere(corridor > 0.45)
    for _ in range(int(rng.integers(1, 3))):
        pick = favorable[int(rng.integers(0, len(favorable)))]
        row, col = int(pick[0]), int(pick[1])
        life_bonus = float(np.clip((float(shear[row, col]) - 12.0) / 12.0, 0.0, 0.7))
        t0 = float(rng.uniform(0.25, 1.15))
        t_peak = t0 + float(rng.uniform(0.55, 1.05))
        systems.append({
            "kind": "storm",
            "row": row + float(rng.uniform(-0.4, 0.4)),
            "col": col + float(rng.uniform(-0.4, 0.4)),
            "u": float(rng.normal(7.0, 2.0)),
            "v": float(rng.normal(2.5, 2.0)),
            "t0": t0,
            "t_peak": t_peak,
            "t1": t_peak + float(rng.uniform(0.75, 1.25)) + life_bonus,
            "peak": float(rng.uniform(50.0, 62.0)),
            "radius": float(rng.uniform(14.0, 24.0)),
        })
    for _ in range(int(rng.integers(1, 3))):
        t0 = float(rng.uniform(0.2, 2.2))
        systems.append({
            "kind": "shower",
            "row": float(rng.integers(1, nlat - 1)),
            "col": float(rng.integers(1, nlon - 1)),
            "u": float(rng.normal(6.0, 2.0)),
            "v": float(rng.normal(2.0, 2.0)),
            "t0": t0,
            "t_peak": t0 + float(rng.uniform(0.3, 0.6)),
            "t1": t0 + float(rng.uniform(0.7, 1.2)),
            "peak": float(rng.uniform(24.0, 36.0)),
            "radius": float(rng.uniform(9.0, 16.0)),
        })
    return systems


def load_npz_case(path: Path) -> Case:
    """Load one gridded case written by app.era5 (or any file with the same schema).

    Required: lat, lon (1-D cell centers, lat ascending), dt_min, and at least one
    label source: reflectivity, lightning_count, or convective_precip.
    Optional arrays that are absent stay absent.
    """
    data = np.load(path, allow_pickle=False)
    if "lat" not in data or "lon" not in data:
        raise ValueError(f"{path.name} must include 1-D lat and lon center arrays.")
    lat = np.asarray(data["lat"], dtype=np.float64)
    lon = np.asarray(data["lon"], dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(f"{path.name} lat and lon must be 1-D.")
    if lat.size > 1 and lat[1] < lat[0]:
        raise ValueError(f"{path.name} lat must ascend (south to north).")

    fields: dict[str, np.ndarray] = {}
    reference_shape = None
    for name in OBSERVATION_FIELDS:
        if name not in data:
            continue
        arr = np.asarray(data[name], dtype=np.float64)
        if arr.ndim != 3 or arr.shape[1] != lat.size or arr.shape[2] != lon.size:
            raise ValueError(f"{path.name} field {name} must have shape (time, lat, lon).")
        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(f"{path.name} field {name} shape {arr.shape} differs from {reference_shape}.")
        fields[name] = arr
    if reference_shape is None:
        raise ValueError(f"{path.name} has no recognised fields.")

    labels: dict[str, np.ndarray] = {}
    definitions: dict[str, str] = {}
    if "reflectivity" in fields:
        labels["thunderstorm"] = fields["reflectivity"] >= THUNDERSTORM_DBZ
        definitions["thunderstorm"] = f"Reflectivity at the valid time is at least {THUNDERSTORM_DBZ:.0f} dBZ."
    elif "vil" in fields:
        labels["thunderstorm"] = fields["vil"] >= VIL_KG_M2
        definitions["thunderstorm"] = (
            f"PROXY: SEVIR vertically integrated liquid at the valid time is at least {VIL_KG_M2:g} kg/m². "
            "This is not a reflectivity threshold and not an observed warning."
        )
    elif "convective_precip" in fields:
        labels["thunderstorm"] = fields["convective_precip"] >= CONVECTIVE_PRECIP_MM_H
        definitions["thunderstorm"] = (
            f"PROXY: ERA5 convective precipitation at the valid time is at least {CONVECTIVE_PRECIP_MM_H:g} mm/h. "
            "This is a reanalysis convection proxy, not an observed thunderstorm."
        )
    if "lightning_count" in fields:
        labels["lightning"] = fields["lightning_count"] >= LIGHTNING_COUNT_MIN
        definitions["lightning"] = f"Lightning count at the valid time is at least {LIGHTNING_COUNT_MIN:.0f}."
    if not labels:
        raise ValueError(f"{path.name} has no label source (reflectivity, vil, convective_precip, or lightning_count).")
    definitions["status"] = "Prototype labels. Not an official warning definition."

    dlat = float(np.median(np.diff(lat))) if lat.size > 1 else 0.25
    dlon = float(np.median(np.diff(lon))) if lon.size > 1 else 0.25
    derived = grid_from_bounds(
        float(lat[0] - 0.5 * dlat), float(lat[-1] + 0.5 * dlat),
        float(lon[0] - 0.5 * dlon), float(lon[-1] + 0.5 * dlon),
        int(lat.size), int(lon.size),
    )
    grid = GridSpec(
        lat_min=derived.lat_min, lat_max=derived.lat_max, lon_min=derived.lon_min, lon_max=derived.lon_max,
        n_lat=derived.n_lat, n_lon=derived.n_lon, lat=lat, lon=lon,
        dlat=abs(dlat), dlon=abs(dlon), dx_m=derived.dx_m, dy_m=derived.dy_m,
    )
    dt_min = int(data["dt_min"]) if "dt_min" in data else 60
    times = [str(value) for value in data["times"]] if "times" in data else None
    domain_name = str(data["domain_name"]) if "domain_name" in data else path.stem
    source_name = str(data["source_name"]) if "source_name" in data else path.name
    sevir = "vil" in fields and "reflectivity" not in fields and "convective_precip" not in fields
    return Case(
        case_id=path.stem,
        seed=-1,
        grid=grid,
        fields=fields,
        labels=labels,
        label_definitions=definitions,
        n_times=int(reference_shape[0]),
        dt_min=dt_min,
        source="SEVIR" if sevir else "HISTORICAL",
        source_name=source_name,
        domain_name=domain_name,
        datasets=SEVIR_DATASETS if sevir else ERA5_DATASETS,
        disclaimer=SEVIR_DISCLAIMER if sevir else ERA5_DISCLAIMER,
        times=times,
    )


def load_historical_cases(folder: Path) -> list[Case]:
    return [load_npz_case(path) for path in sorted(folder.glob("*.npz"))]
