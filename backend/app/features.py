"""Spatial and temporal features.

Each feature is computed only when its input arrays exist. A missing
satellite, radar, instability, or wind field drops the features that depend
on it. Nothing is imputed. Tendencies are per time step of the case.
"""

from __future__ import annotations

import numpy as np

from .cases import Case
from .config import FEATURE_REQUIRES, feature_meta, leads_for_step
from .gridutil import advect, convergence, estimate_motion, gradient_magnitude, neighbor_max, neighbor_sum, wind_direction_deg


_MOTION = ("motion_u", "motion_v", "motion_speed")


def available_feature_names(case: Case) -> list[str]:
    radar = case.has("reflectivity") or case.has("vil")
    names: list[str] = []
    for name, requires in FEATURE_REQUIRES.items():
        if name in _MOTION:
            if radar:
                names.append(name)
            continue
        if all(case.has(key) for key in requires):
            names.append(name)
    return names


def _wind_angles(u: np.ndarray, v: np.ndarray):
    speed = np.hypot(u, v)
    rad = np.deg2rad(wind_direction_deg(u, v))
    return speed, np.sin(rad), np.cos(rad)


def motion_at(case: Case, t: int) -> tuple[np.ndarray, np.ndarray] | None:
    if t < 1:
        return None
    if case.has("reflectivity"):
        field = case.fields["reflectivity"]
    elif case.has("vil"):
        field = case.fields["vil"]
    else:
        return None
    return estimate_motion(field[t - 1], field[t], case.grid.dx_m, case.grid.dy_m, case.dt_min * 60.0)


def feature_frame(case: Case, t: int, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Features at analysis time t using only frames at or before t."""
    if t < 2:
        raise ValueError("feature_frame needs two preceding frames.")
    names = names or available_feature_names(case)
    grid = case.grid
    dt_s = case.dt_min * 60.0
    columns: dict[str, np.ndarray] = {}
    f = case.fields

    def take(name: str, array: np.ndarray) -> None:
        if name in names:
            columns[name] = np.asarray(array, dtype=np.float64).reshape(-1)

    if case.has("reflectivity"):
        refl = f["reflectivity"]
        take("reflectivity", refl[t])
        take("refl_trend_1", refl[t] - refl[t - 1])
        take("refl_trend_2", refl[t] - refl[t - 2])
        take("refl_grad_mag", gradient_magnitude(refl[t], grid.dx_m, grid.dy_m))
        take("neighbor_refl_max", neighbor_max(refl[t]))
        if any(name in names for name in ("motion_u", "motion_v", "motion_speed", "upstream_refl_1")):
            motion = motion_at(case, t)
            if motion is not None:
                mu, mv = motion
                take("upstream_refl_1", advect(refl[t], mu, mv, dt_s, grid.dx_m, grid.dy_m))
                take("motion_u", mu)
                take("motion_v", mv)
                take("motion_speed", np.hypot(mu, mv))
    if case.has("vil"):
        vil = f["vil"]
        take("vil", vil[t])
        take("vil_trend_1", vil[t] - vil[t - 1])
        take("vil_trend_2", vil[t] - vil[t - 2])
        take("vil_grad_mag", gradient_magnitude(vil[t], grid.dx_m, grid.dy_m))
        take("neighbor_vil_max", neighbor_max(vil[t]))
        if any(name in names for name in ("motion_u", "motion_v", "motion_speed", "upstream_vil_1")):
            motion = motion_at(case, t)
            if motion is not None:
                mu, mv = motion
                take("upstream_vil_1", advect(vil[t], mu, mv, dt_s, grid.dx_m, grid.dy_m))
                if not case.has("reflectivity"):
                    take("motion_u", mu)
                    take("motion_v", mv)
                    take("motion_speed", np.hypot(mu, mv))
    if case.has("ir_bt"):
        ir = f["ir_bt"]
        take("ir_bt", ir[t])
        take("ctt_cooling_1", ir[t - 1] - ir[t])
        take("ctt_cooling_2", ir[t - 2] - ir[t])
        take("ir_grad_mag", gradient_magnitude(ir[t], grid.dx_m, grid.dy_m))
    if case.has("temperature"):
        take("temperature", f["temperature"][t])
        take("temp_tendency_1", f["temperature"][t] - f["temperature"][t - 1])
    if case.has("rh"):
        take("rh", f["rh"][t])
        take("rh_tendency_1", f["rh"][t] - f["rh"][t - 1])
    if case.has("pressure"):
        take("pressure", f["pressure"][t])
        take("pressure_tendency_1", f["pressure"][t] - f["pressure"][t - 1])
    if case.has("wind_u") and case.has("wind_v"):
        speed, wind_sin, wind_cos = _wind_angles(f["wind_u"][t], f["wind_v"][t])
        take("wind_speed", speed)
        take("wind_sin", wind_sin)
        take("wind_cos", wind_cos)
        take("convergence", convergence(f["wind_u"][t], f["wind_v"][t], grid.dx_m, grid.dy_m))
    if case.has("cape"):
        take("cape", f["cape"][t])
        take("cape_tendency_1", f["cape"][t] - f["cape"][t - 1])
        take("cape_grad_mag", gradient_magnitude(f["cape"][t], grid.dx_m, grid.dy_m))
    if case.has("cin"):
        take("cin", f["cin"][t])
    if case.has("shear"):
        take("shear", f["shear"][t])
    if case.has("k_index"):
        take("k_index", f["k_index"][t])
    if case.has("total_totals"):
        take("total_totals", f["total_totals"][t])
    if case.has("convective_precip"):
        cp = f["convective_precip"]
        take("convective_precip", cp[t])
        take("cp_trend_1", cp[t] - cp[t - 1])
        take("neighbor_cp_max", neighbor_max(cp[t]))
    if case.has("lightning_count"):
        flashes = f["lightning_count"]
        area = (grid.dx_m / 1000.0) * (grid.dy_m / 1000.0)
        take("lightning_count", flashes[t])
        take("lightning_density", flashes[t] / area)
        take("lightning_rate_change", flashes[t] - flashes[t - 1])
        take("neighbor_lightning", neighbor_sum(flashes[t]))

    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"{case.case_id} is missing inputs for: {', '.join(missing)}")
    return np.column_stack([columns[name] for name in names]), list(names)


def lead_steps(case: Case, lead_min: int) -> int:
    if lead_min % case.dt_min != 0:
        raise ValueError("Lead time must be a multiple of the case time step.")
    return lead_min // case.dt_min


def label_at(case: Case, hazard: str, t: int, lead_min: int) -> np.ndarray:
    future = t + lead_steps(case, lead_min)
    if future >= case.n_times:
        raise ValueError("Lead extends past the case.")
    return case.labels[hazard][future].astype(np.float64).reshape(-1)


def persistence_at(case: Case, hazard: str, t: int) -> np.ndarray:
    return case.labels[hazard][t].astype(np.float64).reshape(-1)


def extrapolation_available(case: Case) -> bool:
    return case.has("reflectivity") or case.has("vil")


def extrapolation_at(case: Case, hazard: str, t: int, lead_min: int) -> np.ndarray:
    """Advect the current event mask with reflectivity-estimated motion."""
    motion = motion_at(case, t)
    if motion is None:
        raise ValueError("Extrapolation needs reflectivity.")
    mu, mv = motion
    grid = case.grid
    mask = case.labels[hazard][t].astype(np.float64)
    return advect(mask, mu, mv, float(lead_min * 60.0), grid.dx_m, grid.dy_m).reshape(-1)


def iter_inits(case: Case) -> range:
    last = case.n_times - 1 - (max(leads_for_step(case.dt_min)) // case.dt_min)
    return range(2, last + 1) if last >= 2 else range(0)


def feature_catalog(names: list[str], dt_min: int) -> list[dict[str, str]]:
    meta = feature_meta(dt_min)
    return [{"id": name, **meta[name]} for name in names]
