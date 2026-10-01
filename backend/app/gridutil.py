"""Grid geometry, sampling, gradients, and motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON


@dataclass(frozen=True)
class GridSpec:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n_lat: int
    n_lon: int
    lat: np.ndarray
    lon: np.ndarray
    dlat: float
    dlon: float
    dx_m: float
    dy_m: float

    @property
    def n_cells(self) -> int:
        return int(self.n_lat * self.n_lon)

    def cell_id(self, row: int, col: int) -> int:
        return row * self.n_lon + col


def default_grid() -> GridSpec:
    return grid_from_bounds(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, N_LAT, N_LON)


def grid_from_bounds(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_lat: int,
    n_lon: int,
) -> GridSpec:
    dlat = (lat_max - lat_min) / n_lat
    dlon = (lon_max - lon_min) / n_lon
    lat = lat_min + (np.arange(n_lat) + 0.5) * dlat
    lon = lon_min + (np.arange(n_lon) + 0.5) * dlon
    mean_lat = 0.5 * (lat_min + lat_max)
    dx_m = float(dlon * 111_320.0 * np.cos(np.deg2rad(mean_lat)))
    dy_m = float(dlat * 110_540.0)
    return GridSpec(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        n_lat=n_lat,
        n_lon=n_lon,
        lat=lat.astype(np.float64),
        lon=lon.astype(np.float64),
        dlat=float(dlat),
        dlon=float(dlon),
        dx_m=dx_m,
        dy_m=dy_m,
    )


def polygon(grid: GridSpec, row: int, col: int) -> list[list[float]]:
    south = grid.lat_min + row * grid.dlat
    north = south + grid.dlat
    west = grid.lon_min + col * grid.dlon
    east = west + grid.dlon
    ring = [
        [round(west, 5), round(south, 5)],
        [round(east, 5), round(south, 5)],
        [round(east, 5), round(north, 5)],
        [round(west, 5), round(north, 5)],
        [round(west, 5), round(south, 5)],
    ]
    return ring


def smooth(field: np.ndarray, passes: int = 1) -> np.ndarray:
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64)
    kernel /= kernel.sum()
    out = np.asarray(field, dtype=np.float64)
    nlat, nlon = out.shape
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        acc = np.zeros_like(out)
        for a in range(3):
            for b in range(3):
                acc += kernel[a, b] * padded[a : a + nlat, b : b + nlon]
        out = acc
    return out


def bilinear_sample(field: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sample a grid with edge clamping. rows/cols are in index coordinates."""
    nlat, nlon = field.shape
    r = np.clip(rows, 0.0, nlat - 1.0)
    c = np.clip(cols, 0.0, nlon - 1.0)
    r0 = np.floor(r).astype(np.int32)
    c0 = np.floor(c).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, nlat - 1)
    c1 = np.clip(c0 + 1, 0, nlon - 1)
    wr = r - r0
    wc = c - c0
    return (
        field[r0, c0] * (1.0 - wr) * (1.0 - wc)
        + field[r0, c1] * (1.0 - wr) * wc
        + field[r1, c0] * wr * (1.0 - wc)
        + field[r1, c1] * wr * wc
    )


def advect(field: np.ndarray, u: np.ndarray, v: np.ndarray, lead_s: float, dx_m: float, dy_m: float) -> np.ndarray:
    """Backward-advect a field. u is eastward m/s, v is northward m/s.

    The value written to a cell is the value that arrives if the field moves
    with (u, v) over lead_s. Sampling is bilinear with edge clamping.
    """
    nlat, nlon = field.shape
    rr, cc = np.meshgrid(np.arange(nlat, dtype=np.float64), np.arange(nlon, dtype=np.float64), indexing="ij")
    sample_r = rr - (v * lead_s) / dy_m
    sample_c = cc - (u * lead_s) / dx_m
    return bilinear_sample(field, sample_r, sample_c)


def gradient_magnitude(field: np.ndarray, dx_m: float, dy_m: float) -> np.ndarray:
    d_dy, d_dx = np.gradient(field, dy_m / 1000.0, dx_m / 1000.0)
    return np.hypot(d_dx, d_dy)


def convergence(u: np.ndarray, v: np.ndarray, dx_m: float, dy_m: float) -> np.ndarray:
    """Horizontal convergence in 10^-5 s^-1. Positive means converging."""
    _du_dy, du_dx = np.gradient(u, dy_m, dx_m)
    dv_dy, _dv_dx = np.gradient(v, dy_m, dx_m)
    return -(du_dx + dv_dy) * 1.0e5


def neighbor_max(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    nlat, nlon = field.shape
    stacks = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            stacks.append(padded[1 + di : 1 + di + nlat, 1 + dj : 1 + dj + nlon])
    return np.maximum.reduce(stacks)


def neighbor_sum(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, constant_values=0.0)
    nlat, nlon = field.shape
    total = np.zeros_like(field, dtype=np.float64)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            total += padded[1 + di : 1 + di + nlat, 1 + dj : 1 + dj + nlon]
    return total


def _centroid(field: np.ndarray, thresh: float) -> tuple[float, float] | None:
    weight = np.clip(field - thresh, 0.0, None)
    mass = float(weight.sum())
    if mass < 25.0:
        return None
    rr, cc = np.indices(field.shape)
    return float((weight * rr).sum() / mass), float((weight * cc).sum() / mass)


def _shift_to_uv(
    prev: np.ndarray,
    now: np.ndarray,
    dx_m: float,
    dy_m: float,
    dt_s: float,
    thresh: float,
) -> tuple[float, float, bool]:
    a = _centroid(prev, thresh)
    b = _centroid(now, thresh)
    if a is None or b is None:
        return 0.0, 0.0, False
    drow = b[0] - a[0]
    dcol = b[1] - a[1]
    v = (drow * dy_m) / dt_s
    u = (dcol * dx_m) / dt_s
    speed = float(np.hypot(u, v))
    if speed > 40.0:
        scale = 40.0 / speed
        u *= scale
        v *= scale
    return u, v, True


def _summed(array: np.ndarray) -> np.ndarray:
    table = np.zeros((array.shape[0] + 1, array.shape[1] + 1), dtype=np.float64)
    table[1:, 1:] = np.cumsum(np.cumsum(array, axis=0), axis=1)
    return table


def _window(table: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    return float(table[r1, c1] - table[r0, c1] - table[r1, c0] + table[r0, c0])


def estimate_motion(
    prev: np.ndarray,
    now: np.ndarray,
    dx_m: float,
    dy_m: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate eastward and northward motion from two reflectivity frames.

    Local 5×5 centroids are used where an echo exists. Elsewhere the domain
    centroid displacement is used. Motion is zero when no echo is present.
    """
    nlat, nlon = now.shape
    gu, gv, gok = _shift_to_uv(prev, now, dx_m, dy_m, dt_s, thresh=20.0)
    u = np.full((nlat, nlon), gu if gok else 0.0, dtype=np.float64)
    v = np.full((nlat, nlon), gv if gok else 0.0, dtype=np.float64)
    rad = 2
    rr, cc = np.indices((nlat, nlon))
    prev_w = np.clip(prev - 15.0, 0.0, None)
    now_w = np.clip(now - 15.0, 0.0, None)
    sat_p = _summed(prev_w)
    sat_pr = _summed(prev_w * rr)
    sat_pc = _summed(prev_w * cc)
    sat_n = _summed(now_w)
    sat_nr = _summed(now_w * rr)
    sat_nc = _summed(now_w * cc)
    active = np.argwhere((now >= 18.0) | (prev >= 18.0))
    for i, j in active:
        r0, r1 = max(0, int(i) - rad), min(nlat, int(i) + rad + 1)
        c0, c1 = max(0, int(j) - rad), min(nlon, int(j) + rad + 1)
        mass0 = _window(sat_p, r0, r1, c0, c1)
        mass1 = _window(sat_n, r0, r1, c0, c1)
        if mass0 < 25.0 or mass1 < 25.0:
            continue
        drow = _window(sat_nr, r0, r1, c0, c1) / mass1 - _window(sat_pr, r0, r1, c0, c1) / mass0
        dcol = _window(sat_nc, r0, r1, c0, c1) / mass1 - _window(sat_pc, r0, r1, c0, c1) / mass0
        vv = (drow * dy_m) / dt_s
        uu = (dcol * dx_m) / dt_s
        speed = float(np.hypot(uu, vv))
        if speed > 40.0:
            scale = 40.0 / speed
            uu *= scale
            vv *= scale
        u[i, j] = uu
        v[i, j] = vv
    return smooth(u, 1), smooth(v, 1)


def wind_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological direction the wind is from, degrees clockwise from north."""
    return np.mod(270.0 - np.rad2deg(np.arctan2(v, u)), 360.0)


def motion_toward_deg(u: float, v: float) -> float:
    """Direction a storm is moving toward, degrees clockwise from north."""
    return float(np.mod(np.rad2deg(np.arctan2(u, v)), 360.0))
