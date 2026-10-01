"""Sanity checks that do not touch the network and do not invent scores."""

from __future__ import annotations

import numpy as np

from .cases import simulate_case
from .features import available_feature_names, feature_frame
from .gridutil import advect


def check_advection() -> None:
    field = np.zeros((20, 20), dtype=np.float64)
    field[10, 5] = 1.0
    u = np.full_like(field, 10.0)
    v = np.zeros_like(field)
    out = advect(field, u, v, lead_s=900.0, dx_m=9000.0, dy_m=9000.0)
    if out[10, 6] <= out[10, 5] or abs(out[10, 6] - 1.0) > 1.0e-6:
        raise AssertionError("Eastward advection did not shift the value by one cell.")


def check_optional_features() -> None:
    case = simulate_case(7)
    full = available_feature_names(case)
    if "cape" not in full or "cin" not in full:
        raise AssertionError("Simulator case should expose CAPE and CIN.")
    del case.fields["cape"]
    reduced = available_feature_names(case)
    if "cape" in reduced or "cape_grad_mag" in reduced or "cape_tendency_1" in reduced:
        raise AssertionError("CAPE features were built after CAPE was removed.")
    if "cin" not in reduced or "reflectivity" not in reduced:
        raise AssertionError("Unrelated features should remain when CAPE is absent.")
    matrix, names = feature_frame(case, t=4, names=reduced)
    if matrix.shape[1] != len(names) or not np.isfinite(matrix).all():
        raise AssertionError("Feature matrix is malformed.")


def check_no_future_index() -> None:
    case = simulate_case(11)
    t = 4
    for name, array in list(case.fields.items()):
        case.fields[name] = array[: t + 1]
    matrix, names = feature_frame(case, t)
    if matrix.shape[0] != case.grid.n_cells or matrix.shape[1] != len(names):
        raise AssertionError("Unexpected feature-frame shape after truncating the future.")


def run() -> None:
    check_advection()
    check_optional_features()
    check_no_future_index()
    print("Checks passed.", flush=True)
