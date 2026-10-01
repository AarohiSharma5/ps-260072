"""In-memory model service. Interactive cases are held out of training in both modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb

from .cases import Case, load_npz_case, simulate_case
from .config import (
    ALL_HAZARDS,
    ARTIFACT_DIR,
    DEMO_SEEDS,
    HISTORICAL_DIR,
    SEVIR_DIR,
    MODEL_DIR,
    REFERENCE_PLACES,
    RISK_HIGH,
    RISK_SEVERE,
    RISK_THRESHOLD_STATUS,
    RISK_WATCH,
    feature_meta,
    leads_for_step,
    panel_leads_for_step,
    step_text,
)
from .features import extrapolation_available, extrapolation_at, feature_frame, label_at, motion_at, persistence_at
from .gridutil import motion_toward_deg, polygon, wind_direction_deg
from .metricslib import operating_metrics


class ArtifactsMissing(RuntimeError):
    pass


class UnknownCase(KeyError):
    pass


@dataclass
class Prepared:
    case: Case
    t: int
    names: list[str]
    matrix: np.ndarray
    probabilities: dict[str, np.ndarray]  # "hazard_lead" -> probs
    extrapolation: dict[str, np.ndarray]
    leads: tuple[int, ...]
    panel_leads: tuple[int, ...]
    hazards: tuple[str, ...]


def risk_level(probability: float) -> str:
    if probability >= RISK_SEVERE:
        return "SEVERE RISK"
    if probability >= RISK_HIGH:
        return "HIGH RISK"
    if probability >= RISK_WATCH:
        return "WATCH"
    return "NORMAL"


def _flat(array, digits: int = 3) -> list:
    out = []
    for value in np.asarray(array, dtype=np.float64).reshape(-1):
        out.append(None if not np.isfinite(value) else round(float(value), digits))
    return out


def clock_for(case: Case, index: int) -> str:
    if case.times and 0 <= index < len(case.times):
        stamp = case.times[index].replace("Z", "")
        try:
            return datetime.fromisoformat(stamp).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            return case.times[index]
    minutes = index * case.dt_min
    return f"T+{minutes // 60:02d}:{minutes % 60:02d}"


def choose_analysis_index(case: Case, leads: tuple[int, ...]) -> int:
    """Pick a time whose next leads still contain labelled events, with history behind it."""
    hazard = "thunderstorm" if "thunderstorm" in case.labels else case.hazards[0]
    mask = case.labels[hazard]
    max_step = max(leads) // case.dt_min
    best_t, best_score = 3, -1.0e9
    for t in range(3, case.n_times - max_step):
        future = sum(float(mask[t + lead // case.dt_min].sum()) for lead in leads)
        score = future * 5.0 + float(mask[t].sum())
        if future <= 0:
            score = -50.0 + float(mask[t].sum()) * 0.01
        if score > best_score:
            best_t, best_score = t, score
    return best_t


class Engine:
    def __init__(self) -> None:
        self.metadata: dict | None = None
        self.metrics: dict | None = None
        self.models: dict[str, xgb.Booster] = {}
        self._cases: dict[str, Case] = {}
        self._prepared: dict[str, Prepared] = {}

    @property
    def ready(self) -> bool:
        return self.metadata is not None and bool(self.models)

    def load(self) -> None:
        meta_path = ARTIFACT_DIR / "metadata.json"
        metrics_path = ARTIFACT_DIR / "metrics.json"
        self.metadata = self.metrics = None
        self.models, self._cases, self._prepared = {}, {}, {}
        if not meta_path.exists() or not metrics_path.exists():
            return
        self.metadata = json.loads(meta_path.read_text())
        self.metrics = json.loads(metrics_path.read_text())
        for key, info in self.metadata["models"].items():
            booster = xgb.Booster()
            booster.load_model(str(MODEL_DIR / info["file"]))
            self.models[key] = booster
        if self.metadata["data_mode"] in ("HISTORICAL", "SEVIR"):
            folder = SEVIR_DIR if self.metadata["data_mode"] == "SEVIR" else HISTORICAL_DIR
            for case_id in self.metadata["split"]["test_cases"]:
                path = folder / f"{case_id}.npz"
                if path.exists():
                    self._cases[case_id] = load_npz_case(path)
            if not self._cases:
                raise ArtifactsMissing("Model was trained on historical files that are no longer present. Re-run python -m app.era5 convert and python -m app.train.")
        else:
            for seed in DEMO_SEEDS:
                self._cases[f"sim-{seed}"] = simulate_case(seed)

    def require(self) -> None:
        if not self.ready:
            raise ArtifactsMissing("Model artifacts are missing. From the backend directory run: python -m app.train")

    @property
    def data_mode(self) -> str:
        return self.metadata["data_mode"] if self.metadata else "UNAVAILABLE"

    def default_case_id(self) -> str:
        self.require()
        return next(iter(self._cases))

    def case(self, case_id: str) -> Case:
        self.require()
        if case_id not in self._cases:
            raise UnknownCase(case_id)
        return self._cases[case_id]

    def prepare(self, case_id: str) -> Prepared:
        if case_id in self._prepared:
            return self._prepared[case_id]
        case = self.case(case_id)
        names = list(self.metadata["feature_names"])
        leads = tuple(self.metadata["leads_min"])
        hazards = tuple(h for h in self.metadata["hazards"] if any(k.startswith(h + "_") for k in self.models))
        t = choose_analysis_index(case, leads)
        matrix, _ = feature_frame(case, t, names)
        dmat = xgb.DMatrix(matrix, feature_names=names, missing=np.nan)
        probabilities = {key: np.asarray(model.predict(dmat), dtype=np.float64) for key, model in self.models.items()}
        extrapolation = {}
        if extrapolation_available(case):
            for hazard in hazards:
                for lead in leads:
                    extrapolation[f"{hazard}_{lead}"] = extrapolation_at(case, hazard, t, lead)
        prepared = Prepared(case, t, names, matrix, probabilities, extrapolation, leads, panel_leads_for_step(case.dt_min), hazards)
        self._prepared[case_id] = prepared
        return prepared

    # ----- summaries -------------------------------------------------------------

    def case_summaries(self) -> list[dict]:
        self.require()
        out = []
        for case_id in self._cases:
            prepared = self.prepare(case_id)
            item = self._summary(prepared)
            item["scores"] = self._case_scores(prepared)
            out.append(item)
        return out

    def _summary(self, p: Prepared) -> dict:
        case, t = p.case, p.t
        hazard = "thunderstorm" if "thunderstorm" in case.labels else case.hazards[0]
        mask = case.labels[hazard]
        now_cells = int(mask[t].sum())
        max_step = max(p.leads) // case.dt_min
        future_cells = int(mask[t + 1 : t + max_step + 1].sum())
        if case.source == "DEMO":
            refl = case.fields["reflectivity"]
            title = f"Simulated cell, peak {refl.max():.0f} dBZ"
            summary = (
                f"Synthetic case on the {case.domain_name}. {now_cells} cells meet the storm label at analysis, "
                f"{future_cells} cell-hours in the following {step_text(max(p.leads))}."
            )
        else:
            title = f"ERA5 {clock_for(case, t)[:10]}"
            summary = (
                f"ERA5 reanalysis day over {case.domain_name}. Analysis {clock_for(case, t)}. "
                f"{now_cells} cells at or above the convective-precipitation proxy now, "
                f"{future_cells} cell-steps in the following {step_text(max(p.leads))}. Held out of training."
            )
        return {
            "id": case.case_id,
            "title": title,
            "summary": summary,
            "data_mode": case.source,
            "in_training_set": False,
            "analysis_index": t,
            "clock": clock_for(case, t),
            "clock_kind": "utc" if case.times else "simulated",
            "step_min": case.dt_min,
        }

    def _case_scores(self, p: Prepared) -> dict:
        scores = {}
        for hazard in p.hazards:
            for lead in p.leads:
                key = f"{hazard}_{lead}"
                if key not in p.probabilities:
                    continue
                observed = label_at(p.case, hazard, p.t, lead)
                m = operating_metrics(observed, p.probabilities[key])
                scores[key] = {
                    "n": int(observed.size),
                    "mean_predicted": round(float(p.probabilities[key].mean()), 4),
                    "observed_fraction": round(float(observed.mean()), 4),
                    "brier": None if m["brier"] is None else round(m["brier"], 4),
                    "f1": round(m["f1"], 4),
                    "roc_auc": None if m["roc_auc"] is None else round(m["roc_auc"], 4),
                }
        return {
            "note": "Scores for this one case. Cells are correlated. Use the test-set metrics for the model comparison.",
            "targets": scores,
        }

    # ----- main payload ----------------------------------------------------------

    def scenario(self, case_id: str) -> dict:
        p = self.prepare(case_id)
        case, grid, t = p.case, p.case.grid, p.t
        f = case.fields
        meta = feature_meta(case.dt_min)

        def frame(name: str):
            return _flat(f[name][t]) if case.has(name) else None

        def series(name: str, idx: list[int]):
            return [_flat(f[name][i]) for i in idx] if case.has(name) else None

        def column(name: str):
            return _flat(p.matrix[:, p.names.index(name)]) if name in p.names else None

        history_idx = list(range(max(0, t - 3), t + 1))
        verify_offsets = [0, *p.leads]
        verify_idx = [t + off // case.dt_min for off in verify_offsets]

        prediction = {h: {str(l): _flat(p.probabilities[f"{h}_{l}"]) for l in p.leads if f"{h}_{l}" in p.probabilities} for h in p.hazards}
        extrapolation = (
            {h: {str(l): _flat(p.extrapolation[f"{h}_{l}"]) for l in p.leads} for h in p.hazards} if p.extrapolation else None
        )
        persistence = {h: _flat(persistence_at(case, h, t), 0) for h in p.hazards}

        wind = None
        if case.has("wind_u") and case.has("wind_v"):
            u, v = f["wind_u"][t], f["wind_v"][t]
            wind = {"speed": _flat(np.hypot(u, v)), "direction": _flat(wind_direction_deg(u, v)), "u": _flat(u), "v": _flat(v)}
        area_km2 = (grid.dx_m / 1000.0) * (grid.dy_m / 1000.0)
        motion = motion_at(case, t)

        cells = [
            {"grid_id": grid.cell_id(r, c), "row": r, "col": c,
             "latitude": round(float(grid.lat[r]), 5), "longitude": round(float(grid.lon[c]), 5),
             "polygon": polygon(grid, r, c)}
            for r in range(grid.n_lat) for c in range(grid.n_lon)
        ]
        places = [pl for pl in REFERENCE_PLACES if grid.lat_min <= pl["lat"] <= grid.lat_max and grid.lon_min <= pl["lon"] <= grid.lon_max]

        return {
            "data_mode": case.source,
            "disclaimer": case.disclaimer,
            "source_name": case.source_name,
            "model_version": self.metadata["model_version"],
            "model_trained_at": self.metadata["trained_at"],
            "product_generated_at": datetime.now(timezone.utc).isoformat(),
            "case": self._summary(p),
            "domain": self._domain(case),
            "step_min": case.dt_min,
            "step_text": step_text(case.dt_min),
            "hazards": list(p.hazards),
            "leads_min": list(p.leads),
            "panel_leads_min": list(p.panel_leads),
            "timeline_min": [0, *p.leads],
            "thresholds": {
                "watch": RISK_WATCH, "high": RISK_HIGH, "severe": RISK_SEVERE, "status": RISK_THRESHOLD_STATUS,
                "rule": "Summary risk is the prototype threshold applied to the highest available hazard probability at the panel leads. Not an official warning standard.",
            },
            "confidence_note": "Prototype confidence is the mean distance of the panel-lead probabilities from 0.5, scaled to 0–1. It is a prediction margin, not calibrated uncertainty.",
            "label_definitions": case.label_definitions,
            "grid": cells,
            "places": places,
            "analysis": {
                "offset_min": 0,
                "clock": clock_for(case, t),
                "reflectivity": frame("reflectivity"),
                "vil": frame("vil"),
                "ir_bt": frame("ir_bt"),
                "temperature": frame("temperature"),
                "rh": frame("rh"),
                "pressure": frame("pressure"),
                "wind_speed": wind["speed"] if wind else None,
                "wind_direction": wind["direction"] if wind else None,
                "wind_u": wind["u"] if wind else None,
                "wind_v": wind["v"] if wind else None,
                "cape": frame("cape"),
                "cin": frame("cin"),
                "shear": frame("shear"),
                "k_index": frame("k_index"),
                "total_totals": frame("total_totals"),
                "convective_precip": frame("convective_precip"),
                "lightning_count": frame("lightning_count"),
                "lightning_density": _flat(f["lightning_count"][t] / area_km2) if case.has("lightning_count") else None,
                "refl_trend": column("refl_trend_1"),
                "ctt_cooling": column("ctt_cooling_1"),
                "cp_trend": column("cp_trend_1"),
                "lightning_rate_change": column("lightning_rate_change"),
                "motion_u": _flat(motion[0]) if motion else None,
                "motion_v": _flat(motion[1]) if motion else None,
                "convergence": column("convergence"),
                "thunderstorm_indicator": _flat(case.labels["thunderstorm"][t], 0) if "thunderstorm" in case.labels else None,
                "lightning_indicator": _flat(case.labels["lightning"][t], 0) if "lightning" in case.labels else None,
            },
            "history": {
                "offsets_min": [(i - t) * case.dt_min for i in history_idx],
                "clocks": [clock_for(case, i) for i in history_idx],
                "reflectivity": series("reflectivity", history_idx),
                "vil": series("vil", history_idx),
                "ir_bt": series("ir_bt", history_idx),
                "lightning_count": series("lightning_count", history_idx),
                "convective_precip": series("convective_precip", history_idx),
                "cape": series("cape", history_idx),
            },
            "prediction": prediction,
            "extrapolation": extrapolation,
            "persistence": persistence,
            "verification": {
                "note": (
                    "Later frames of the same dataset, shown only as verification. They are not the forecast."
                    if case.source == "HISTORICAL"
                    else "Later frames of the same SEVIR patch, shown only as verification. They are not the forecast."
                    if case.source == "SEVIR"
                    else "Later frames of the simulator, shown only as verification. They are not observations and not the forecast."
                ),
                "offsets_min": verify_offsets,
                "clocks": [clock_for(case, i) for i in verify_idx],
                "reflectivity": series("reflectivity", verify_idx),
                "vil": series("vil", verify_idx),
                "lightning_count": series("lightning_count", verify_idx),
                "convective_precip": series("convective_precip", verify_idx),
                "thunderstorm": [_flat(case.labels["thunderstorm"][i], 0) for i in verify_idx] if "thunderstorm" in case.labels else None,
                "lightning_event": [_flat(case.labels["lightning"][i], 0) for i in verify_idx] if "lightning" in case.labels else None,
            },
            "case_scores": self._case_scores(p),
            "datasets": self._datasets(case),
            "feature_names": p.names,
            "feature_meta": {name: meta[name] for name in p.names},
            "layers": self._layers(case),
        }

    def _domain(self, case: Case) -> dict:
        g = case.grid
        return {
            "name": case.domain_name,
            "lat_min": g.lat_min, "lat_max": g.lat_max, "lon_min": g.lon_min, "lon_max": g.lon_max,
            "n_lat": g.n_lat, "n_lon": g.n_lon,
            "center_lat": round(float(g.lat.mean()), 4), "center_lon": round(float(g.lon.mean()), 4),
            "cell_km_lat": round(g.dy_m / 1000.0, 2), "cell_km_lon": round(g.dx_m / 1000.0, 2),
        }

    def _datasets(self, case: Case) -> list[dict]:
        rows = []
        for item in case.datasets:
            row = dict(item)
            if row["status"] == "DEMO":
                row["timestamp_label"] = "Simulator clock"
                row["freshness"] = "Not a live feed. Values exist only inside the synthetic case."
            elif row["status"] in ("HISTORICAL", "SEVIR"):
                row["timestamp_label"] = clock_for(case, case.n_times - 1)
                row["freshness"] = "Archive reanalysis. ERA5 is published about five days behind real time; this case is not live."
            else:
                row["timestamp_label"] = None
                row["freshness"] = "Not connected in this prototype."
            rows.append(row)
        return rows

    def _layers(self, case: Case) -> list[dict]:
        s1 = step_text(case.dt_min)
        specs = [
            ("reflectivity", "Radar reflectivity", "reflectivity", "dBZ"),
            ("vil", "Vertically integrated liquid", "vil", "kg/m²"),
            ("ir_bt", "Cloud-top brightness temperature", "ir_bt", "K"),
            ("lightning_density", "Lightning density", "lightning_count", f"flashes / km² / {s1}"),
            ("convective_precip", "Convective precipitation", "convective_precip", "mm/h"),
            ("temperature", "Temperature", "temperature", "°C"),
            ("rh", "Relative humidity", "rh", "%"),
            ("cape", "CAPE", "cape", "J/kg"),
            ("cin", "CIN", "cin", "J/kg"),
            ("shear", "Bulk shear", "shear", "m/s"),
            ("k_index", "K index", "k_index", "°C"),
            ("total_totals", "Totals-Totals", "total_totals", "°C"),
            ("wind", "Wind", "wind_u", "m/s"),
        ]
        return [
            {"id": lid, "label": label, "unit": unit, "available": case.has(field),
             "status": case.source if case.has(field) else "UNAVAILABLE"}
            for lid, label, field, unit in specs
        ]

    # ----- narrower endpoints -----------------------------------------------------

    def grid(self, case_id: str) -> dict:
        sc = self.scenario(case_id)
        return {"data_mode": sc["data_mode"], "domain": sc["domain"], "cells": sc["grid"]}

    def observations(self, case_id: str, offset_min: int) -> dict:
        p = self.prepare(case_id)
        case = p.case
        if offset_min not in (0, *p.leads):
            raise ValueError(f"offset_min must be one of {[0, *p.leads]}.")
        index = p.t + offset_min // case.dt_min
        role = "analysis" if offset_min == 0 else "verification"
        names = ("reflectivity", "ir_bt", "temperature", "rh", "pressure", "cape", "cin", "shear", "k_index", "total_totals", "convective_precip", "lightning_count")
        cells = []
        g = case.grid
        for r in range(g.n_lat):
            for c in range(g.n_lon):
                cell = {"grid_id": g.cell_id(r, c), "latitude": round(float(g.lat[r]), 5), "longitude": round(float(g.lon[c]), 5)}
                for n in names:
                    v = float(case.fields[n][index, r, c]) if case.has(n) else None
                    cell[n] = None if v is None or not np.isfinite(v) else round(v, 3)
                if case.has("wind_u") and case.has("wind_v"):
                    u, v = float(case.fields["wind_u"][index, r, c]), float(case.fields["wind_v"][index, r, c])
                    cell["wind_speed"] = round(float(np.hypot(u, v)), 3)
                    cell["wind_direction"] = round(float(wind_direction_deg(np.array(u), np.array(v))), 1)
                else:
                    cell["wind_speed"] = cell["wind_direction"] = None
                cells.append(cell)
        return {
            "data_mode": case.source, "role": role, "case_id": case_id, "offset_min": offset_min,
            "clock": clock_for(case, index),
            "note": "Analysis fields are what the model sees." if role == "analysis" else "Later state of the same dataset, for verification only. Not a forecast.",
            "cells": cells,
        }

    def predictions(self, case_id: str) -> dict:
        p = self.prepare(case_id)
        g = p.case.grid
        records = []
        for r in range(g.n_lat):
            for c in range(g.n_lon):
                cid = g.cell_id(r, c)
                values, core = {}, []
                for hazard in ALL_HAZARDS:
                    for lead in p.panel_leads:
                        key = f"{hazard}_{lead}"
                        if key in p.probabilities:
                            v = float(p.probabilities[key][cid])
                            values[f"{hazard}_probability_{lead}"] = round(v, 4)
                            core.append(v)
                        else:
                            values[f"{hazard}_probability_{lead}"] = None
                records.append({
                    "grid_id": cid, "latitude": round(float(g.lat[r]), 5), "longitude": round(float(g.lon[c]), 5),
                    **values,
                    "confidence": round(float(np.mean([abs(v - 0.5) * 2 for v in core])), 4) if core else None,
                    "risk_level": risk_level(max(core)) if core else "UNAVAILABLE",
                    "data_mode": p.case.source,
                })
        return {
            "data_mode": p.case.source, "disclaimer": p.case.disclaimer, "model_version": self.metadata["model_version"],
            "case_id": case_id, "clock": clock_for(p.case, p.t), "leads_min": list(p.panel_leads), "hazards": list(p.hazards),
            "risk_rule": "risk_level is the prototype threshold on the maximum available hazard probability across the listed leads. Null probabilities mean the hazard has no data in this mode.",
            "thresholds": {"watch": RISK_WATCH, "high": RISK_HIGH, "severe": RISK_SEVERE, "status": RISK_THRESHOLD_STATUS},
            "cells": records,
        }

    def atmosphere(self, case_id: str, grid_id: int) -> dict:
        sc = self.scenario(case_id)
        if grid_id < 0 or grid_id >= len(sc["grid"]):
            raise UnknownCase(f"grid {grid_id}")
        a, h = sc["analysis"], sc["history"]
        at = lambda n: (a.get(n) or [None] * (grid_id + 1))[grid_id]  # noqa: E731
        hist = lambda n: [fr[grid_id] for fr in h[n]] if h.get(n) else None  # noqa: E731
        u, v = at("motion_u"), at("motion_v")
        motion = None
        if u is not None and v is not None:
            motion = {"eastward_m_s": u, "northward_m_s": v, "speed_km_h": round(float(np.hypot(u, v)) * 3.6, 2),
                      "toward_deg": round(motion_toward_deg(u, v), 1), "note": "Estimated from the previous reflectivity frame. Not an official track."}
        return {
            "data_mode": sc["data_mode"], "grid_id": grid_id,
            "latitude": sc["grid"][grid_id]["latitude"], "longitude": sc["grid"][grid_id]["longitude"], "clock": a["clock"],
            "conditions": {
                "temperature_c": at("temperature"), "relative_humidity_percent": at("rh"), "pressure_hpa": at("pressure"),
                "wind_speed_m_s": at("wind_speed"), "wind_direction_deg": at("wind_direction"),
                "cape_j_kg": at("cape"), "cin_j_kg": at("cin"), "bulk_shear_m_s": at("shear"),
                "k_index_c": at("k_index"), "total_totals_c": at("total_totals"), "convective_precip_mm_h": at("convective_precip"),
            },
            "recent_evolution": {
                "offsets_min": h["offsets_min"], "reflectivity_dbz": hist("reflectivity"), "ir_bt_k": hist("ir_bt"),
                "lightning_count": hist("lightning_count"), "convective_precip_mm_h": hist("convective_precip"),
                "reflectivity_change": at("refl_trend"), "cloud_top_cooling": at("ctt_cooling"),
                "lightning_rate_change": at("lightning_rate_change"), "convective_precip_change": at("cp_trend"), "storm_movement": motion,
            },
        }

    def explanation(self, case_id: str, grid_id: int, hazard: str, lead: int) -> dict:
        p = self.prepare(case_id)
        key = f"{hazard}_{lead}"
        if key not in self.models:
            raise ValueError(f"No model for {key} in this data mode. Available: {sorted(self.models)}.")
        if grid_id < 0 or grid_id >= p.case.grid.n_cells:
            raise UnknownCase(f"grid {grid_id}")
        meta = feature_meta(p.case.dt_min)
        booster = self.models[key]
        row = p.matrix[grid_id : grid_id + 1]
        dmat = xgb.DMatrix(row, feature_names=p.names, missing=np.nan)
        probability = float(booster.predict(dmat)[0])
        contrib = np.asarray(booster.predict(dmat, pred_contribs=True)[0], dtype=np.float64)
        parts = []
        for i, name in enumerate(p.names):
            value = float(row[0, i])
            parts.append({"feature": name, "label": meta[name]["label"], "unit": meta[name]["unit"],
                          "value": None if not np.isfinite(value) else round(value, 3), "contribution": round(float(contrib[i]), 4)})
        positive = sorted((x for x in parts if x["contribution"] > 0), key=lambda x: x["contribution"], reverse=True)
        rank = {x["feature"]: i + 1 for i, x in enumerate(positive)}
        for x in parts:
            c = x["contribution"]
            x["statement"] = ("High contribution to model prediction" if c > 0 and rank[x["feature"]] <= 3
                              else "Increased the model score" if c > 0 else "Decreased the model score" if c < 0 else "No contribution to the model score")
        parts.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return {
            "data_mode": p.case.source, "grid_id": grid_id, "hazard": hazard, "lead_min": lead,
            "probability": round(probability, 4), "bias_log_odds": round(float(contrib[-1]), 4),
            "contributions_are": "XGBoost pred_contribs on the log-odds scale, summed with the base rate.",
            "disclaimer": "These contributions attribute the model score. They do not establish that the feature caused the event.",
            "features": parts[:8],
        }

    def status(self) -> dict:
        self.require()
        first = next(iter(self._cases.values()))
        return {
            "system": "AKASH nowcast prototype", "problem_statement": "SIH 26072",
            "model_version": self.metadata["model_version"], "model_trained_at": self.metadata["trained_at"],
            "data_mode": self.data_mode, "interactive_mode": self.data_mode, "disclaimer": first.disclaimer,
            "source_name": first.source_name, "domain": self.metadata["domain"], "step_min": first.dt_min,
            "leads_min": self.metadata["leads_min"], "hazards": self.metadata["hazards"],
            "skipped_targets": self.metadata.get("skipped_targets", []),
            "live_feeds_connected": False,
            "datasets": self._datasets(first), "model_ready": True,
            "period": {"first": clock_for(first, 0), "last": clock_for(list(self._cases.values())[-1], list(self._cases.values())[-1].n_times - 1)} if first.times else None,
        }

    def methodology(self) -> dict:
        self.require()
        meta = self.metadata
        split = meta["split"]
        mode = meta["data_mode"]
        first = next(iter(self._cases.values()))
        dt = meta["step_min"]
        if mode == "HISTORICAL":
            obs = (
                f"{first.source_name} Variables used: 2 m temperature and dewpoint, surface pressure, 10 m wind, CAPE, CIN, K index, Totals-Totals, and convective precipitation. "
                "IMD radar, INSAT-3D/3DS, GPM IMERG, and LIS were considered and left out: radar and satellite archives need an account or are not a matching hourly grid, IMERG needs a PPS login, and LIS VHRMC is a monthly climatology rather than an hourly lightning observation. "
                "Pressure-level winds were not downloaded, so bulk shear stays absent."
            )
        elif mode == "SEVIR":
            obs = (
                f"{first.source_name} Each case is one SEVIR storm patch: NEXRAD vertically integrated liquid and a lightning-flash grid, "
                "both coarsened from 1 km to 8 km. The Mumbai-analogue catalog only chooses which US storms to include. "
                "Infrared channels, ERA5, IMD radar, and INSAT are not in these cases."
            )
        else:
            obs = "A deployment would ingest radar reflectivity, satellite infrared temperature, a lightning network, and an analysis of the environment. None are connected. The map is driven by a deterministic simulator with the same array layout."
        return {
            "title": "How this prototype makes a nowcast", "data_mode": mode, "disclaimer": meta["disclaimer"], "source_name": meta.get("source_name"),
            "steps": [
                {"id": "observations", "title": "Observations", "body": obs},
                {"id": "preprocessing", "title": "Preprocessing", "body":
                    "Each source is stored as a time × latitude × longitude array on one grid. ERA5 is written to one NPZ per day: Kelvin to °C, dewpoint to relative humidity (Magnus), Pa to hPa, hourly convective precipitation from m to mm/h, latitude sorted south to north. Non-finite values stay missing and XGBoost treats them as missing."
                    if mode == "HISTORICAL" else
                    "Each SEVIR patch is 384×384 pixels at about 1 km. An 8×8 block mean makes an 8 km VIL cell, and an 8×8 block sum makes the lightning count. Latitude is ordered south to north. Cell centers are spaced linearly between the patch corners; the native grid is a Lambert projection."
                    if mode == "SEVIR" else
                    "Each source is stored as a time × latitude × longitude array on one grid. The simulator smooths infrared temperature so it is coarser than reflectivity. No radar QC chain is included."},
                {"id": "alignment", "title": "Spatial and temporal alignment", "body":
                    f"All fields sit on the {meta['domain']['n_lat']}×{meta['domain']['n_lon']} grid of {meta['domain']['name']} at a {step_text(dt)} step. A real multi-sensor system would resample each sensor onto that grid before this point."},
                {"id": "features", "title": "Feature engineering", "body":
                    "At an analysis time the pipeline builds one- and two-step tendencies, spatial gradients, neighborhood maxima, wind convergence, and, when reflectivity exists, storm motion and upstream reflectivity. Features whose inputs are missing are dropped, not imputed."},
                {"id": "model", "title": "ML model", "body":
                    "Each hazard and lead has its own XGBoost classifier (binary:logistic, histogram trees, depth 4, class weight capped at 25). Training stops early on the validation cases. Probabilities are raw logistic outputs, not a second calibration model. Targets with under 10 positive training rows are skipped and reported as such."},
                {"id": "probability", "title": "Probabilistic prediction", "body": " ".join(v for k, v in meta["labels"].items())},
                {"id": "nowcast", "title": "Spatial nowcast", "body":
                    "The model is applied independently to every grid cell. The map colors that probability at the selected lead."},
                {"id": "map", "title": "Interactive map", "body":
                    "The interface selects a lead, a base field, and overlays. Verification outlines are the dataset's own later label, drawn only when that layer is on."},
            ],
            "split": {
                "scheme": split["scheme"], "train_cases": len(split["train_cases"]), "val_cases": len(split["val_cases"]), "test_cases": len(split["test_cases"]),
                "train_rows": split["train_rows"], "val_rows": split["val_rows"], "test_rows": split["test_rows"],
                "first_train_case": split["train_cases"][0], "last_test_case": split["test_cases"][-1],
                "leakage_controls": [
                    "Cases (days in ERA5 mode) are split in chronological order. Test days come after every training day." if mode == "HISTORICAL" else "Cases are split in order. No cell from a test case is in training.",
                    "The interactive cases are the held-out test cases." if mode == "HISTORICAL" else "The three interactive demonstration seeds are not in the train, validation, or test lists.",
                    "Features at time t use frames t, t−1, and t−2 only.",
                    "Labels are taken from the frame at the lead time.",
                    "Early stopping uses the validation cases only. Test cases are scored after the models are fixed.",
                ],
            },
            "features": meta["feature_catalog"], "baselines": meta["baselines"], "thresholds": meta["risk_thresholds"],
            "hazards": meta["hazards"], "skipped_targets": meta.get("skipped_targets", []), "leads_min": meta["leads_min"], "step_min": dt,
            "scaling": [
                "A sector is one grid. A national mosaic is the same pipeline on adjacent sectors, then a join of the probability grids.",
                "Radar domains can keep their own grids. The feature builder only needs cell size, coordinates, and aligned arrays.",
                "The NPZ loader is the integration point. The ERA5 adapter (app/era5.py) shows the pattern: a feed adapter writes arrays, the model does not change.",
                "Adding real radar, INSAT brightness temperature, or lightning means writing those arrays into the same NPZ; the reflectivity-based motion features and the lightning hazard switch on automatically.",
                "What a deployment still has to add: radar calibration and blockage, satellite parallax, lightning-network detection efficiency, probability calibration on observed events, and a warning policy signed off by the forecasting authority.",
            ],
            "limitations": [
                "No live feed is connected. ERA5 is an archive reanalysis published about five days behind real time." if mode == "HISTORICAL" else "No live or archive meteorological feed is connected.",
                "The thunderstorm label in ERA5 mode is a convective-precipitation proxy from the reanalysis model, not an observed storm or lightning report." if mode == "HISTORICAL" else "Skill scores measure the simulator, not operational prediction over India.",
                "Grid cells within a day are spatially and temporally correlated, so test-row counts overstate the number of independent events.",
                "Hourly ERA5 supports +1 h, +2 h, +3 h leads only. Sub-hourly nowcasting needs radar or satellite at 5–15 minute cadence." if mode == "HISTORICAL" else "Simulator skill does not transfer to real radar.",
                "Risk colours use prototype probability cuts that have not been verified against an official standard.",
                "Local contributions explain the model. They are not a physical causal analysis.",
                "Bulk shear needs ERA5 pressure-level winds. That dataset was not seeded, so the shear feature is absent." if mode == "HISTORICAL" else "Bulk shear in the simulator is one magnitude field.",
                f"Source: {first.source_name}. Licence CC-BY; cite Hersbach et al. (2023), DOI 10.24381/cds.adbb2d47." if mode == "HISTORICAL" else "The simulator is documented in app/cases.py.",
            ],
        }


ENGINE = Engine()
