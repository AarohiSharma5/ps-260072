"""Train one XGBoost classifier per hazard and lead. Split is by case, in chronological order."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import xgboost as xgb

from .cases import Case, load_historical_cases, simulate_case
from .config import (
    ARTIFACT_DIR,
    DEMO_SEEDS,
    HISTORICAL_DIR,
    SEVIR_DIR,
    MODEL_DIR,
    MODEL_VERSION,
    RISK_HIGH,
    RISK_SEVERE,
    RISK_THRESHOLD_STATUS,
    RISK_WATCH,
    TEST_SEEDS,
    TRAIN_SEEDS,
    VAL_SEEDS,
    feature_meta,
    leads_for_step,
)
from .features import (
    available_feature_names,
    extrapolation_at,
    extrapolation_available,
    feature_catalog,
    feature_frame,
    iter_inits,
    label_at,
    persistence_at,
)
from .metricslib import operating_metrics, reliability_curve, support


def _split_cases(cases: list[Case]) -> tuple[list[Case], list[Case], list[Case]]:
    ordered = sorted(cases, key=lambda case: case.times[0] if case.times else case.case_id)
    n = len(ordered)
    n_test = max(1, n // 5)
    n_val = max(1, n // 5)
    n_train = n - n_val - n_test
    return ordered[:n_train], ordered[n_train : n_train + n_val], ordered[n_train + n_val :]


def load_model_cases() -> tuple[list[Case], list[Case], list[Case], str]:
    sevir = load_historical_cases(SEVIR_DIR) if SEVIR_DIR.exists() else []
    if len(sevir) >= 5:
        return (*_split_cases(sevir), "SEVIR")
    historical = load_historical_cases(HISTORICAL_DIR) if HISTORICAL_DIR.exists() else []
    if historical:
        if len(historical) < 5:
            raise SystemExit(
                f"{HISTORICAL_DIR} has {len(historical)} case(s). Need at least 5 daily files for a "
                "chronological train/validation/test split, or remove them to use the simulator."
            )
        return (*_split_cases(historical), "HISTORICAL")
    sim = lambda seeds: [simulate_case(seed) for seed in seeds]  # noqa: E731
    return sim(TRAIN_SEEDS), sim(VAL_SEEDS), sim(TEST_SEEDS), "DEMO"


def _stack_split(cases: list[Case], feature_names: list[str], hazards: tuple[str, ...], leads: tuple[int, ...]) -> dict:
    xs: list[np.ndarray] = []
    y = {h: {lead: [] for lead in leads} for h in hazards}
    p = {h: {lead: [] for lead in leads} for h in hazards}
    e = {h: {lead: [] for lead in leads} for h in hazards}
    use_extrap = all(extrapolation_available(case) for case in cases)
    for case in cases:
        for t in iter_inits(case):
            matrix, names = feature_frame(case, t, feature_names)
            if names != feature_names:
                raise RuntimeError(f"Feature order changed for {case.case_id}.")
            xs.append(matrix)
            for h in hazards:
                persist = persistence_at(case, h, t)
                for lead in leads:
                    y[h][lead].append(label_at(case, h, t, lead))
                    p[h][lead].append(persist)
                    if use_extrap:
                        e[h][lead].append(extrapolation_at(case, h, t, lead))
    if not xs:
        raise RuntimeError("No training rows. Cases are shorter than the longest lead.")
    cat = lambda d: {lead: np.concatenate(v) for lead, v in d.items()}  # noqa: E731
    return {
        "X": np.vstack(xs),
        "labels": {h: cat(y[h]) for h in hazards},
        "persistence": {h: cat(p[h]) for h in hazards},
        "extrapolation": {h: cat(e[h]) for h in hazards} if use_extrap else None,
    }


def _train_one(x_train, y_train, x_val, y_val, names) -> xgb.Booster:
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    if positives <= 0.0:
        raise RuntimeError("A training target has no positive events.")
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc" if len(np.unique(y_val)) > 1 else "logloss",
        "max_depth": 4,
        "eta": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 20,
        "lambda": 1.2,
        "tree_method": "hist",
        "scale_pos_weight": float(min(negatives / positives, 25.0)),
        "seed": 42,
        "nthread": 4,
    }
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=names, missing=np.nan)
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=names, missing=np.nan)
    booster = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtrain, "train"), (dval, "val")],
                        early_stopping_rounds=20, verbose_eval=False)
    return booster[: int(booster.best_iteration) + 1]


def _importance(booster: xgb.Booster, names: list[str], meta: dict) -> list[dict]:
    score = booster.get_score(importance_type="gain")
    total = float(sum(score.values())) or 1.0
    rows = [{"feature": n, **meta[n], "gain": float(score.get(n, 0.0)), "gain_share": float(score.get(n, 0.0)) / total} for n in names]
    rows.sort(key=lambda item: item["gain_share"], reverse=True)
    return rows


def _round(payload):
    if isinstance(payload, float):
        return None if np.isnan(payload) else round(payload, 5)
    if isinstance(payload, dict):
        return {k: _round(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_round(v) for v in payload]
    return payload


def train() -> dict:
    from . import checks

    checks.run()
    print("Loading cases and building features…", flush=True)
    train_cases, val_cases, test_cases, source_mode = load_model_cases()
    first = train_cases[0]
    dt_min = first.dt_min
    leads = leads_for_step(dt_min)
    hazards = first.hazards
    feature_names = available_feature_names(first)
    for case in (*train_cases, *val_cases, *test_cases):
        if case.dt_min != dt_min:
            raise SystemExit(f"{case.case_id} has a {case.dt_min}-minute step; the first case has {dt_min}.")
        if case.hazards != hazards:
            raise SystemExit(f"{case.case_id} has hazards {case.hazards}; expected {hazards}.")
        names = available_feature_names(case)
        if names != feature_names:
            raise SystemExit(
                f"{case.case_id} has a different feature set: {sorted(set(feature_names) ^ set(names))}. "
                "This prototype trains one feature set; it does not impute."
            )
    meta = feature_meta(dt_min)
    train_tbl = _stack_split(train_cases, feature_names, hazards, leads)
    val_tbl = _stack_split(val_cases, feature_names, hazards, leads)
    test_tbl = _stack_split(test_cases, feature_names, hazards, leads)
    print(
        f"Rows  train={len(train_tbl['X'])}  val={len(val_tbl['X'])}  test={len(test_tbl['X'])}  "
        f"features={len(feature_names)}  hazards={hazards}  leads={leads}  step={dt_min} min  source={source_mode}",
        flush=True,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict = {
        "model_version": MODEL_VERSION,
        "data_mode": source_mode,
        "step_min": dt_min,
        "leads_min": list(leads),
        "hazards": list(hazards),
        "disclaimer": first.disclaimer,
        "label_definitions": first.label_definitions,
        "threshold_note": "Precision, recall, F1, and the confusion matrix use probability threshold 0.5 for the model and both baselines.",
        "independence_note": (
            "Test rows are grid cells from held-out cases. Neighboring cells and successive times are correlated, "
            "so the row count is not the number of independent storms."
        ),
        "extrapolation_available": test_tbl["extrapolation"] is not None,
        "targets": {},
    }
    model_index = {}
    skipped = []
    for hazard in hazards:
        for lead in leads:
            key = f"{hazard}_{lead}"
            y_train = train_tbl["labels"][hazard][lead]
            y_val = val_tbl["labels"][hazard][lead]
            y_test = test_tbl["labels"][hazard][lead]
            if y_train.sum() < 10:
                print(f"Skipping {key}: only {int(y_train.sum())} positive training rows.", flush=True)
                skipped.append(key)
                continue
            print(f"Training {key}…", flush=True)
            booster = _train_one(train_tbl["X"], y_train, val_tbl["X"], y_val, feature_names)
            path = MODEL_DIR / f"{key}.json"
            booster.save_model(path)
            predicted = booster.predict(xgb.DMatrix(test_tbl["X"], feature_names=feature_names, missing=np.nan))
            block = {
                "hazard": hazard,
                "lead_min": lead,
                "n_trees": int(booster.num_boosted_rounds()),
                "support": support(y_test),
                "model": operating_metrics(y_test, predicted),
                "persistence": operating_metrics(y_test, test_tbl["persistence"][hazard][lead]),
                "extrapolation": operating_metrics(y_test, test_tbl["extrapolation"][hazard][lead]) if test_tbl["extrapolation"] else None,
                "reliability": reliability_curve(y_test, predicted),
                "feature_importance": _importance(booster, feature_names, meta),
            }
            metrics["targets"][key] = _round(block)
            m = block["model"]
            print(
                f"  trees={block['n_trees']}  positives={block['support']['positives']}/{block['support']['n']}  "
                f"F1={m['f1']:.3f}  AUC={m['roc_auc']}  persist_F1={block['persistence']['f1']:.3f}",
                flush=True,
            )
            model_index[key] = {"file": path.name, "n_trees": block["n_trees"]}
    if not model_index:
        raise SystemExit("No target had enough positive events to train. Lower the proxy threshold or choose a stormier period.")

    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "data_mode": source_mode,
        "step_min": dt_min,
        "leads_min": list(leads),
        "hazards": list(hazards),
        "skipped_targets": skipped,
        "disclaimer": first.disclaimer,
        "feature_names": feature_names,
        "feature_catalog": feature_catalog(feature_names, dt_min),
        "models": model_index,
        "split": {
            "scheme": "chronological case-blocked split" if source_mode == "HISTORICAL" else "case-order blocked split",
            "train_cases": [c.case_id for c in train_cases],
            "val_cases": [c.case_id for c in val_cases],
            "test_cases": [c.case_id for c in test_cases],
            "train_rows": int(len(train_tbl["X"])),
            "val_rows": int(len(val_tbl["X"])),
            "test_rows": int(len(test_tbl["X"])),
            "demo_seeds_excluded": list(DEMO_SEEDS) if source_mode == "DEMO" else [],
        },
        "domain": {
            "name": first.domain_name,
            "lat_min": first.grid.lat_min, "lat_max": first.grid.lat_max,
            "lon_min": first.grid.lon_min, "lon_max": first.grid.lon_max,
            "n_lat": first.grid.n_lat, "n_lon": first.grid.n_lon,
        },
        "labels": first.label_definitions,
        "source_name": first.source_name,
        "risk_thresholds": {"watch": RISK_WATCH, "high": RISK_HIGH, "severe": RISK_SEVERE, "status": RISK_THRESHOLD_STATUS},
        "baselines": {
            "persistence": "The analysis-time event is repeated at every lead. It does not move or grow.",
            "extrapolation": (
                "The analysis-time event mask is advected with motion estimated from the previous reflectivity frame."
                if test_tbl["extrapolation"] is not None
                else "Unavailable: motion extrapolation needs radar reflectivity, which this dataset does not have."
            ),
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps({"trained_at": trained_at, **metrics}, indent=2))
    print(f"Wrote artifacts to {ARTIFACT_DIR}", flush=True)
    return metadata


if __name__ == "__main__":
    train()
