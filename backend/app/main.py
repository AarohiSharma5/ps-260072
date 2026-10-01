"""HTTP API for the nowcast prototype."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .config import MODEL_VERSION
from .service import ENGINE, ArtifactsMissing, UnknownCase

app = FastAPI(
    title="AKASH Nowcast Prototype",
    version=MODEL_VERSION,
    summary="SIH 26072 thunderstorm and lightning nowcast prototype. Interactive data are simulated.",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load() -> None:
    try:
        ENGINE.load()
    except ArtifactsMissing as exc:
        print(f"WARNING: {exc}")


def _guard(func):
    try:
        return func()
    except ArtifactsMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UnknownCase as exc:
        raise HTTPException(status_code=404, detail=f"Unknown case or cell: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _default_case() -> str:
    return ENGINE.default_case_id()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model_loaded": ENGINE.ready, "model_version": MODEL_VERSION, "data_mode": ENGINE.data_mode}


@app.get("/api/status")
def status() -> dict:
    return _guard(ENGINE.status)


@app.get("/api/methodology")
def methodology() -> dict:
    return _guard(ENGINE.methodology)


@app.get("/api/metrics")
def metrics() -> dict:
    def _read():
        ENGINE.require()
        return ENGINE.metrics

    return _guard(_read)


@app.get("/api/features/importance")
def feature_importance(target: str = Query(default="thunderstorm_60")) -> dict:
    def _read():
        ENGINE.require()
        block = ENGINE.metrics["targets"].get(target)
        if block is None:
            known = sorted(ENGINE.metrics["targets"])
            raise ValueError(f"Unknown target '{target}'. Known targets: {', '.join(known)}.")
        return {
            "target": target,
            "data_mode": ENGINE.metrics["data_mode"],
            "trained_at": ENGINE.metrics["trained_at"],
            "importance": block["feature_importance"],
            "note": "Gain is the XGBoost split-gain share inside this lead's model. It is not a causal effect.",
        }

    return _guard(_read)


@app.get("/api/cases")
def cases() -> dict:
    def _read():
        return {"data_mode": ENGINE.data_mode, "cases": ENGINE.case_summaries()}

    return _guard(_read)


@app.get("/api/grid")
def grid(case_id: str = Query(default=None)) -> dict:
    return _guard(lambda: ENGINE.grid(case_id or _default_case()))


@app.get("/api/observations")
def observations(case_id: str = Query(default=None), offset_min: int = Query(default=0)) -> dict:
    return _guard(lambda: ENGINE.observations(case_id or _default_case(), offset_min))


@app.get("/api/predictions")
def predictions(case_id: str = Query(default=None)) -> dict:
    return _guard(lambda: ENGINE.predictions(case_id or _default_case()))


@app.get("/api/atmosphere")
def atmosphere(case_id: str = Query(default=None), grid_id: int = Query(default=0)) -> dict:
    return _guard(lambda: ENGINE.atmosphere(case_id or _default_case(), grid_id))


@app.get("/api/scenario")
def scenario(case_id: str = Query(default=None)) -> dict:
    return _guard(lambda: ENGINE.scenario(case_id or _default_case()))


@app.get("/api/cases/{case_id}/cells/{grid_id}/explanation")
def explanation(case_id: str, grid_id: int, hazard: str = Query(default="thunderstorm"), lead: int = Query(default=60)) -> dict:
    return _guard(lambda: ENGINE.explanation(case_id, grid_id, hazard, lead))
