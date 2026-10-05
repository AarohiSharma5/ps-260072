# AKASH: thunderstorm nowcast from INSAT-3DS

Smart India Hackathon 2026, problem statement 26072 (MoES / IMD): AI/ML nowcasting of thunderstorms.

AKASH turns ISRO's **INSAT-3DS cloud-top temperature and pressure** (from MOSDAC) and a weather-model environment into the probability that an airport **reports a thunderstorm within the next 30, 60 or 90 minutes**. The truth is what the airport actually reported (METAR `TS` / `VCTS`). It is a research prototype, not a warning product.

## What is in the app

| Page | What it shows |
|---|---|
| **Nowcast** (`/`) | City switch, a **Live now** card (needs MOSDAC credentials), the skill table for three input sets against a baseline, a replay of any 2025 day against what the airport reported, and what the model relies on |
| **Model** (`/model`) | Out-of-fold scores per lead time, hits and misses, a reliability diagram, a plain reading of the numbers, and a map of the INSAT-3DS pixels each model was trained on |
| **Method** (`/method`) | Pipeline, data sources, limitations, scaling notes, and the same training map |

Cities are defined in `backend/app/cities.py`. **Chennai** (airport VOMM, April to December 2025) is trained and served. **Kolkata** (VECC, April to October 2025) has its airport reports and weather data but is **not trained yet**: it needs its satellite days (see "Status" below). It appears in the app once its results exist.

## Results (Chennai, out-of-fold)

Scores come from days the model never saw: 5-fold cross-validation in 4-day blocks over 93 days, with model settings and the alert threshold chosen inside each training fold only. Storm rate per scene is 2.5 to 4%, so plain accuracy is misleading; use the metrics below.

| Lead | Inputs | ROC-AUC | PR-AUC | Storms caught | False alerts | Success index |
|---|---|---|---|---|---|---|
| +30 min | weather only | 0.83 | 0.09 | 27% | 90% | 0.08 |
| +30 min | + satellite | 0.91 | 0.20 | 59% | 80% | 0.18 |
| +30 min | + airport reports | 0.96 | 0.61 | 68% | 37% | 0.49 |
| +60 min | weather only | 0.84 | 0.12 | 47% | 87% | 0.11 |
| +60 min | + satellite | 0.90 | 0.24 | 59% | 78% | 0.20 |
| +60 min | + airport reports | 0.94 | 0.53 | 60% | 43% | 0.41 |
| +90 min | weather only | 0.84 | 0.14 | 47% | 86% | 0.12 |
| +90 min | + satellite | 0.88 | 0.23 | 63% | 79% | 0.19 |
| +90 min | + airport reports | 0.91 | 0.46 | 53% | 56% | 0.32 |

The "a storm was reported in the last hour" baseline scores a success index of 0.47, 0.43 and 0.38 at +30, +60 and +90 min. The satellite clearly helps over the weather model alone. Once a storm is already at the airport, the full model only ties that baseline at +30 and +60 min and trails it at +90; the satellite-only model is the one that can warn before a storm starts, at the price of many false alerts.

Model: XGBoost, one per lead time, 27 weather features, 28 satellite features and 4 airport-report features, with four small, heavily regularised settings (depth 2 to 4, 150 to 300 trees) chosen by inner cross-validation.

**Limitations.** One year, one airport per model, a few hundred storm cases per city (so real uncertainty), a 35 km satellite pixel, and a point observation as the truth. Radar and a lightning network are not used because no open archive was available. Live mode feeds the weather *forecast* as the environment, which is not identical to the historical forecasts used in training.

## Run locally

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ../frontend && npm install && npm run build      # the API serves frontend/dist
cd ../backend && python -m uvicorn app.main:app --port 8000
```

Open http://127.0.0.1:8000. For frontend development use `./run.sh` (API on 8000, Vite on 5173); macOS needs `brew install libomp` once for XGBoost.

## Build the nowcast yourself

```bash
cd backend && source .venv/bin/activate
cp tools/mosdac/config.example.json tools/mosdac/config.json     # add your MOSDAC login; git-ignored
export MOSDAC_USER=...  MOSDAC_PASS=...                          # or rely on config.json

# 1. satellite: one wide India window serves every city; resumes after the daily limit
python -m app.mosdac fetch --region india --start 2025-04-01 --end 2025-12-31 --step 2 --wait-on-quota
#    or run the three resumable passes: backend/tools/fetch_india.sh

# 2. airport reports (Iowa Environmental Mesonet) and weather (Open-Meteo)
python -m app.nowcast_data fetch --city chennai
python -m app.nowcast_data fetch --city kolkata

# 3. train and report (writes backend/artifacts/<city>/)
python -m app.nowcast_train --city chennai
python -m app.nowcast_report --city chennai

# map footprint only (needs just the satellite pixel grid)
python -m app.nowcast_report --city kolkata --domain-only
```

MOSDAC allows 5,000 file downloads per account per day and one 30-minute scene is one file (about 48 a day), so a full year takes several quota days. `app.mosdac fetch --wait-on-quota` sleeps and retries; keep the machine awake.

## Live mode

With `MOSDAC_USER` and `MOSDAC_PASS` set on the server, the Nowcast page shows a **Live now** card. The server downloads the newest INSAT-3DS scene and the two before it (for the cooling trend), reads the current Open-Meteo forecast and the airport's latest reports, builds a row with exactly the training feature code, and scores it with the saved models. Results are cached for five minutes and only new scenes are downloaded. If MOSDAC or a source fails, the last good forecast is shown marked as stale. Without credentials the card is hidden.

Live mode and the bulk download share one account's quota, and live mode needs about 50 files a day. Test it outside the web app with `python -m app.nowcast_live --city chennai`.

## Status

- Chennai: trained, served, live mode tested end to end against real MOSDAC data.
- Kolkata: reports and weather downloaded; satellite days missing (12 of the wide-window days are on disk), so no model yet.
- Not yet verified: a comparison of live satellite features against the training table for the same scene (it needs fresh MOSDAC quota). The training-side feature code was verified to reproduce the earlier table exactly.
- Docker image: builds (about 1.6 GB) and serves every page and API route in a container.

## Deploy

One container serves the API and the built web app on one port (`PORT`, default 8000).

```bash
docker build -t akash-nowcast .
docker run -p 8000:8000 akash-nowcast                       # http://localhost:8000
docker run -p 8000:8000 -e MOSDAC_USER=... -e MOSDAC_PASS=... akash-nowcast   # with live mode
```

On Fly.io, Render, Railway or Hugging Face Spaces (Docker), use the repository's `Dockerfile`, set the health check to `/api/health`, give it at least 512 MB of memory, and add `MOSDAC_USER` / `MOSDAC_PASS` as secrets in the host's dashboard. The image carries the trained models and per-city results; it carries no credentials and no raw downloads. `backend/requirements-serve.txt` is the runtime-only dependency list. Rebuild the image after retraining.

## API

`/api/health`, `/api/nowcast/cities`, `/api/nowcast/domains`, `/api/nowcast/{city}/summary`, `/api/nowcast/{city}/days`, `/api/nowcast/{city}/day?day=YYYY-MM-DD`, `/api/nowcast/{city}/live`. Interactive docs at `/docs`.

## Layout

- `backend/app/cities.py` — city definitions (airport, window, season)
- `backend/app/mosdac.py` — MOSDAC client (login, search, resumable download, quota waiting)
- `backend/app/nowcast_data.py` — airport reports and Open-Meteo weather per city
- `backend/app/nowcast_train.py` — feature builders shared with live mode, nested cross-validation, final models
- `backend/app/nowcast_report.py` — charts, the compact files the app reads, the map footprint
- `backend/app/nowcast_service.py`, `main.py` — read-only API and FastAPI app
- `backend/app/nowcast_live.py` — live forecast
- `backend/artifacts/<city>/` — metrics, models (`xgb_*.json`), `timeline.csv`, `ts_reports.csv`, `domain.json`
- `frontend/src` — Nowcast, Model and Method pages and the MapLibre training-footprint map

## Legacy grid engine

`backend/app/{service,train,cases,features,era5,arco,sevir}.py` hold an earlier grid-based prototype (a simulator, ERA5 and US SEVIR storm patches). Its interactive map was removed from the interface, but its endpoints still load at startup: `/api/status`, `/api/cases`, `/api/scenario`, `/api/metrics`, `/api/methodology` and related routes. Nothing in the current pages uses them.

## Data credit

INSAT-3DS data from MOSDAC, ISRO. Airport reports from the Iowa Environmental Mesonet. Environment fields from Open-Meteo (CC-BY 4.0). The legacy grid engine used ERA5: Hersbach, H. et al. (2023), ERA5 hourly data on single levels, Copernicus C3S CDS, DOI 10.24381/cds.adbb2d47, CC-BY.
