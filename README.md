# AKASH nowcast prototype

Smart India Hackathon problem statement 26072: short-range thunderstorm and lightning probability on a grid, one configurable region.

The same pipeline runs in two data modes. The interface shows the active mode on every screen.

| Mode | Source | Step | Leads | Hazards | Label |
|---|---|---|---|---|---|
| **DEMO** | deterministic simulator (radar, IR, lightning, environment all synthetic) | 15 min | +15 … +90 | thunderstorm, lightning | reflectivity ≥ 40 dBZ; ≥ 1 flash |
| **HISTORICAL** | [Copernicus ERA5 hourly single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) (T, Td, p, 10 m wind, CAPE, CIN, K index, Totals-Totals, convective precipitation) | 60 min | +1 h, +2 h, +3 h | thunderstorm only | **proxy**: ERA5 convective precipitation ≥ 1 mm/h |

ERA5 has no radar, no satellite brightness temperature, no lightning and no bulk shear. In HISTORICAL mode those layers show **UNAVAILABLE**, the lightning hazard is not trained, and the motion-extrapolation baseline is off. The thunderstorm label is a reanalysis convection proxy and is labelled as such throughout. Nothing in either mode is a live forecast.

## Run (DEMO mode, no credentials needed)

macOS once: `brew install libomp`

```bash
chmod +x run.sh
./run.sh
```

Open http://127.0.0.1:5173. API docs at http://127.0.0.1:8000/docs.

## Seeded ERA5 (no CDS token)

The running historical cases, when present, come from the public ARCO-ERA5 copy of the same single-level dataset (DOI 10.24381/cds.adbb2d47). IMD radar, INSAT-3D/3DS, GPM IMERG, LIS VHRMC, and ERA5 pressure-level shear are not in those files: the first four need an account or are not hourly observations, and the pressure-level store is one global 37-level chunk per hour.

```bash
cd backend && source .venv/bin/activate
python -m app.arco seed --start 2025-05-12 --end 2025-05-23
python -m app.train
```

## Switch to ERA5 from the CDS (HISTORICAL mode)

1. Create a free account at https://cds.climate.copernicus.eu, open your profile, copy the API token.
2. Accept the ERA5 licence once on the dataset page (Download tab → "Terms of use").
3. Write `~/.cdsapirc`:

   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your token>
   ```

4. Fetch, convert, retrain, restart:

   ```bash
   cd backend && source .venv/bin/activate
   python -m app.era5 all            # default: central India, 2025-04-01 … 2025-05-31
   python -m app.train
   python -m uvicorn app.main:app --port 8000
   ```

   Options: `--start/--end YYYY-MM-DD`, `--north/--south/--west/--east`, `--domain-name`. Requests are queued by the CDS; a two-month box like the default is usually a few minutes to an hour.

   To return to DEMO mode, delete `backend/data/historical/*.npz` and rerun `python -m app.train`.

Training splits ERA5 days chronologically (60 / 20 / 20). The interactive cases on the map are the held-out test days. If a lead has fewer than 10 positive training rows it is skipped and listed on the Method page rather than trained on nothing.

## Nowcast (home page): Chennai and Kolkata, with a live mode

The home page replays a cross-validated thunderstorm nowcast for an airport city, Chennai (VOMM, Oct-Dec monsoon storms) or Kolkata (VECC, Apr-Oct pre-monsoon storms). Inputs are INSAT-3DS cloud-top temperature and pressure from MOSDAC (product `3SIMG_L2B_CTP`, one scene every 30 min), Open-Meteo environment fields (CAPE, CIN, shear, humidity), and, for the third model, the airport's own recent reports. The label is a thunderstorm (`TS` or `VCTS`) reported at the airport in the next 30, 60 or 90 minutes, from the Iowa Environmental Mesonet METAR archive.

Scores are out-of-fold: 5-fold cross-validation in 4-day blocks, with the model settings and alert threshold chosen inside each training fold. One location per model, one year. Not a warning product.

```bash
cd backend && source .venv/bin/activate
cp tools/mosdac/config.example.json tools/mosdac/config.json     # then add your MOSDAC login; this file is git-ignored
export MOSDAC_USER=...  MOSDAC_PASS=...                          # or rely on config.json
python -m app.mosdac fetch --region india --start 2025-04-01 --end 2025-12-31 --wait-on-quota   # one wide window serves every city; resumes after the daily limit
python -m app.nowcast_data fetch --city chennai        # METAR labels + Open-Meteo environment
python -m app.nowcast_data fetch --city kolkata
python -m app.nowcast_train --city chennai             # cross-validated skill, final models, out-of-fold probabilities
python -m app.nowcast_report --city chennai            # charts + the compact files the web app reads
```

`backend/tools/fetch_india.sh` runs the whole download in three resumable passes (every second day first, then the gaps, then November-December).

**Live mode.** With `MOSDAC_USER` and `MOSDAC_PASS` set on the server, the home page shows a "Live now" card: the newest INSAT-3DS scene (and the two before it, for the cooling trend), the current Open-Meteo forecast, and the airport's latest reports go through the same feature code as training and the saved models. Results are cached for five minutes and only new scenes are downloaded. Without credentials the card is hidden. Try it without the web app: `python -m app.nowcast_live --city chennai`.

Endpoints: `/api/nowcast/cities`, `/api/nowcast/{city}/summary`, `/days`, `/day?day=YYYY-MM-DD`, `/live`.

## Deploy

One container serves the API and the built web app on one port (`PORT`, default 8000).

```bash
docker build -t akash-nowcast .
docker run -p 8000:8000 akash-nowcast        # http://localhost:8000
```

On Render, Railway or Hugging Face Spaces (Docker), point the service at this repository's `Dockerfile` and set the health check to `/api/health`. The image carries the trained models, the per-city results and nine held-out SEVIR patches; it carries no credentials and no raw downloads. Set `MOSDAC_USER` and `MOSDAC_PASS` as secrets in the host's dashboard to turn on live mode. `backend/requirements-serve.txt` is the runtime-only dependency list.

Data credit: INSAT-3DS data from MOSDAC, ISRO. Airport reports from the Iowa Environmental Mesonet. Environment fields from Open-Meteo (CC-BY 4.0).

## What a judge can do

1. Open Nowcast: pick Chennai or Kolkata, read the "Live now" card (when MOSDAC credentials are set), then replay any 2025 day against what the airport reported.
2. Pick +30, +60 or +90 min and compare weather-only, +satellite and +airport-report models against the "storm in the last hour" baseline.
3. Open Model for out-of-fold scores, hits and misses, a reliability diagram and an honest reading of the numbers, with a map of the INSAT-3DS pixels each model was trained on.
4. Open Method for the pipeline, data sources, limitations and scaling notes.

## Layout

- `backend/app/era5.py` — CDS fetch + NetCDF → NPZ converter (the feed-adapter pattern)
- `backend/app/cases.py` — simulator and NPZ loader; labels are defined here per source
- `backend/app/features.py` — step-relative features, dropped when inputs are absent
- `backend/app/train.py` — per-hazard, per-lead XGBoost with chronological split
- `backend/app/service.py`, `main.py` — FastAPI
- `backend/app/nowcast_*.py`, `mosdac.py`, `cities.py` — the India nowcast (data, training, report, service, live)
- `frontend/src` — Nowcast, Model and Method pages; MapLibre map of the training footprint

## Endpoints

`/api/status` `/api/cases` `/api/grid` `/api/observations` `/api/predictions` `/api/atmosphere` `/api/metrics` `/api/features/importance` `/api/scenario` `/api/methodology` `/api/cases/{id}/cells/{grid}/explanation`

## Citation

Hersbach, H. et al. (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI 10.24381/cds.adbb2d47. Licence CC-BY.
# ps-260072
