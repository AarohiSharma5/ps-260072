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

## What a judge can do

1. Open the map on the analysis of the selected case.
2. Switch fields: radar, brightness temperature, convective precipitation, humidity, CAPE, CIN, K index, Totals-Totals, wind, lightning. Unavailable layers are greyed with their status.
3. Choose thunderstorm or lightning probability.
4. Move the timeline or press Play.
5. Click a cell for the forecast, the environment, recent change, and the model's own feature contributions.
6. Turn on verification to compare the forecast with the dataset's later state.
7. Open Model for held-out precision, recall, F1, ROC-AUC, confusion matrix, reliability, and baselines.
8. Open Method for the pipeline, split, source citation and scaling notes.

## Layout

- `backend/app/era5.py` — CDS fetch + NetCDF → NPZ converter (the feed-adapter pattern)
- `backend/app/cases.py` — simulator and NPZ loader; labels are defined here per source
- `backend/app/features.py` — step-relative features, dropped when inputs are absent
- `backend/app/train.py` — per-hazard, per-lead XGBoost with chronological split
- `backend/app/service.py`, `main.py` — FastAPI
- `frontend/src` — MapLibre map, timeline, cell panel, analytics, method

## Endpoints

`/api/status` `/api/cases` `/api/grid` `/api/observations` `/api/predictions` `/api/atmosphere` `/api/metrics` `/api/features/importance` `/api/scenario` `/api/methodology` `/api/cases/{id}/cells/{grid}/explanation`

## Citation

Hersbach, H. et al. (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI 10.24381/cds.adbb2d47. Licence CC-BY.
# ps-260072
