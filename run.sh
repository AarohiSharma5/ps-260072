#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x backend/.venv/bin/python ]]; then
  python3 -m venv backend/.venv
fi
# shellcheck disable=SC1091
source backend/.venv/bin/activate
pip install -r backend/requirements.txt

if [[ ! -f backend/artifacts/metadata.json ]]; then
  (cd backend && python -m app.train)
fi
if ls backend/data/historical/*.npz >/dev/null 2>&1; then
  echo "HISTORICAL mode: ERA5 cases found in backend/data/historical"
else
  echo "DEMO mode: no ERA5 cases. To use ERA5, add ~/.cdsapirc and run: (cd backend && python -m app.era5 all && python -m app.train)"
fi

(cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000) &
BACK_PID=$!
trap 'kill $BACK_PID' EXIT

cd frontend
if [[ ! -d node_modules ]]; then
  npm install
fi
npm run dev -- --host 127.0.0.1 --port 5173
