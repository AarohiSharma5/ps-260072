#!/usr/bin/env bash
# Resumable INSAT-3DS download (wide India window). Waits and retries when MOSDAC's daily limit is hit.
# Finished days are skipped, so this is safe to stop and start again.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { "$PY" -u -m app.mosdac fetch --region india --workers 6 --wait-on-quota --wait-min 20 "$@"; }
run --start 2025-04-01 --end 2025-10-31 --step 2   # pass 1: every second day, April-October
run --start 2025-04-02 --end 2025-10-30 --step 2   # pass 2: the days in between
run --start 2025-11-01 --end 2025-12-31 --step 1   # pass 3: November-December (Chennai)
echo "ALL PASSES DONE"
