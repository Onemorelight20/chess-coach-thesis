#!/usr/bin/env bash
# Collect full-range (400-2800 Elo), ~2000-players-per-pool feature datasets for
# all four time-control formats, into data/.
#
# RESUMABLE: kill this anytime (Ctrl-C, or `pkill -f src.data_layer.collector`)
# and just re-run `bash scripts/collect_2k.sh` to continue. Finished pools are
# skipped (their data/dataset_<pool>_2k.parquet exists); an interrupted pool
# resumes from its data/dataset_<pool>_2k.parquet.partial checkpoint (flushed
# every 200 games), so you lose at most a few minutes of engine work.
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep the Mac awake for the whole run (idle + system sleep); re-exec self under
# caffeinate so this also covers resumes. Harmless if caffeinate is unavailable.
if [ -z "${_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
    exec env _CAFFEINATED=1 caffeinate -is "$0" "$@"
fi

export PYTHONUNBUFFERED=1   # live progress in the log for a long-running job

PY=.venv/bin/python
PGN=lichess_db_standard_rated_2026-01.pgn
mkdir -p data

for EVENT in blitz bullet rapid classical; do
    OUT="data/dataset_${EVENT}_2k.parquet"
    if [ -f "$OUT" ]; then
        echo ">> $EVENT already complete ($OUT) — skipping"
        continue
    fi
    echo ">> collecting $EVENT -> $OUT  ($(date '+%H:%M:%S'))"
    "$PY" -m src.data_layer.collector "$PGN" "$OUT" \
        --event "$EVENT" --stratified \
        --rating-min 400 --rating-max 2800 --bin-width 200 --per-bin 180 \
        --games-per-player 8 --max-scan 10000000
done
echo "ALL POOLS DONE ($(date '+%H:%M:%S'))"
