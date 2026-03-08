#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${1:-23}"
MAX_ASSETS="${2:-80}"
HEAVY="${3:-}"
THREADS="${4:-${ASSYNTRAX_THREADS:-1}}"
STEP_TIMEOUT_SEC="${5:-${ASSYNTRAX_STEP_TIMEOUT_SEC:-900}}"
ASSET_TIMEOUT_SEC="${6:-${ASSYNTRAX_ASSET_TIMEOUT_SEC:-180}}"
MAX_POINTS="${7:-${ASSYNTRAX_MAX_POINTS:-1200}}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

cd "$ROOT"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export LOKY_MAX_CPU_COUNT="$THREADS"

echo "[ops] repo=$ROOT run_id=$RUN_ID threads=$THREADS step_timeout_sec=$STEP_TIMEOUT_SEC asset_timeout_sec=$ASSET_TIMEOUT_SEC max_points=$MAX_POINTS"

if [[ "$HEAVY" == "heavy" ]]; then
  python3 scripts/ops/run_daily_master.py --seed "$SEED" --max-assets "$MAX_ASSETS" --run-id "$RUN_ID" --step-timeout-sec "$STEP_TIMEOUT_SEC" --asset-timeout-sec "$ASSET_TIMEOUT_SEC" --max-points "$MAX_POINTS" --with-heavy
else
  python3 scripts/ops/run_daily_master.py --seed "$SEED" --max-assets "$MAX_ASSETS" --run-id "$RUN_ID" --step-timeout-sec "$STEP_TIMEOUT_SEC" --asset-timeout-sec "$ASSET_TIMEOUT_SEC" --max-points "$MAX_POINTS"
fi

python3 scripts/ops/train_model_c_gnn.py
python3 scripts/ops/build_copilot_shadow.py --run-id "$RUN_ID"
python3 scripts/ops/build_platform_db.py --run-id "$RUN_ID"
if python3 scripts/ops/build_ai_operational_brief.py --run-dir "results/lab_corr_macro/$RUN_ID"; then
  echo "[ops] ai_operational_brief updated (run local)"
else
  echo "[ops] WARN ai_operational_brief by run_id failed; retry latest pointer" >&2
  if python3 scripts/ops/build_ai_operational_brief.py; then
    echo "[ops] ai_operational_brief updated (latest pointer fallback)"
  else
    echo "[ops] WARN failed to refresh ai_operational_brief" >&2
  fi
fi

echo "[ops] done"
