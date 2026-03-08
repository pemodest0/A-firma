#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STEP_TIMEOUT_SEC="${STEP_TIMEOUT_SEC:-14400}"
RESUME="${RESUME:-1}"
CONFIG_PATH="${CONFIG_PATH:-config/profit_shadow_mode_target_800.json}"
CANONICAL_LOCK_PATH="${CANONICAL_LOCK_PATH:-$ROOT/results/ops/profit_shadow_target_800_attack/canonical_shadow_profile.json}"
REUSE_LATEST="${REUSE_LATEST:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"
CANONICAL_RUN_STATE_FILE="${CANONICAL_RUN_STATE_FILE:-$ROOT/results/ops/profit_shadow_target_800_attack/canonical_current_run_id.txt}"
RUN_STATE_FILE="${RUN_STATE_FILE:-$ROOT/results/ops/profit_shadow_target_800/current_run_id.txt}"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3" \
    "/usr/local/bin/python3" \
    "/opt/homebrew/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"
  do
    if [[ -n "$candidate" ]] && [[ -x "$candidate" ]] && "$candidate" -c "import numpy, pandas" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "[profit_shadow_target_800] no compatible python interpreter found" >&2
  exit 1
fi

mkdir -p "$ROOT/results/ops/profit_shadow_target_800/logs"

if [[ -z "${RUN_ID:-}" ]]; then
  if [[ -f "$RUN_STATE_FILE" ]]; then
    RUN_ID="$(tr -d '[:space:]' < "$RUN_STATE_FILE")"
  fi
  if [[ -z "${RUN_ID:-}" ]]; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
    printf '%s\n' "$RUN_ID" > "$RUN_STATE_FILE"
  fi
fi

cd "$ROOT"
if [[ -f "$CANONICAL_LOCK_PATH" ]]; then
  echo "[profit_shadow_target_800] repo=$ROOT canonical_lock=$CANONICAL_LOCK_PATH resume=$RESUME reuse_latest=$REUSE_LATEST python=$PYTHON_BIN"
  "$PYTHON_BIN" scripts/ops/run_profit_shadow_canonical.py \
    --lock-path "$CANONICAL_LOCK_PATH" \
    --run-state-file "$CANONICAL_RUN_STATE_FILE" \
    --resume "$RESUME" \
    --step-timeout-sec "$STEP_TIMEOUT_SEC" \
    --reuse-latest "$REUSE_LATEST"
else
  echo "[profit_shadow_target_800] repo=$ROOT run_id=$RUN_ID config=$CONFIG_PATH resume=$RESUME python=$PYTHON_BIN"
  "$PYTHON_BIN" scripts/ops/run_profit_shadow_suite.py \
    --config-path "$CONFIG_PATH" \
    --run-id "$RUN_ID" \
    --resume "$RESUME" \
    --step-timeout-sec "$STEP_TIMEOUT_SEC"
  SUMMARY_PATH="$ROOT/results/ops/profit_shadow_target_800/runs/$RUN_ID/summary.json"
  if [[ -f "$SUMMARY_PATH" ]]; then
    NEXT_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
    printf '%s\n' "$NEXT_RUN_ID" > "$RUN_STATE_FILE"
  fi
fi

echo "[profit_shadow_target_800] done run_id=$RUN_ID"
