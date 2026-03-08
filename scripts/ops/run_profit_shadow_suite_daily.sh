#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STEP_TIMEOUT_SEC="${STEP_TIMEOUT_SEC:-7200}"
CONFIG_PATH="${CONFIG_PATH:-config/profit_shadow_mode.json}"
PYTHON_BIN="${PYTHON_BIN:-}"

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
  echo "[profit_shadow] no compatible python interpreter found" >&2
  exit 1
fi

mkdir -p "$ROOT/results/ops/profit_shadow/logs"

cd "$ROOT"
echo "[profit_shadow] repo=$ROOT run_id=$RUN_ID config=$CONFIG_PATH python=$PYTHON_BIN"

"$PYTHON_BIN" scripts/ops/run_profit_shadow_suite.py \
  --config-path "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --step-timeout-sec "$STEP_TIMEOUT_SEC"

echo "[profit_shadow] done run_id=$RUN_ID"
