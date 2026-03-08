#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
REFRESH_PRICES="${REFRESH_PRICES:-1}"
REBUILD_HISTORICAL_REPLAY="${REBUILD_HISTORICAL_REPLAY:-0}"
STEP_TIMEOUT_SEC="${STEP_TIMEOUT_SEC:-2400}"

mkdir -p "$ROOT/results/ops/invest_shadow/logs"

cd "$ROOT"
echo "[investment_shadow] repo=$ROOT run_id=$RUN_ID refresh_prices=$REFRESH_PRICES"

python3 scripts/ops/run_investment_shadow.py \
  --run-id "$RUN_ID" \
  --refresh-prices "$REFRESH_PRICES" \
  --rebuild-historical-replay "$REBUILD_HISTORICAL_REPLAY" \
  --step-timeout-sec "$STEP_TIMEOUT_SEC"

echo "[investment_shadow] done run_id=$RUN_ID"
