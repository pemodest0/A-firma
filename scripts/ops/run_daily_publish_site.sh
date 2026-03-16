#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${SEED:-23}"
MAX_ASSETS="${MAX_ASSETS:-80}"
WITH_HEAVY="${WITH_HEAVY:-0}"
PUBLISH_MODE="${PUBLISH_MODE:-deploy}"
SKIP_MASTER="${SKIP_MASTER:-}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY_OUTROOT="$ROOT/results/ops/agents/daily_publish"
PYTHON_BIN="${PYTHON_BIN:-}"
MASTER_CODE=0
DEPLOY_CODE=0
SMOKE_CODE=0
CURRENT_STEP="bootstrap"
LAST_COMPLETED_STEP=""

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3" \
    "/usr/local/bin/python3" \
    "/opt/homebrew/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"
  do
    if [[ -n "$candidate" ]] && [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "[daily_publish] python3 não encontrado" >&2
  exit 1
fi

if [[ -z "$SKIP_MASTER" ]]; then
  if [[ "$PUBLISH_MODE" == "local" ]]; then
    SKIP_MASTER="1"
  else
    SKIP_MASTER="0"
  fi
fi

write_publish_summary() {
  local status="$1"
  mkdir -p "$SUMMARY_OUTROOT/$RUN_ID"
  SUMMARY_OUTROOT_ENV="$SUMMARY_OUTROOT" \
  RUN_ID_ENV="$RUN_ID" \
  STATUS_ENV="$status" \
  MASTER_CODE_ENV="$MASTER_CODE" \
  DEPLOY_CODE_ENV="$DEPLOY_CODE" \
  SMOKE_CODE_ENV="$SMOKE_CODE" \
  CURRENT_STEP_ENV="$CURRENT_STEP" \
  LAST_COMPLETED_STEP_ENV="$LAST_COMPLETED_STEP" \
  PUBLISH_MODE_ENV="$PUBLISH_MODE" \
  SKIP_MASTER_ENV="$SKIP_MASTER" \
  "$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from scripts.ops.agent_guides import attach_agent_guide

root = Path(os.environ["SUMMARY_OUTROOT_ENV"])
run_id = os.environ["RUN_ID_ENV"]
payload = attach_agent_guide(
    {
        "status": os.environ["STATUS_ENV"],
        "run_id": run_id,
        "cycle_run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "master_code": int(os.environ["MASTER_CODE_ENV"]),
        "deploy_code": int(os.environ["DEPLOY_CODE_ENV"]),
        "smoke_code": int(os.environ["SMOKE_CODE_ENV"]),
        "publish_mode": os.environ["PUBLISH_MODE_ENV"],
        "skip_master": os.environ["SKIP_MASTER_ENV"] == "1",
        "failed_step": os.environ["CURRENT_STEP_ENV"] if os.environ["STATUS_ENV"] == "fail" else "",
        "last_completed_step": os.environ["LAST_COMPLETED_STEP_ENV"],
    },
    "daily-publish",
)
run_dir = root / run_id
run_dir.mkdir(parents=True, exist_ok=True)
for target in [run_dir / "summary.json", root / "latest_summary.json", root / "latest_publish.json"]:
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

on_error() {
  local code="$1"
  write_publish_summary "fail"
  exit "$code"
}

trap 'on_error $?' ERR

if [[ -d "$HOME/.nvm/versions/node" ]]; then
  while IFS= read -r node_bin; do
    export PATH="$node_bin:$PATH"
  done < <(find "$HOME/.nvm/versions/node" -maxdepth 3 -type d -name bin | sort -r)
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export ASSYNTRAX_CYCLE_RUN_ID="$RUN_ID"

cd "$ROOT"
echo "[daily_publish] repo=$ROOT run_id=$RUN_ID"

# Atualiza preços antes de recalcular qualquer leitura.
CURRENT_STEP="daily_ingestion"
"$PYTHON_BIN" scripts/ops/run_daily_ingestion_agent.py --priority-only --cycle-run-id "$RUN_ID"
LAST_COMPLETED_STEP="$CURRENT_STEP"
CURRENT_STEP="daily_backfill"
"$PYTHON_BIN" scripts/ops/run_daily_backfill_agent.py --cycle-run-id "$RUN_ID"
LAST_COMPLETED_STEP="$CURRENT_STEP"

# Coleta de acerto/erro diária para histórico do site.
CURRENT_STEP="prediction_truth"
"$PYTHON_BIN" scripts/ops/update_prediction_truth_daily.py --run-id "$RUN_ID"
LAST_COMPLETED_STEP="$CURRENT_STEP"
CURRENT_STEP="daily_master"
if [[ "$SKIP_MASTER" == "1" ]]; then
  echo "[daily_publish] skipping daily_master for this run"
elif [[ "$WITH_HEAVY" == "1" ]]; then
  "$PYTHON_BIN" scripts/ops/run_daily_master.py --seed "$SEED" --max-assets "$MAX_ASSETS" --run-id "$RUN_ID" --with-heavy || MASTER_CODE=$?
else
  "$PYTHON_BIN" scripts/ops/run_daily_master.py --seed "$SEED" --max-assets "$MAX_ASSETS" --run-id "$RUN_ID" || MASTER_CODE=$?
fi
LAST_COMPLETED_STEP="$CURRENT_STEP"

# Recalcula modos oficiais e vigilância com os preços recém-ingestados.
CURRENT_STEP="daily_operation"
"$PYTHON_BIN" scripts/ops/run_daily_operation_agent.py --cycle-run-id "$RUN_ID"
LAST_COMPLETED_STEP="$CURRENT_STEP"
CURRENT_STEP="daily_shadow_gods"
"$PYTHON_BIN" scripts/ops/run_daily_shadow_gods_agent.py --cycle-run-id "$RUN_ID"
LAST_COMPLETED_STEP="$CURRENT_STEP"
CURRENT_STEP="daily_shadow_gods_historical"
"$PYTHON_BIN" scripts/ops/run_daily_shadow_gods_historical_agent.py --cycle-run-id "$RUN_ID"
LAST_COMPLETED_STEP="$CURRENT_STEP"
CURRENT_STEP="daily_vigilance"
"$PYTHON_BIN" scripts/ops/run_daily_vigilance_agent.py --cycle-run-id "$RUN_ID"
LAST_COMPLETED_STEP="$CURRENT_STEP"
CURRENT_STEP="daily_data_quality"
"$PYTHON_BIN" scripts/ops/run_daily_data_quality_agent.py --cycle-run-id "$RUN_ID"
LAST_COMPLETED_STEP="$CURRENT_STEP"
CURRENT_STEP="site_snapshot"
"$PYTHON_BIN" scripts/ops/build_site_finance_snapshot.py --cycle-run-id "$RUN_ID"
LAST_COMPLETED_STEP="$CURRENT_STEP"

# Atualiza artefatos públicos do motor para o frontend no deploy.
CURRENT_STEP="sync_lab_corr"
bash scripts/sync_lab_corr_to_website.sh
LAST_COMPLETED_STEP="$CURRENT_STEP"

CURRENT_STEP="deploy"
cd "$ROOT/website-ui"
if [[ "$PUBLISH_MODE" == "local" ]]; then
  echo "[daily_publish] local mode: skipping external deploy"
else
  npx vercel --prod --yes || DEPLOY_CODE=$?
fi
cd "$ROOT"
LAST_COMPLETED_STEP="$CURRENT_STEP"
write_publish_summary "ok"
CURRENT_STEP="daily_smoke_test"
"$PYTHON_BIN" scripts/ops/run_daily_smoke_test_agent.py --cycle-run-id "$RUN_ID" || SMOKE_CODE=$?
LAST_COMPLETED_STEP="$CURRENT_STEP"

echo "[daily_publish] done run_id=$RUN_ID master_code=$MASTER_CODE"
if [[ "$DEPLOY_CODE" -ne 0 ]]; then
  write_publish_summary "fail"
  exit "$DEPLOY_CODE"
fi
if [[ "$SMOKE_CODE" -ne 0 ]]; then
  write_publish_summary "fail"
  exit "$SMOKE_CODE"
fi
write_publish_summary "ok"
exit 0
