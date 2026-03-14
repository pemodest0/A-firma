#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${SEED:-23}"
MAX_ASSETS="${MAX_ASSETS:-80}"
WITH_HEAVY="${WITH_HEAVY:-0}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
PYTHON_BIN="${PYTHON_BIN:-}"

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

if [[ -d "$HOME/.nvm/versions/node" ]]; then
  while IFS= read -r node_bin; do
    export PATH="$node_bin:$PATH"
  done < <(find "$HOME/.nvm/versions/node" -maxdepth 3 -type d -name bin | sort -r)
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

cd "$ROOT"
echo "[daily_publish] repo=$ROOT run_id=$RUN_ID"

# Atualiza preços antes de recalcular qualquer leitura.
"$PYTHON_BIN" scripts/ops/run_daily_ingestion_agent.py

# Coleta de acerto/erro diária para histórico do site.
"$PYTHON_BIN" scripts/ops/update_prediction_truth_daily.py --run-id "$RUN_ID"

MASTER_CODE=0
if [[ "$WITH_HEAVY" == "1" ]]; then
  "$PYTHON_BIN" scripts/ops/run_daily_master.py --seed "$SEED" --max-assets "$MAX_ASSETS" --run-id "$RUN_ID" --with-heavy || MASTER_CODE=$?
else
  "$PYTHON_BIN" scripts/ops/run_daily_master.py --seed "$SEED" --max-assets "$MAX_ASSETS" --run-id "$RUN_ID" || MASTER_CODE=$?
fi

# Recalcula modos oficiais e vigilância com os preços recém-ingestados.
"$PYTHON_BIN" scripts/ops/run_daily_operation_agent.py
"$PYTHON_BIN" scripts/ops/run_daily_vigilance_agent.py
"$PYTHON_BIN" scripts/ops/run_daily_data_quality_agent.py
"$PYTHON_BIN" scripts/ops/build_site_finance_snapshot.py

# Atualiza artefatos públicos do motor para o frontend no deploy.
bash scripts/sync_lab_corr_to_website.sh

cd "$ROOT/website-ui"
npx vercel --prod --yes

echo "[daily_publish] done run_id=$RUN_ID master_code=$MASTER_CODE"
exit 0
