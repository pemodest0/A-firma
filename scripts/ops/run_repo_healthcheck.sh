#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTDIR="$ROOT/results/ops/healthcheck/$RUN_ID"
LOG="$OUTDIR/healthcheck.log"
CHECKS_TSV="$OUTDIR/checks.tsv"
REPORT="$OUTDIR/report.md"

mkdir -p "$OUTDIR"
: > "$LOG"
: > "$CHECKS_TSV"

FAIL_COUNT=0

run_check() {
  local name="$1"
  shift
  local cmd="$*"
  echo "[$(date -u +%H:%M:%S)] START $name :: $cmd" | tee -a "$LOG"
  if (cd "$ROOT" && eval "$cmd") >>"$LOG" 2>&1; then
    echo -e "${name}\tok\t${cmd}" >>"$CHECKS_TSV"
    echo "[$(date -u +%H:%M:%S)] OK    $name" | tee -a "$LOG"
  else
    echo -e "${name}\tfail\t${cmd}" >>"$CHECKS_TSV"
    echo "[$(date -u +%H:%M:%S)] FAIL  $name" | tee -a "$LOG"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

cd "$ROOT"
git status --short >"$OUTDIR/git_status.txt" || true
git rev-parse --short HEAD >"$OUTDIR/git_head.txt" || true

run_check "python_compileall" "python3 -m compileall -q engine scripts tools tests"
run_check "engine_purity_audit" "python3 tools/engine_purity_audit.py"
run_check "anti_leakage_guard" "python3 tools/anti_leakage_guard.py"
run_check "frontend_payload_audit" "python3 scripts/ops/audit_frontend_payloads.py"
run_check "daily_master_dry_run" "python3 scripts/ops/run_daily_master.py --dry-run --run-id HEALTH_${RUN_ID}"
run_check "frontend_lint" "cd website-ui && npm run lint"
run_check "frontend_clean_next_types" "rm -rf website-ui/.next/types"
run_check "frontend_typecheck" "cd website-ui && npm run typecheck"
run_check "frontend_build" "cd website-ui && npm run build -- --webpack"

{
  echo "# Repo Healthcheck"
  echo
  echo "- run_id: \`$RUN_ID\`"
  echo "- repo: \`$ROOT\`"
  echo "- head: \`$(cat "$OUTDIR/git_head.txt" 2>/dev/null || echo unknown)\`"
  echo "- status: \`$([ "$FAIL_COUNT" -eq 0 ] && echo ok || echo fail)\`"
  echo "- failed_checks: \`$FAIL_COUNT\`"
  echo
  echo "## Checks"
  while IFS=$'\t' read -r name status cmd; do
    echo "- [$status] \`$name\`"
    echo "  - cmd: \`$cmd\`"
  done <"$CHECKS_TSV"
  echo
  echo "## Artefatos"
  echo "- log: \`results/ops/healthcheck/$RUN_ID/healthcheck.log\`"
  echo "- checks: \`results/ops/healthcheck/$RUN_ID/checks.tsv\`"
  echo "- git status: \`results/ops/healthcheck/$RUN_ID/git_status.txt\`"
} >"$REPORT"

echo "[healthcheck] report=$REPORT failed_checks=$FAIL_COUNT"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi
