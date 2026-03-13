#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

run_step() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  echo "CMD: $*"
  "$@"
}

run_step "historical_closure" \
  python3 scripts/bench/validation/run_profit_historical_closure_suite.py

run_step "pbo" \
  python3 scripts/bench/validation/run_profit_pbo_suite.py

run_step "execution_phase" \
  python3 scripts/bench/validation/run_profit_execution_phase_suite.py

run_step "universe_resilience" \
  python3 scripts/bench/validation/run_profit_universe_resilience_suite.py

run_step "bad_year_defense" \
  python3 scripts/bench/validation/run_profit_bad_year_defense_suite.py

run_step "u800_alpha" \
  python3 scripts/bench/validation/run_profit_u800_alpha_suite.py

run_step "marketmode_criticality" \
  python3 scripts/bench/validation/run_profit_marketmode_criticality_suite.py

run_step "meta_mode_selector" \
  python3 scripts/bench/validation/run_profit_meta_mode_selector_suite.py

run_step "hypothesis_lab_board" \
  python3 scripts/bench/validation/run_profit_hypothesis_lab_suite.py --publish-ops
