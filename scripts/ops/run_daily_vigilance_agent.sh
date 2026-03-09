#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
  echo "[daily_vigilance_agent] python3 nao encontrado" >&2
  exit 1
fi

mkdir -p "$ROOT/results/ops/agents/logs"
cd "$ROOT"
"$PYTHON_BIN" scripts/ops/run_daily_vigilance_agent.py
