#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

APPLY=0
INCLUDE_RESULTS_CACHE=0

for arg in "$@"; do
  case "$arg" in
    --yes) APPLY=1 ;;
    --include-results-cache) INCLUDE_RESULTS_CACHE=1 ;;
    *)
      echo "Unknown arg: $arg" >&2
      echo "Usage: bash scripts/maintenance/cleanup_repo.sh [--yes] [--include-results-cache]" >&2
      exit 2
      ;;
  esac
done

declare -a TARGETS=(
  ".pytest_cache"
  "website-ui/.next"
  "website-ui/.turbo"
  "logs"
)

if [[ "$INCLUDE_RESULTS_CACHE" -eq 1 ]]; then
  TARGETS+=("results/_tmp" "results/_figs")
fi

echo "[cleanup] mode=$([[ "$APPLY" -eq 1 ]] && echo apply || echo dry-run)"
echo "[cleanup] targets:"
for t in "${TARGETS[@]}"; do
  echo "  - $t"
done

if [[ "$APPLY" -ne 1 ]]; then
  echo "[cleanup] dry-run complete. Re-run with --yes to apply."
  exit 0
fi

for t in "${TARGETS[@]}"; do
  if [[ -e "$t" ]]; then
    rm -rf "$t"
    echo "[cleanup] removed $t"
  fi
done

echo "[cleanup] done"
