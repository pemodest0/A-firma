#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "[energy-sync] repo=$ROOT"
python3 scripts/data/sync_energy_ons_one_shot.py "$@"
echo "[energy-sync] done"
