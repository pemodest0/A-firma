#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p \
  data/download \
  data/clean \
  data/processed \
  data/validated

for d in data/download data/clean data/processed data/validated; do
  if [[ ! -f "$d/.gitkeep" ]]; then
    touch "$d/.gitkeep"
  fi
done

echo "[data-layout] ok: created/verified data/download, data/clean, data/processed, data/validated"
