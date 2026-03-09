#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$HOME/Downloads/Assyntrax}"
AGENTS_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS_DIR"
mkdir -p "$REPO/results/ops/agents/logs"

for name in \
  com.assyntrax.daily-operation-agent \
  com.assyntrax.daily-vigilance-agent
do
  src="$REPO/scripts/ops/launchd/${name}.plist"
  dst="$AGENTS_DIR/${name}.plist"
  sed "s|__REPO_PATH__|$REPO|g" "$src" > "$dst"
  launchctl unload "$dst" >/dev/null 2>&1 || true
  launchctl load "$dst"
done

echo "[ok] agentes carregados:"
echo " - com.assyntrax.daily-operation-agent"
echo " - com.assyntrax.daily-vigilance-agent"
echo "check: launchctl list | grep assyntrax"
