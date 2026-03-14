#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$HOME/Downloads/Assyntrax}"
AGENTS_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS_DIR"
mkdir -p "$REPO/results/ops/agents/logs"

for name in \
  com.assyntrax.daily-ingestion-agent \
  com.assyntrax.daily-backfill-agent \
  com.assyntrax.daily-operation-agent \
  com.assyntrax.daily-vigilance-agent \
  com.assyntrax.daily-data-quality-agent \
  com.assyntrax.daily-publish \
  com.assyntrax.daily-smoke-test-agent \
  com.assyntrax.daily-watchdog-agent
do
  src="$REPO/scripts/ops/launchd/${name}.plist"
  dst="$AGENTS_DIR/${name}.plist"
  sed "s|__REPO_PATH__|$REPO|g" "$src" > "$dst"
  launchctl unload "$dst" >/dev/null 2>&1 || true
  launchctl load "$dst"
done

echo "[ok] agentes carregados:"
echo " - com.assyntrax.daily-ingestion-agent"
echo " - com.assyntrax.daily-backfill-agent"
echo " - com.assyntrax.daily-operation-agent"
echo " - com.assyntrax.daily-vigilance-agent"
echo " - com.assyntrax.daily-data-quality-agent"
echo " - com.assyntrax.daily-publish"
echo " - com.assyntrax.daily-smoke-test-agent"
echo " - com.assyntrax.daily-watchdog-agent"
echo "check: launchctl list | grep assyntrax"
