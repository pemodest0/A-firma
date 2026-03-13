#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$HOME/Downloads/Assyntrax}"
PLIST_SRC="$REPO/scripts/ops/launchd/com.assyntrax.daily-publish.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.assyntrax.daily-publish.plist"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$REPO/results/ops/logs"

sed "s|__REPO_PATH__|$REPO|g" "$PLIST_SRC" > "$PLIST_DST"

launchctl unload "$PLIST_DST" >/dev/null 2>&1 || true
launchctl load "$PLIST_DST"

echo "[ok] launchd job loaded: com.assyntrax.daily-publish"
echo "check: launchctl list | grep assyntrax"
echo "logs:  tail -f $REPO/results/ops/logs/daily_publish.out.log"
