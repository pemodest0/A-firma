#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$HOME/A-firma}"
PLIST_SRC="$REPO/scripts/ops/launchd/com.assyntrax.investment-shadow.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.assyntrax.investment-shadow.plist"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$REPO/results/ops/invest_shadow/logs"

sed "s|__REPO_PATH__|$REPO|g" "$PLIST_SRC" > "$PLIST_DST"

launchctl unload "$PLIST_DST" >/dev/null 2>&1 || true
launchctl load "$PLIST_DST"

echo "[ok] launchd job loaded: com.assyntrax.investment-shadow"
echo "check: launchctl list | grep assyntrax"
echo "logs:  tail -f $REPO/results/ops/invest_shadow/logs/daily.out.log"
