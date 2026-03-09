#!/usr/bin/env bash
set -euo pipefail

OLD_JOBS=(
  "com.assyntrax.profit-shadow"
  "com.assyntrax.profit-shadow-target800"
  "com.assyntrax.investment-shadow"
)

for job in "${OLD_JOBS[@]}"; do
  plist="$HOME/Library/LaunchAgents/${job}.plist"
  launchctl unload "$plist" >/dev/null 2>&1 || true
  rm -f "$plist"
done

echo "[ok] jobs antigos removidos:"
for job in "${OLD_JOBS[@]}"; do
  echo " - $job"
done
