#!/bin/zsh
set -euo pipefail

ROOT="/Users/sanilbaweja/Projects/parameter-golf"
PLIST="$ROOT/launchd/parameter-golf-overnight-round2.plist"
LABEL="com.sanilbaweja.parameter-golf.overnight-round2"
UID_NUM="$(id -u)"

launchctl bootout "gui/$UID_NUM" "$PLIST" >/dev/null 2>&1 || true
sudo pmset schedule cancelall

echo "Removed launchd job $LABEL and cleared scheduled power events."
