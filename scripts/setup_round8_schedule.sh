#!/bin/zsh
set -euo pipefail

ROOT="/Users/sanilbaweja/Projects/parameter-golf"
PLIST="$ROOT/launchd/parameter-golf-overnight-round8.plist"
LABEL="com.sanilbaweja.parameter-golf.overnight-round8"
UID_NUM="$(id -u)"
RUN_TIME_FMT="$(date -v+5M +'%m/%d/%y %H:%M:00')"
RUN_HOUR="$(date -j -f '%m/%d/%y %H:%M:%S' "$RUN_TIME_FMT" +'%H')"
RUN_MINUTE="$(date -j -f '%m/%d/%y %H:%M:%S' "$RUN_TIME_FMT" +'%M')"

mkdir -p "$ROOT/logs/overnight_local_ab_round8"
chmod +x "$ROOT/scripts/run_overnight_local_ab_round8.sh"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "missing repo venv python: $ROOT/.venv/bin/python" >&2
  exit 1
fi

/usr/libexec/PlistBuddy -c "Set :StartCalendarInterval:Hour $RUN_HOUR" "$PLIST"
/usr/libexec/PlistBuddy -c "Set :StartCalendarInterval:Minute $RUN_MINUTE" "$PLIST"

echo "Scheduling wake for $RUN_TIME_FMT"
sudo pmset schedule wakeorpoweron "$RUN_TIME_FMT"

launchctl bootout "gui/$UID_NUM" "$PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST"
launchctl enable "gui/$UID_NUM/$LABEL"

echo
echo "Scheduled:"
echo "  wake: $RUN_TIME_FMT"
echo "  start: $(printf '%02d:%02d' "$RUN_HOUR" "$RUN_MINUTE")"
echo "  launchd label: $LABEL"
echo "  plist: $PLIST"
echo
echo "Verify with:"
echo "  pmset -g sched"
echo "  launchctl print gui/$UID_NUM/$LABEL"
