#!/bin/bash
# Nightly Rubber Band Radar run on the Mac mini (launchd: com.jalal.rubber-band).
# Computes the five dials from QQQ closes + Composer backtest curves, publishes the
# snapshot to the secret gist the dashboard reads, and alerts the 📡 thread only when
# a dial changes colour. Secrets never live here: Composer creds come from
# composer-auto-research/.env, the Telegram token from this repo's .env, and the
# alert chat id from the launchd EnvironmentVariables (RUBBER_BAND_ALERT_CHAT).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO" || exit 1
echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') rubber-band nightly ==="
"$PY" scripts/rubber_band.py run
rc=$?
if [ $rc -eq 0 ]; then
  "$PY" scripts/defensive_trigger.py evaluate        # decision layer: red streaks -> GO DEFENSIVE / RE-ENTER (2026-09-08)
fi
echo "=== exit $rc ==="
exit $rc
