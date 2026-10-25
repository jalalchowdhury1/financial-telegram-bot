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
# Decision layer: red streaks -> GO DEFENSIVE / RE-ENTER (2026-09-08). Runs even when the
# radar failed: a gist error comes AFTER the snapshot is on disk, and an old snapshot is a
# no-op (evaluate skips an asOf it already counted). Its failure alerts and fails the job.
"$PY" scripts/defensive_trigger.py evaluate
erc=$?
if [ $erc -ne 0 ]; then
  "$PY" -c 'import sys; sys.path.insert(0, "scripts"); from defensive_trigger import send_telegram; send_telegram("🪢 <b>Defensive trigger · evaluate FAILED</b> (exit " + sys.argv[1] + ")\nGO DEFENSIVE was not checked tonight. See ~/Library/Logs/rubber-band.log")' "$erc"
fi
echo "=== exit $rc / evaluate $erc ==="
[ $rc -eq 0 ] && [ $erc -eq 0 ]
