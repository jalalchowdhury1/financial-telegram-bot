"""The nightly wrapper must run the GO DEFENSIVE evaluation even when the radar's gist
publish failed (the snapshot is already on disk), and an evaluate crash must alert and
fail the launchd job instead of hiding behind the radar's exit 0 (audit 25 Oct 2026)."""
import os
import shutil
import subprocess

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "rubber_band_nightly.sh")

# Fake interpreter: logs each call; exit codes come from RB_RC / DT_RC.
FAKE_PY = """#!/bin/bash
echo "$*" >> "$CALLS"
case "$1" in
  scripts/rubber_band.py) exit "$RB_RC" ;;
  scripts/defensive_trigger.py) exit "$DT_RC" ;;
esac
exit 0
"""


def _run(tmp_path, rb_rc, dt_rc):
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "rubber_band_nightly.sh")
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    calls = tmp_path / "calls.log"
    env = dict(os.environ, CALLS=str(calls), RB_RC=str(rb_rc), DT_RC=str(dt_rc))
    rc = subprocess.run(["bash", str(tmp_path / "scripts" / "rubber_band_nightly.sh")],
                        env=env, capture_output=True).returncode
    return rc, calls.read_text().splitlines()


def test_clean_night_runs_both_and_exits_zero(tmp_path):
    rc, calls = _run(tmp_path, 0, 0)
    assert rc == 0
    assert calls == ["scripts/rubber_band.py run", "scripts/defensive_trigger.py evaluate"]


def test_radar_failure_still_evaluates_and_fails_the_job(tmp_path):
    rc, calls = _run(tmp_path, 1, 0)
    assert rc != 0
    assert "scripts/defensive_trigger.py evaluate" in calls
    assert not any(c.startswith("-c") for c in calls)      # the radar sends its own alert


def test_evaluate_crash_alerts_and_fails_the_job(tmp_path):
    rc, calls = _run(tmp_path, 0, 1)
    assert rc != 0
    alert = [c for c in calls if c.startswith("-c")]
    assert len(alert) == 1 and "send_telegram" in alert[0] and "evaluate FAILED" in alert[0]
