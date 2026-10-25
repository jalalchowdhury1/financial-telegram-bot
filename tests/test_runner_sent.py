"""scripts/runner_sent.sh counts only daily_report.yml runs that actually SENT the brief.
A guard-skipped run also ends "success"; counting it made the health check call a
Lambda-fail day 'delivered by backstop' and let a late cron send a second brief."""
import os
import subprocess

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FAKE_GH = """#!/bin/bash
echo "$*" >> "$GH_LOG"
if [ "$1 $2" = "run list" ]; then printf '101\\n102\\n'; exit 0; fi
# run view <id>: 101 sent the brief, 102 was skipped by the guard
[ "$3" = "101" ] && echo 1 || echo 0
"""


def _run(tmp_path, *args):
    gh = tmp_path / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "GH_LOG": str(tmp_path / "log")}
    out = subprocess.run(["bash", os.path.join(ROOT, "scripts", "runner_sent.sh"), *args],
                         capture_output=True, text=True, env=env, check=True).stdout.strip()
    return out, (tmp_path / "log").read_text()


def test_a_guard_skipped_run_is_not_a_delivery(tmp_path):
    out, log = _run(tmp_path)
    assert out == "1"
    assert "< 86400)" in log


def test_window_can_be_narrowed(tmp_path):
    _, log = _run(tmp_path, "3600")
    assert "< 3600)" in log


def _wf(name):
    with open(os.path.join(ROOT, ".github", "workflows", name)) as fh:
        return yaml.safe_load(fh)


def test_cron_guard_also_skips_when_a_runner_already_sent_today():
    guard = next(s for s in _wf("daily_report.yml")["jobs"]["generate-and-send-report"]["steps"]
                 if s.get("id") == "guard")
    assert "runner_sent.sh" in guard["run"] and "GH_TOKEN" in guard["env"]


def test_health_check_evidence_counts_only_real_sends_and_alert_failure_keeps_the_fix():
    steps = [s for job in _wf("health-check.yml")["jobs"].values() for s in job.get("steps", [])]
    assert any("runner_sent.sh" in (s.get("run") or "") for s in steps)
    fix = next(s for s in steps if s.get("name") == "Auto-remediate a confirmed missed report")
    assert "!cancelled()" in fix["if"]
