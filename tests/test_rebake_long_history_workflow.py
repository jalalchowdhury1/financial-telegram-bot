"""rebake-long-history.yml: its notify step, executed for real against stub git/gh/curl.

The quarterly job's whole output is silence or one Telegram line, and its production path
had never run when the owner asked (2026-10-11) for no alerts from it unless he has to
act. So the step's bash runs here under GitHub's own `bash -eo pipefail`, once for each
way a run can end. Green = nothing for him to do: merged, or nothing new to ship (a
same-day re-run), silent unless a chart is at risk (🟡). Red = he must act (🔴): the bake
died, baked nothing at all, or its PR could not be merged. GitHub blips are retried.
"""
import os
import re
import subprocess

import pytest
import yaml

WORKFLOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows",
                        "rebake-long-history.yml")

STUBS = {
    "git": r'''#!/bin/bash
S="$STUB_STATE"
count() { n=$(cat "$S/$1" 2>/dev/null || echo 0); echo $((n + 1)) > "$S/$1"; echo "$n"; }
case "$1" in
  status) [ -z "${STUB_DIRTY:-}" ] || echo " M dashboard/public/history/dxy.json" ;;
  push) if [ "$(count pushes)" -lt "${STUB_PUSH_FAILS:-0}" ]; then echo "remote: Internal Server Error"; exit 1; fi ;;
esac
exit 0
''',
    "gh": r'''#!/bin/bash
S="$STUB_STATE"
count() { n=$(cat "$S/$1" 2>/dev/null || echo 0); echo $((n + 1)) > "$S/$1"; echo "$n"; }
case "$1 $2" in
  "pr create")
    if [ "$(count creates)" -lt "${STUB_CREATE_FAILS:-0}" ]; then echo "HTTP 502: Bad Gateway"; exit 1; fi
    touch "$S/pr"; echo "https://github.com/o/r/pull/7" ;;
  "pr view")
    case "$*" in
      *state*) if [ -f "$S/merged" ]; then echo MERGED; else echo OPEN; fi ;;
      *url*) [ -f "$S/pr" ] || exit 1; echo "https://github.com/o/r/pull/7" ;;
    esac ;;
  "pr checks")
    printf 'backend-tests\tpass\t1m2s\thttps://x\t\n'
    printf 'dashboard-tests\t%s\t6m40s\thttps://x\t\n' "${STUB_DASH:-pass}" ;;
  "pr merge")
    if [ "$(count merges)" -lt "${STUB_MERGE_FAILS:-0}" ]; then echo "Pull request is not mergeable"; exit 1; fi
    touch "$S/merged" ;;
  "workflow run")
    if [ "$(count dispatches)" -lt "${STUB_DISPATCH_FAILS:-0}" ]; then echo "HTTP 500"; exit 1; fi ;;
esac
exit 0
''',
    "curl": r'''#!/bin/bash
for a in "$@"; do case "$a" in text=*) printf '%s' "${a#text=}" > "$STUB_STATE/telegram" ;; esac; done
echo "telegram HTTP 200"
''',
    "python": "#!/bin/bash\necho 34\n",    # the step's only python call counts the index's stats
    "sleep": "#!/bin/bash\n",
}


WROTE = "\nwrote public/history/ + lib/data/longHistoryIndex.json"   # the bake's summary line


def _script(outcome):
    wf = yaml.safe_load(open(WORKFLOW))
    step = next(s for s in wf["jobs"]["rebake"]["steps"] if "gh pr create" in (s.get("run") or ""))
    run = step["run"].replace("${{ steps.bake.outcome }}", outcome).replace("${{ github.run_id }}", "99")
    assert "${{" not in run, re.findall(r"\$\{\{[^}]*\}\}", run)
    return run


@pytest.fixture
def run_step(tmp_path):
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    for name, body in STUBS.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)

    def run(log=WROTE + "\n", outcome="success", **stub):
        (tmp_path / "bake.log").write_text(log)
        env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "RUNNER_TEMP": str(tmp_path),
               "STUB_STATE": str(state), "GH_TOKEN": "x", "TELEGRAM_TOKEN": "t",
               "TELEGRAM_CHAT_ID": "c", "RUN_URL": "https://run", "COMPARE_URL": "https://compare",
               **{f"STUB_{k.upper()}": str(v) for k, v in stub.items()}}
        proc = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _script(outcome)],
                              env=env, capture_output=True, text=True, timeout=60)
        sent = state / "telegram"
        return proc.returncode, (sent.read_text() if sent.exists() else None)

    run.opened_pr = lambda: (state / "pr").exists()
    return run


def test_merged_with_every_stat_refreshed_is_silent_and_green(run_step):
    assert run_step(dirty=1) == (0, None)


def test_merged_while_a_stat_kept_its_old_bake_is_silent_and_green(run_step):
    """The kept file still draws its whole chart for ~2 years and next quarter retries it,
    so a missed stat is listed in the PR body and nobody is pinged."""
    log = f"dxy  FAILED download: CNBC .DXY ALL\n{WROTE}; FAILED/REFUSED: ['dxy']\n"
    assert run_step(log=log, dirty=1) == (0, None)


def test_merged_with_a_chart_at_risk_pings_yellow_and_stays_green(run_step):
    log = f"{WROTE}; FAILED/REFUSED: ['dxy']\nRUNWAY LOW: dxy (chart gap from 2027-06-01)\n"
    code, sent = run_step(log=log, dirty=1)
    assert code == 0                                   # green: the merged data must deploy
    assert sent.startswith("🟡") and "dxy (chart gap from 2027-06-01)" in sent


def test_a_red_ci_leaves_the_pr_open_pings_red_and_ends_red(run_step):
    code, sent = run_step(dirty=1, dash="fail")
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "pull/7" in sent


def test_nothing_new_to_ship_is_silent_green_and_opens_no_pr(run_step):
    """A re-run on the same day bakes the same files. Green costs no deploy:
    deploy-dashboard.yml deploys a rebake only when main moved."""
    assert run_step() == (0, None)
    assert not run_step.opened_pr()


def test_nothing_new_while_a_stat_kept_its_old_bake_is_silent_and_green(run_step):
    assert run_step(log=f"{WROTE}; FAILED/REFUSED: ['dxy']\n", outcome="failure") == (0, None)


def test_nothing_new_but_a_chart_at_risk_pings_yellow_and_stays_green(run_step):
    log = f"{WROTE}; FAILED/REFUSED: ['dxy']\nRUNWAY LOW: dxy (chart gap from 2027-06-01)\n"
    code, sent = run_step(log=log, outcome="failure")
    assert code == 0
    assert sent.startswith("🟡") and "dxy (chart gap from 2027-06-01)" in sent


def test_every_stat_failing_pings_red_and_ends_red(run_step):
    log = f"{WROTE}; FAILED/REFUSED: ['yieldCurve', 'tnx']\nNOTHING BAKED: every stat kept its old file\n"
    code, sent = run_step(log=log, outcome="failure")
    assert code != 0
    assert sent.startswith("🔴") and "baked nothing" in sent and "yieldCurve" in sent


def test_a_bake_that_died_before_its_summary_ships_nothing_and_pings_red(run_step):
    """Files it wrote before dying are not shipped without the index that describes them."""
    log = "Traceback (most recent call last):\nRuntimeError: curl exit 22: https://docs.google.com/x\n"
    code, sent = run_step(log=log, outcome="failure", dirty=1)
    assert code != 0
    assert sent.startswith("🔴") and "stopped before it finished" in sent and "failure" in sent
    assert not run_step.opened_pr()


def test_a_merge_that_needs_a_second_try_still_merges_silently(run_step):
    assert run_step(dirty=1, merge_fails=1) == (0, None)


def test_a_github_blip_on_pr_create_is_retried(run_step):
    assert run_step(dirty=1, create_fails=1) == (0, None)


def test_a_pr_that_never_opens_pings_red_with_the_compare_link(run_step):
    code, sent = run_step(dirty=1, create_fails=9)
    assert code != 0
    assert sent.startswith("🔴") and "https://compare" in sent


def test_a_github_blip_on_push_is_retried(run_step):
    assert run_step(dirty=1, push_fails=1) == (0, None)


def test_a_push_that_keeps_failing_pings_red_and_ends_red(run_step):
    code, sent = run_step(dirty=1, push_fails=9)
    assert code != 0
    assert sent.startswith("🔴") and "could not push" in sent


def test_a_github_blip_on_the_ci_dispatch_is_retried(run_step):
    assert run_step(dirty=1, dispatch_fails=1) == (0, None)


def test_a_ci_that_never_starts_leaves_the_pr_open_and_pings_red(run_step):
    code, sent = run_step(dirty=1, dispatch_fails=9)
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent
