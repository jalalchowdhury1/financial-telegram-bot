"""rebake-long-history.yml: its notify step, executed for real against stub git/gh/curl.

The quarterly job's whole output is silence or one Telegram line, and its production path
had never run when the owner asked (2026-10-11) for no alerts from it unless he has to
act. So the step's bash runs here under GitHub's own `bash -eo pipefail`, once for each
way a run can end. Green = nothing for him to do: merged, or nothing new to ship (a
same-day re-run), silent unless a chart is at risk (🟡). Red = he must act (🔴): the bake
died, baked nothing at all, or its PR could not be merged. GitHub blips are retried, and a
PR that main moved past (branch protection is strict) is re-cut on the newest main.
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
  push) if [ "$(count pushes)" -lt "${STUB_PUSH_FAILS:-0}" ]; then echo "remote: Internal Server Error"; exit 1; fi
        echo "$3" >> "$S/pushed" ;;
  fetch) [ -z "${STUB_FETCH_FAILS:-}" ] || exit 1 ;;
  diff) [ -z "${STUB_MAIN_CHANGED_BAKE:-}" ] || exit 1 ;;     # main changed the baked files itself
  rev-parse) echo 0123abc ;;
esac
exit 0
''',
    # PRs are numbered from 7; the newest one is live. Main has moved past the first
    # STUB_BEHIND of them, so GitHub refuses to merge those.
    "gh": r'''#!/bin/bash
S="$STUB_STATE"
count() { n=$(cat "$S/$1" 2>/dev/null || echo 0); echo $((n + 1)) > "$S/$1"; echo "$n"; }
prs() { cat "$S/prs" 2>/dev/null || echo 0; }
url() { echo "https://github.com/o/r/pull/$((6 + $(prs)))"; }
behind() { [ "$(prs)" -le "${STUB_BEHIND:-0}" ]; }
case "$1 $2" in
  "pr create")
    if [ "$(count creates)" -lt "${STUB_CREATE_FAILS:-0}" ]; then echo "HTTP 502: Bad Gateway"; exit 1; fi
    echo $(($(prs) + 1)) > "$S/prs"; url ;;
  "pr view")
    [ -z "${STUB_VIEW_FAILS:-}" ] || exit 1
    case "$*" in
      *mergeStateStatus*) if behind; then echo BEHIND; else echo BLOCKED; fi ;;
      *state*) if [ -f "$S/merged" ]; then echo MERGED; else echo OPEN; fi ;;
      *url*) [ "$(prs)" -gt 0 ] || exit 1; url ;;
    esac ;;
  "pr checks")
    if [ "$(count polls)" -lt "${STUB_PENDING:-0}" ]; then printf 'backend-tests\tpending\t0\thttps://x\t\n'; exit 8; fi
    printf 'backend-tests\tpass\t1m2s\thttps://x\t\n'
    printf 'dashboard-tests\t%s\t6m40s\thttps://x\t\n' "${STUB_DASH:-pass}" ;;
  "pr merge")
    if behind; then echo "X Pull request is not mergeable: the head branch is not up to date with the base branch"; exit 1; fi
    if [ "$(count merges)" -lt "${STUB_MERGE_FAILS:-0}" ]; then echo "Pull request is not mergeable"; exit 1; fi
    touch "$S/merged"
    if [ -n "${STUB_MERGE_EXIT1:-}" ]; then echo "merged, but could not delete the local branch"; exit 1; fi ;;
  "pr close")
    echo "$*" >> "$S/closed" ;;
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
GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}   # no signing etc.


def _script(outcome, attempt):
    wf = yaml.safe_load(open(WORKFLOW))
    step = next(s for s in wf["jobs"]["rebake"]["steps"] if "gh pr create" in (s.get("run") or ""))
    run = (step["run"].replace("${{ steps.bake.outcome }}", outcome)
           .replace("${{ github.run_id }}", "99").replace("${{ github.run_attempt }}", str(attempt)))
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

    def run(log=WROTE + "\n", outcome="success", attempt=1, cwd=None, **stub):
        (tmp_path / "bake.log").write_text(log)
        env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "RUNNER_TEMP": str(tmp_path),
               "HOME": str(tmp_path), "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
               "STUB_STATE": str(state), "GH_TOKEN": "x", "TELEGRAM_TOKEN": "t",
               "TELEGRAM_CHAT_ID": "c", "RUN_URL": "https://run", "REPO_URL": "https://github.com/o/r",
               **{f"STUB_{k.upper()}": str(v) for k, v in stub.items()}}
        proc = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _script(outcome, attempt)],
                              env=env, cwd=cwd, capture_output=True, text=True, timeout=60)
        sent = state / "telegram"
        return proc.returncode, (sent.read_text() if sent.exists() else None)

    def lines(name):
        f = state / name
        return f.read_text().splitlines() if f.exists() else []

    run.opened_pr = lambda: (state / "prs").exists()
    run.pushed = lambda: lines("pushed")
    run.closed = lambda: lines("closed")
    run.use_real_git = lambda: (bin_dir / "git").unlink()
    return run


class Checkout:
    """A depth-1 clone of a local origin, as actions/checkout leaves it, with the bake's
    output already written into it, and a second clone that can move main meanwhile."""

    def __init__(self, root):
        self.origin, self.seed, self.clone = root / "origin.git", root / "seed", root / "clone"
        self._git("init", "-q", "--bare", "-b", "main", str(self.origin))
        self._git("init", "-q", "-b", "main", str(self.seed))
        self._write(self.seed, "dashboard/public/history/dxy.json", "old bake\n")
        self._write(self.seed, "dashboard/lib/data/longHistoryIndex.json", "{}\n")
        self._write(self.seed, ".github/workflows/ci.yml", "v1\n")
        self._git("-C", str(self.seed), "add", "-A")
        self._git("-C", str(self.seed), "commit", "-qm", "seed")
        self._git("-C", str(self.seed), "push", "-q", str(self.origin), "main")
        self._git("clone", "-q", "--depth", "1", f"file://{self.origin}", str(self.clone))
        self._write(self.clone, "dashboard/public/history/dxy.json", "new bake\n")
        self._write(self.clone, "dashboard/lib/data/longHistoryIndex.json", '{"new": 1}\n')

    @staticmethod
    def _write(root, rel, text):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)

    @staticmethod
    def _git(*args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], env=GIT_ENV,
                              check=True, capture_output=True, text=True).stdout

    def main_moves(self, rel, text):
        self._write(self.seed, rel, text)
        self._git("-C", str(self.seed), "commit", "-qam", f"main changes {rel}")
        self._git("-C", str(self.seed), "push", "-q", str(self.origin), "main")

    def at_origin(self, *args):
        return self._git("-C", str(self.origin), *args)


@pytest.fixture
def checkout(tmp_path):
    return Checkout(tmp_path)


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
    assert sent.startswith("🔴") and "NOT merged" in sent and "CI failed" in sent and "pull/7" in sent


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


def test_github_refusing_the_merge_every_time_pings_red(run_step):
    code, sent = run_step(dirty=1, merge_fails=99)
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "refused" in sent and "pull/7" in sent


def test_a_merge_github_then_cannot_confirm_still_counts(run_step):
    """gh pr merge said yes, then every `gh pr view` failed: it merged, so no false alarm and
    no red run (a red run would also skip the deploy of the merged data)."""
    assert run_step(dirty=1, view_fails=1) == (0, None)


def test_a_merge_that_landed_despite_an_error_exit_is_silent(run_step):
    assert run_step(dirty=1, merge_exit1=1) == (0, None)


def test_checks_still_running_are_waited_for(run_step):
    assert run_step(dirty=1, pending=5) == (0, None)


def test_ci_that_never_turns_green_leaves_the_pr_open_and_pings_red(run_step):
    code, sent = run_step(dirty=1, pending=999)
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "40 min" in sent and "pull/7" in sent


def test_a_pr_main_moved_past_is_re_cut_on_the_newest_main_and_merges_silently(run_step):
    """Branch protection is strict, so GitHub refuses a PR once main moves past it, and main
    takes ~50 commits on a busy day. GitHub's token cannot merge main into the old branch
    when main brought a workflow change, so the bake is re-cut on the newest main as a fresh
    PR and the old one is closed."""
    assert run_step(dirty=1, behind=1) == (0, None)
    assert run_step.pushed() == ["rebake/long-history-99-1", "rebake/long-history-99-1-2"]
    [closed] = run_step.closed()
    assert "pull/7" in closed and "Superseded by https://github.com/o/r/pull/8" in closed


def test_main_moving_past_every_cut_pings_red_with_the_newest_pr(run_step):
    code, sent = run_step(dirty=1, behind=3)
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "main kept moving" in sent and "pull/9" in sent
    assert len(run_step.pushed()) == 3


def test_a_re_cut_never_overwrites_baked_files_that_changed_on_main(run_step):
    code, sent = run_step(dirty=1, behind=1, main_changed_bake=1)
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "baked files" in sent and "pull/7" in sent
    assert run_step.pushed() == ["rebake/long-history-99-1"]


def test_a_re_cut_with_real_git_is_the_newest_main_plus_the_bake_and_nothing_else(run_step, checkout):
    """Real git, stub GitHub: main moved with a workflow change (what GitHub's token could not
    merge into the old branch), so the re-cut must sit on the new main and add only the bake."""
    run_step.use_real_git()
    checkout.main_moves(".github/workflows/ci.yml", "v2\n")
    assert run_step(behind=1, cwd=checkout.clone) == (0, None)
    cut = "rebake/long-history-99-1-2"
    assert checkout.at_origin("rev-parse", f"{cut}~1") == checkout.at_origin("rev-parse", "main")
    assert checkout.at_origin("diff", "--name-only", "main", cut).split() == [
        "dashboard/lib/data/longHistoryIndex.json", "dashboard/public/history/dxy.json"]
    assert checkout.at_origin("show", f"{cut}:dashboard/public/history/dxy.json") == "new bake\n"
    assert checkout.at_origin("show", f"{cut}:.github/workflows/ci.yml") == "v2\n"


def test_a_re_cut_with_real_git_leaves_a_bake_that_landed_on_main_alone(run_step, checkout):
    run_step.use_real_git()
    checkout.main_moves("dashboard/public/history/dxy.json", "a hand bake\n")
    code, sent = run_step(behind=1, cwd=checkout.clone)
    assert code != 0 and "changed the baked files" in sent
    assert checkout.at_origin("branch", "--format=%(refname:short)").split() == ["main", "rebake/long-history-99-1"]


def test_a_re_run_cuts_its_own_branch(run_step):
    """A re-run keeps the run id; without the attempt number its push would hit the first
    attempt's branch and read as "could not push"."""
    assert run_step(dirty=1, attempt=2) == (0, None)
    assert run_step.pushed() == ["rebake/long-history-99-2"]


def test_a_github_blip_on_pr_create_is_retried(run_step):
    assert run_step(dirty=1, create_fails=1) == (0, None)


def test_a_pr_that_never_opens_pings_red_with_the_compare_link(run_step):
    code, sent = run_step(dirty=1, create_fails=9)
    assert code != 0
    assert sent.startswith("🔴") and "https://github.com/o/r/compare/rebake/long-history-99-1?expand=1" in sent


def test_a_github_blip_on_push_is_retried(run_step):
    assert run_step(dirty=1, push_fails=1) == (0, None)


def test_a_push_that_keeps_failing_pings_red_and_ends_red(run_step):
    code, sent = run_step(dirty=1, push_fails=9)
    assert code != 0
    assert sent.startswith("🔴") and "could not push" in sent
    assert not run_step.opened_pr()


def test_a_github_blip_on_the_ci_dispatch_is_retried(run_step):
    assert run_step(dirty=1, dispatch_fails=1) == (0, None)


def test_a_ci_that_never_starts_leaves_the_pr_open_and_pings_red(run_step):
    code, sent = run_step(dirty=1, dispatch_fails=9)
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "CI would not start" in sent
