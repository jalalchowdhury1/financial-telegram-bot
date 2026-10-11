"""rebake-bankruptcies.yml: its ship step, executed for real against stub git/gh/python.

The monthly job's whole output is silence or one Telegram line, and the owner wants no
alert from it unless he has to act (2026-10-11). So the step's bash runs here under
GitHub's own `bash -eo pipefail`, once for each way a run can end. Green and silent:
no new quarter, or a new one merged, or main already has the bake. Red and 🔴, with a link
to run it again: the rebuild failed, or its PR could not be merged. GitHub blips are
retried, a PR that main moved past (branch protection is strict) is re-cut on the newest
main, and older bot PRs are closed once one merges.

Also here: ci.yml's `report` job, which is what lets a bot PR merge at all.
"""
import os
import re
import subprocess

import pytest
import yaml

WORKFLOWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows")
BAKE = "dashboard/lib/data/bankruptciesBaked.json"

STUBS = {
    "git": r'''#!/bin/bash
S="$STUB_STATE"
count() { n=$(cat "$S/$1" 2>/dev/null || echo 0); echo $((n + 1)) > "$S/$1"; echo "$n"; }
case "$1" in
  status) [ -z "${STUB_DIRTY:-}" ] || echo " M dashboard/lib/data/bankruptciesBaked.json" ;;
  push) if [ "$(count pushes)" -lt "${STUB_PUSH_FAILS:-0}" ]; then echo "remote: Internal Server Error"; exit 1; fi
        echo "$3" >> "$S/pushed" ;;
  fetch) [ -z "${STUB_FETCH_FAILS:-}" ] || exit 1 ;;
  diff) case "$3" in                                          # the bake's commit is 0123abc
          *~1) [ -z "${STUB_MAIN_CHANGED_BAKE:-}" ] || exit 1 ;;  # main changed the baked file itself
          *) [ -n "${STUB_MAIN_HAS_BAKE:-}" ] || exit 1 ;;          # main already has this exact bake
        esac ;;
  rev-parse) echo 0123abc ;;
esac
exit 0
''',
    # PRs are numbered from 7; the newest one is live. Main has moved past the first
    # STUB_BEHIND of them, so GitHub refuses to merge those. STUB_OPEN lists other open
    # bot PRs (what `gh pr list` returns after its jq filter).
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
    if [ "$(count polls)" -lt "${STUB_PENDING:-0}" ]; then exit 8; fi    # no statuses yet
    printf 'backend-tests\tpass\t0\thttps://x\tdispatched CI run: success\n'
    printf 'dashboard-tests\t%s\t0\thttps://x\tdispatched CI run: %s\n' "${STUB_DASH:-pass}" "${STUB_DASH:-pass}" ;;
  "pr merge")
    if behind; then echo "X Pull request is not mergeable: the head branch is not up to date with the base branch"; exit 1; fi
    if [ "$(count merges)" -lt "${STUB_MERGE_FAILS:-0}" ]; then echo "Pull request is not mergeable"; exit 1; fi
    touch "$S/merged"
    if [ -n "${STUB_MERGE_EXIT1:-}" ]; then echo "merged, but could not delete the local branch"; exit 1; fi ;;
  "pr list")
    for n in ${STUB_OPEN:-}; do echo "$n"; done; echo "$(($(prs) + 6))" ;;   # the bot's own PR is open too
  "pr close")
    echo "$*" >> "$S/closed" ;;
  "workflow run")
    if [ "$(count dispatches)" -lt "${STUB_DISPATCH_FAILS:-0}" ]; then echo "HTTP 500"; exit 1; fi ;;
esac
exit 0
''',
    # Two python calls: `python -c` counts the bake's quarters, `python -` sends $MSG.
    "python": r'''#!/bin/bash
if [ "$1" = "-c" ]; then echo 101; exit 0; fi
cat > /dev/null
printf '%s' "$MSG" > "$STUB_STATE/telegram"
''',
    "sleep": "#!/bin/bash\n",
}

GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}   # no signing etc.


def _load(name):
    with open(os.path.join(WORKFLOWS, name)) as fh:
        return yaml.safe_load(fh)


def _script(outcome, attempt):
    wf = _load("rebake-bankruptcies.yml")
    step = next(s for s in wf["jobs"]["rebake"]["steps"] if "gh pr create" in (s.get("run") or ""))
    run = (step["run"].replace("${{ steps.rebake.outcome }}", outcome)
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

    def run(log="no change (100 quarters)\n", outcome="success", attempt=1, cwd=None, **stub):
        (tmp_path / "rebake.log").write_text(log)
        env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "RUNNER_TEMP": str(tmp_path),
               "HOME": str(tmp_path), "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
               "STUB_STATE": str(state), "GH_TOKEN": "x", "TELEGRAM_TOKEN": "t",
               "TELEGRAM_CHAT_ID": "c", "RUN_URL": "https://run", "REPO_URL": "https://github.com/o/r",
               "WF_URL": "https://wf",
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
    """A depth-1 clone of a local origin, as actions/checkout leaves it, with the rebuild's
    output already written into it, and a second clone that can move main meanwhile."""

    def __init__(self, root):
        self.origin, self.seed, self.clone = root / "origin.git", root / "seed", root / "clone"
        self._git("init", "-q", "--bare", "-b", "main", str(self.origin))
        self._git("init", "-q", "-b", "main", str(self.seed))
        self._write(self.seed, BAKE, "[old bake]\n")
        self._write(self.seed, ".github/workflows/ci.yml", "v1\n")
        self._git("-C", str(self.seed), "add", "-A")
        self._git("-C", str(self.seed), "commit", "-qm", "seed")
        self._git("-C", str(self.seed), "push", "-q", str(self.origin), "main")
        self._git("clone", "-q", "--depth", "1", f"file://{self.origin}", str(self.clone))
        self._write(self.clone, BAKE, "[new bake]\n")

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


def test_no_new_quarter_is_silent_green_and_opens_no_pr(run_step):
    assert run_step() == (0, None)
    assert not run_step.opened_pr()


def test_a_new_quarter_merged_is_silent_and_green(run_step):
    """The card just shows it; deploy-dashboard.yml deploys after a run that moved main."""
    assert run_step(log="wrote 101 quarters -> x (was 100; new: ['2026-09-30'], revised: none)\n", dirty=1) == (0, None)
    assert run_step.pushed() == ["rebake/bankruptcies-99-1"]


def test_a_bake_that_stopped_growing_pings_red_with_the_reason(run_step):
    log = "2026-03-31 page 404\nNOT GROWING: no 2026-03-31 quarter, which uscourts published months ago; ...\n"
    code, sent = run_step(log=log, outcome="failure")
    assert code != 0
    assert sent.startswith("🔴") and "did not finish (NOT GROWING: no 2026-03-31 quarter" in sent
    assert not run_step.opened_pr()


def test_a_rebuild_that_crashed_pings_red_with_its_outcome(run_step):
    code, sent = run_step(log="Traceback (most recent call last):\nKeyError: 'x'\n", outcome="failure", dirty=1)
    assert code != 0
    assert sent.startswith("🔴") and "outcome: failure" in sent and "https://run" in sent
    assert not run_step.opened_pr()


def test_a_rebuild_that_was_skipped_is_not_read_as_healthy(run_step):
    code, sent = run_step(outcome="skipped")
    assert code != 0 and sent.startswith("🔴") and "outcome: skipped" in sent


def test_a_red_ci_leaves_the_pr_open_pings_red_and_ends_red(run_step):
    code, sent = run_step(dirty=1, dash="fail")
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "CI failed" in sent and "pull/7" in sent


def test_a_merge_that_needs_a_second_try_still_merges_silently(run_step):
    assert run_step(dirty=1, merge_fails=1) == (0, None)


def test_github_refusing_the_merge_every_time_pings_red(run_step):
    code, sent = run_step(dirty=1, merge_fails=99)
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "refused" in sent and "pull/7" in sent


def test_a_merge_github_then_cannot_confirm_still_counts(run_step):
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
    """The old gap: a "NOT auto-merged" ping whenever main moved during the CI wait."""
    assert run_step(dirty=1, behind=1) == (0, None)
    assert run_step.pushed() == ["rebake/bankruptcies-99-1", "rebake/bankruptcies-99-1-2"]
    assert "pull/7" in run_step.closed()[0] and "Superseded by https://github.com/o/r/pull/8" in run_step.closed()[0]


def test_main_moving_past_every_cut_pings_red_with_the_newest_pr(run_step):
    code, sent = run_step(dirty=1, behind=3)
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "main kept moving" in sent and "pull/9" in sent
    assert len(run_step.pushed()) == 3


def test_a_re_cut_never_overwrites_a_bake_that_changed_on_main(run_step):
    code, sent = run_step(dirty=1, behind=1, main_changed_bake=1)
    assert code != 0
    assert sent.startswith("🔴") and "NOT merged" in sent and "baked file" in sent and "pull/7" in sent
    assert run_step.pushed() == ["rebake/bankruptcies-99-1"]


def test_a_bake_main_already_has_closes_its_pr_silently(run_step):
    """Main moved past the PR by taking this very bake: the owner merged an older bot PR with
    the same quarters while this one waited for CI. Nothing is left to ship or to tell him."""
    assert run_step(dirty=1, behind=1, main_has_bake=1, open="52") == (0, None)
    assert run_step.pushed() == ["rebake/bankruptcies-99-1"]
    own, stale = run_step.closed()
    assert "pull/7" in own and "already has this exact bake" in own
    assert stale.split()[2] == "52" and "newer bake" in stale


@pytest.mark.parametrize("ending", [
    dict(log="NOT GROWING: no 2026-03-31 quarter\n", outcome="failure"),   # did not finish
    dict(dirty=1, push_fails=9),                                           # opened no PR
    dict(dirty=1, dash="fail"),                                            # PR not merged
])
def test_every_red_ping_ends_with_how_to_clear_it(run_step, ending):
    """ci_health warns while this workflow's newest run on main is red, so each 🔴 says how
    to clear it. A Re-run would still use this file as it was, so it links Run workflow."""
    code, sent = run_step(**ending)
    assert code != 0 and sent.startswith("🔴") and sent.endswith("tap Run workflow: https://wf")


def test_older_bot_prs_are_closed_once_a_newer_one_merges(run_step):
    """PR #52 sat open from 8 Sep: a run that could not merge left it, and nothing closed it."""
    assert run_step(dirty=1, open="52 61") == (0, None)
    closed = run_step.closed()
    assert [c.split()[2] for c in closed] == ["52", "61"]
    assert all("Superseded by https://github.com/o/r/pull/7" in c for c in closed)


def test_older_bot_prs_stay_open_when_nothing_merged(run_step):
    code, _ = run_step(dirty=1, open="52", dash="fail")
    assert code != 0 and run_step.closed() == []


def test_a_re_run_cuts_its_own_branch(run_step):
    assert run_step(dirty=1, attempt=2) == (0, None)
    assert run_step.pushed() == ["rebake/bankruptcies-99-2"]


def test_a_github_blip_on_pr_create_is_retried(run_step):
    assert run_step(dirty=1, create_fails=1) == (0, None)


def test_a_pr_that_never_opens_pings_red_with_the_compare_link(run_step):
    code, sent = run_step(dirty=1, create_fails=9)
    assert code != 0
    assert sent.startswith("🔴") and "https://github.com/o/r/compare/rebake/bankruptcies-99-1?expand=1" in sent


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


def test_a_re_cut_with_real_git_is_the_newest_main_plus_the_bake_and_nothing_else(run_step, checkout):
    """Real git, stub GitHub: main moved with a workflow change (what GitHub's token could not
    merge into the old branch), so the re-cut must sit on the new main and add only the bake."""
    run_step.use_real_git()
    checkout.main_moves(".github/workflows/ci.yml", "v2\n")
    assert run_step(behind=1, cwd=checkout.clone) == (0, None)
    cut = "rebake/bankruptcies-99-1-2"
    assert checkout.at_origin("rev-parse", f"{cut}~1") == checkout.at_origin("rev-parse", "main")
    assert checkout.at_origin("diff", "--name-only", "main", cut).split() == [BAKE]
    assert checkout.at_origin("show", f"{cut}:{BAKE}") == "[new bake]\n"
    assert checkout.at_origin("show", f"{cut}:.github/workflows/ci.yml") == "v2\n"


def test_a_re_cut_with_real_git_leaves_a_bake_that_landed_on_main_alone(run_step, checkout):
    run_step.use_real_git()
    checkout.main_moves(BAKE, "[a hand bake]\n")
    code, sent = run_step(behind=1, cwd=checkout.clone)
    assert code != 0 and "changed the baked file" in sent
    assert checkout.at_origin("branch", "--format=%(refname:short)").split() == ["main", "rebake/bankruptcies-99-1"]


def test_a_re_cut_with_real_git_sees_that_main_already_has_the_bake(run_step, checkout):
    run_step.use_real_git()
    checkout.main_moves(BAKE, "[new bake]\n")
    assert run_step(behind=1, cwd=checkout.clone) == (0, None)
    assert "pull/7" in run_step.closed()[0] and "already has this exact bake" in run_step.closed()[0]
    assert checkout.at_origin("branch", "--format=%(refname:short)").split() == ["main", "rebake/bankruptcies-99-1"]


# ── ci.yml `report`: the dispatched result, handed to the PR ───────────────────

@pytest.mark.parametrize("backend, dashboard, posted, code", [
    ("success", "success", ["backend-tests=success", "dashboard-tests=success"], 0),
    ("success", "failure", ["backend-tests=success", "dashboard-tests=failure"], 0),
    ("cancelled", "skipped", ["backend-tests=failure", "dashboard-tests=failure"], 0),
])
def test_ci_report_posts_each_jobs_result_as_a_status(tmp_path, backend, dashboard, posted, code):
    assert _report(tmp_path, backend, dashboard) == (code, posted)


def test_ci_report_retries_a_blip_and_fails_loudly_if_it_cannot_post(tmp_path):
    assert _report(tmp_path, "success", "success", fails=1) == (0, ["backend-tests=success", "dashboard-tests=success"])
    code, posted = _report(tmp_path, "success", "success", fails=99)
    assert code != 0 and posted == []


def _report(tmp_path, backend, dashboard, fails=0):
    """Runs the report step's bash with a stub gh that records each status it is asked to post."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log, tries = tmp_path / "posted", tmp_path / "tries"
    for f in (log, tries):
        f.unlink(missing_ok=True)
    (bin_dir / "gh").write_text(r'''#!/bin/bash
n=$(cat "$TRIES" 2>/dev/null || echo 0); echo $((n + 1)) > "$TRIES"
[ "$n" -ge "$FAILS" ] || exit 1
for a in "$@"; do case "$a" in state=*) s=${a#state=} ;; context=*) c=${a#context=} ;; esac; done
case "$*" in *"statuses/abc123 "*) echo "$c=$s" >> "$POSTED" ;; esac
''')
    (bin_dir / "gh").chmod(0o755)
    (bin_dir / "sleep").write_text("#!/bin/bash\n")
    (bin_dir / "sleep").chmod(0o755)
    report = _load("ci.yml")["jobs"]["report"]["steps"][0]["run"]
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/r", "GITHUB_SHA": "abc123",
           "RUN_URL": "https://run", "BACKEND": backend, "DASHBOARD": dashboard,
           "POSTED": str(log), "TRIES": str(tries), "FAILS": str(fails)}
    proc = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", report],
                          env=env, capture_output=True, text=True, timeout=30)
    return proc.returncode, (log.read_text().split() if log.exists() else [])
