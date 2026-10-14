"""Shape tests for the workflows whose ONLY output is a Telegram message.

`rebake-bankruptcies.yml` (and the retired `self-improve.yml`) are "silence is the healthy
state" jobs: nobody watches their logs, so the notify step IS the product. Both
have already been broken by the same bug — a `gh pr create` failure killing the
step before it could notify — so the guarantee is worth a regression test.

On 2026-08-05 (`self-improve`, run 31019492456) the repo setting "Allow GitHub
Actions to create and approve pull requests" was OFF. The branch pushed fine, then
`URL=$(gh pr create ... 2>&1 | tail -1)` failed; under `set -o pipefail` + GitHub's
default `bash -e` the non-zero status propagated out of the ASSIGNMENT and killed
the step, while `2>&1` had already swallowed the error text into `$URL`. Net: no
PR, no error message, and NO TELEGRAM. self-improve.yml was fixed; the identical
code in rebake-bankruptcies.yml was not — and there the notify was a *separate*
step gated on `if: env.MSG != ''` with no `if: always()`, so it could not fire in
the exact failure it exists to catch.
"""
import os
import re

import yaml

WORKFLOWS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows")


def _load(name):
    with open(os.path.join(WORKFLOWS, name)) as fh:
        return yaml.safe_load(fh)


def _steps(wf, job):
    return wf["jobs"][job]["steps"]


def _notifying_steps(steps):
    return [s for s in steps if "send_to_telegram" in (s.get("run") or "")]


def _code(run):
    """The script with comment-only lines removed. Both workflows QUOTE the old broken
    pattern in a comment so it is never reintroduced — that must not read as code."""
    return "\n".join(l for l in run.splitlines() if not l.lstrip().startswith("#"))


NOTIFY_WORKFLOWS = [
    ("rebake-bankruptcies.yml", "rebake"),
]


def test_notify_workflows_have_exactly_one_notifying_step():
    """The Telegram send must live in the SAME step that does the git/PR work.
    Split across two steps, a failure in the first skips the second."""
    for filename, job in NOTIFY_WORKFLOWS:
        steps = _steps(_load(filename), job)
        notifiers = _notifying_steps(steps)
        assert len(notifiers) == 1, f"{filename}: expected 1 notifying step, got {len(notifiers)}"


def test_the_notifying_step_runs_even_when_an_earlier_step_failed():
    """`if: always()` is the whole point — a job that only reports by Telegram must
    still report when something upstream broke. (The rebake adds "and only on main":
    see test_rebake_bankruptcies_ships_only_from_main.)"""
    for filename, job in NOTIFY_WORKFLOWS:
        step = _notifying_steps(_steps(_load(filename), job))[0]
        assert str(step.get("if", "")).strip() in ("always()", "always() && github.ref == 'refs/heads/main'"), (
            f"{filename}: the notifying step must be `if: always()`, got {step.get('if')!r}")


def test_the_notifying_step_also_creates_the_pr():
    """If PR creation lives in an earlier step, its failure kills the notify."""
    for filename, job in NOTIFY_WORKFLOWS:
        step = _notifying_steps(_steps(_load(filename), job))[0]
        assert "gh pr create" in step["run"], f"{filename}: PR creation must be in the notifying step"


def test_pr_creation_failure_cannot_kill_the_step():
    """The fix is `if PR_OUT=$(gh pr create ...); then ... else ... fi` — the exit
    code is CONSUMED by the `if`, so it never propagates. The old swallowing
    assignment must not come back."""
    for filename, job in NOTIFY_WORKFLOWS:
        run = _code(_notifying_steps(_steps(_load(filename), job))[0]["run"])
        # `if`/`elif` both consume the exit status; a bare assignment does not.
        assert re.search(r"\b(?:el)?if PR_OUT=\$\(gh pr create", run), (
            f"{filename}: `gh pr create` must be the condition of an if/elif, not a bare assignment")
        assert "URL=$(gh pr create" not in run, (
            f"{filename}: `URL=$(gh pr create ...)` under pipefail aborts the step before the notify")


def test_a_failed_pr_creation_still_hands_the_owner_a_way_in():
    """Branch is already pushed at that point, so the work is never lost — the
    message must carry the /compare link so the PR can be opened by hand."""
    for filename, job in NOTIFY_WORKFLOWS:
        run = _notifying_steps(_steps(_load(filename), job))[0]["run"]
        assert "/compare/" in run, f"{filename}: no fallback compare link in the failure message"


def test_rebake_survives_a_failing_git_push():
    """A push that fails (protected ref, token scope) has the exact same shape as a
    failing `gh pr create`: without a guard it aborts the step and nothing is sent."""
    run = _notifying_steps(_steps(_load("rebake-bankruptcies.yml"), "rebake"))[0]["run"]
    assert "if ! PUSH_OUT=$(git push" in run or "if git push" in run, (
        "rebake: `git push` must be guarded so a push failure still notifies")


def test_rebake_stays_silent_when_there_is_nothing_to_report():
    """SILENCE IS THE HEALTHY STATE: no new quarter must send no Telegram, so the
    send has to be conditional on a non-empty message INSIDE the always() step."""
    run = _notifying_steps(_steps(_load("rebake-bankruptcies.yml"), "rebake"))[0]["run"]
    assert 'if [ -n "$MSG" ]' in run, "rebake: the send must be guarded by a non-empty MSG"


def test_rebake_reports_a_rebuild_that_did_not_succeed():
    """`continue-on-error` on the rebuild means its failure is invisible unless the
    notify step checks the outcome — including `skipped`, which is not success."""
    run = _notifying_steps(_steps(_load("rebake-bankruptcies.yml"), "rebake"))[0]["run"]
    assert "steps.rebake.outcome" in run
    assert '!= "success"' in run, (
        "rebake: test the outcome against success, so `skipped`/`cancelled` are not read as healthy")


def test_deploy_dashboard_deploys_a_rebake_only_when_main_moved():
    """rebake-long-history.yml ends green when there is nothing for the owner to do, which
    includes a run that had nothing new to merge. For a workflow_run, github.sha is main's
    newest commit (run 38109724192 saw c1ed931d8 while its trigger ran on 06de367fc), so it
    differs from the rebake's own commit only when the rebake merged: no wasted deploy."""
    cond = _load("deploy-dashboard.yml")["jobs"]["deploy"]["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in cond
    assert "github.sha != github.event.workflow_run.head_sha" in cond
    # a dispatch on another branch is a dry run (rebake-long-history.yml ships only from main)
    assert "github.event.workflow_run.head_branch == 'main'" in cond


def test_a_skipped_deploy_cannot_cancel_a_waiting_one():
    """A workflow-level concurrency group is taken before the job's `if` runs, so a rebake
    that shipped nothing (deploy skipped) cancelled a dashboard push's deploy during its 90 s
    wait, and that push never deployed. Job-level, only a deploy that will run takes it."""
    wf = _load("deploy-dashboard.yml")
    assert "concurrency" not in wf
    assert wf["jobs"]["deploy"]["concurrency"] == {"group": "deploy-dashboard", "cancel-in-progress": True}


def test_rebake_long_history_ships_only_from_main():
    """Dispatched on another branch it would self-merge that branch's own commits into main,
    so there it is a dry run, like on a pull request."""
    steps = {s.get("name", ""): s for s in _load("rebake-long-history.yml")["jobs"]["rebake"]["steps"]}
    dry = next(s for n, s in steps.items() if n.startswith("Dry run"))
    bake = steps["Re-bake"]
    ship = next(s for s in steps.values() if "gh pr create" in (s.get("run") or ""))
    assert dry["if"] == "github.ref != 'refs/heads/main'"
    assert bake["if"] == "github.ref == 'refs/heads/main'"
    assert ship["if"] == "always() && github.ref == 'refs/heads/main'"


def test_rebake_bankruptcies_ships_only_from_main():
    """Dispatched on another branch (or run for a PR that changes it) the self-merge would
    carry that branch's commits into main, so there the rebuild is a dry run that fails
    the run when the rebuild failed."""
    steps = {s.get("name", ""): s for s in _steps(_load("rebake-bankruptcies.yml"), "rebake")}
    dry = next(s for n, s in steps.items() if n.startswith("Dry run"))
    ship = _notifying_steps(list(steps.values()))[0]
    assert dry["if"] == "always() && github.ref != 'refs/heads/main'"
    assert '[ "${{ steps.rebake.outcome }}" = "success" ]' in dry["run"]
    assert ship["if"] == "always() && github.ref == 'refs/heads/main'"


def test_dispatched_ci_hands_its_result_to_the_pr_as_statuses():
    """A dispatched run's checks never reach a pull request, so a bot PR stayed BLOCKED
    after its dispatched CI passed (PR #65, 11 Oct 2026). The `report` job copies each
    job's result onto the tested commit as a commit status of the same name: a bot PR then
    merged (PR #67, into a branch protected like main, merged by app/github-actions)."""
    wf = _load("ci.yml")
    report = wf["jobs"]["report"]
    assert report["if"] == "always() && github.event_name == 'workflow_dispatch'"
    assert report["needs"] == ["backend-tests", "dashboard-tests"]
    assert report["permissions"] == {"statuses": "write"}
    run = report["steps"][0]["run"]
    assert 'post backend-tests "$BACKEND"' in run and 'post dashboard-tests "$DASHBOARD"' in run
    assert 'statuses/$GITHUB_SHA' in run
    env = report["steps"][0]["env"]
    assert env["BACKEND"] == "${{ needs.backend-tests.result }}"
    assert env["DASHBOARD"] == "${{ needs.dashboard-tests.result }}"
    # the context names must be the required checks' names, i.e. the job names
    assert {"backend-tests", "dashboard-tests"} <= set(wf["jobs"])



def test_only_the_two_rebakes_can_both_start_ci_and_merge():
    """With ci.yml's `report` job, a token that can dispatch CI (`actions: write`) and merge
    (`pull-requests: write`) can land its own PR on main once the tests pass. The owner OK'd
    that for the two rebakes only (9 and 10 Oct 2026); self-improve's PRs wait for him. A
    workflow with no `permissions` block gets the repo default, read-only (checked 11 Oct)."""
    can_merge_itself = set()
    for name in os.listdir(WORKFLOWS):
        if not name.endswith((".yml", ".yaml")):
            continue
        wf = _load(name)
        for job in wf["jobs"].values():
            perms = job.get("permissions", wf.get("permissions"))    # a job's block replaces the top one
            if perms == "write-all" or (isinstance(perms, dict) and perms.get("actions") == "write"
                                        and perms.get("pull-requests") == "write"):
                can_merge_itself.add(name)
    assert can_merge_itself == {"rebake-bankruptcies.yml", "rebake-long-history.yml"}


REBAKES = ["rebake-bankruptcies.yml", "rebake-long-history.yml"]


def test_a_rebake_checks_out_the_newest_commit_of_its_ref():
    """Without a ref, checkout takes the commit the run was queued at, and a Re-run keeps it.
    A Re-run of a red run must use the fixed script and see a bake the owner merged by hand."""
    for name in REBAKES:
        checkout = next(s for s in _steps(_load(name), "rebake") if s.get("uses", "").startswith("actions/checkout@"))
        assert checkout.get("with", {}).get("ref") == "${{ github.ref }}", name


def test_each_ref_queues_its_rebake_runs_apart():
    """GitHub keeps one pending run per concurrency group and cancels the one before it: with
    one shared group, a push to a PR could cancel a pending scheduled run on main, silently."""
    for name in REBAKES:
        assert "${{ github.ref }}" in _load(name)["concurrency"]["group"], name


def test_a_bankruptcies_dry_run_is_strict():
    """The rebuild keeps each baked quarter it could not fetch, so a scraper that reads
    nothing passes a plain run. Off main (a PR, a branch) it runs --strict and fails instead."""
    rebuild = next(s for s in _steps(_load("rebake-bankruptcies.yml"), "rebake") if s.get("id") == "rebake")
    assert "${{ github.ref != 'refs/heads/main' && '--strict' || '' }}" in rebuild["run"]
