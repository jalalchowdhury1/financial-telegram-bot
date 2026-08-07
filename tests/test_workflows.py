"""Shape tests for the workflows whose ONLY output is a Telegram message.

`rebake-bankruptcies.yml` and `self-improve.yml` are both "silence is the healthy
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
    ("self-improve.yml", "self-improve"),
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
    still report when something upstream broke."""
    for filename, job in NOTIFY_WORKFLOWS:
        step = _notifying_steps(_steps(_load(filename), job))[0]
        assert str(step.get("if", "")).strip() == "always()", (
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
