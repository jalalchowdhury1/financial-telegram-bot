# Self-Healing Robustness Loop — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **DO NOT START until Phase 1 (PR #2) is merged AND the owner has added the `ANTHROPIC_API_KEY` secret + enabled branch protection on `main`.** Those are hard prerequisites.

**Goal:** A weekly (Wednesday-morning) Claude agent, headless in GitHub Actions, that reads the accumulated health history + AGENTS.md + logs, diagnoses real problems, hunts for improvements, and **opens PRs the owner approves** — never self-merging, gated by CI.

**Architecture:** Phase 1's daily prober writes `health/history/*.json`. A new `scripts/collect_health_context.py` distills the last 7 days + unresolved findings into a single bounded digest. `.github/workflows/self-improve.yml` runs `anthropics/claude-code-action@v1` with a guardrailed prompt; the action makes changes on a branch and opens a PR. CI (Phase 1's `ci.yml`) + branch protection + the owner's merge are the gates.

**Tech Stack:** `anthropics/claude-code-action@v1` (GA), `ANTHROPIC_API_KEY`, model `claude-opus-4-8`, GitHub Actions cron, Python (digest), reuse of `bot/utils.send_to_telegram` for the summary.

**Mechanism reference (confirmed via Claude Code docs):**
- Action: `anthropics/claude-code-action@v1`. Inputs: `anthropic_api_key`, `prompt` (text or `@file`), `claude_args` (e.g. `--max-turns`, `--model`, `--allowedTools`), `base_branch`, `branch_prefix`, `track_progress`.
- The action **opens the PR itself** after editing — no separate `gh pr create` needed.
- **Cost bound:** `--max-turns N` is the primary lever (fewer turns = fewer API calls); add a job `timeout-minutes`.
- **Permissions:** needs `contents: write` + `pull-requests: write`. Branch protection on `main` prevents self-merge.
- **CI-gating gotcha:** commits made with the default `GITHUB_TOKEN` do **not** re-trigger workflows, so the agent's PR wouldn't run `ci.yml`. Fix: install the **Claude GitHub App** (or mint a GitHub App token via `actions/create-github-app-token`) so the agent's commits trigger CI. (Verify exact `claude_args` input names against the action README at execution — they evolve.)

---

## Prerequisites checklist (owner, one-time)
- [ ] Phase 1 PR #2 merged.
- [ ] Add repo secret `ANTHROPIC_API_KEY`.
- [ ] Enable **branch protection** on `main`: require PR + require the `ci.yml` checks (`backend-tests`, `dashboard-tests`) to pass before merge. This is what makes "the agent can only propose" physically true.
- [ ] Install the **Claude GitHub App** on the repo (so agent-PR commits trigger CI). If declined, accept that the owner manually re-runs CI on agent PRs.

---

## Task 1: Health-context digest

A bounded, focused input for the agent — the last 7 days of findings, deduped, with the still-unresolved ones highlighted. Keeps token cost down and the agent on-target.

**Files:**
- Create: `scripts/collect_health_context.py`
- Test: `tests/test_collect_health_context.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_health_context.py
import importlib.util, os, json

_spec = importlib.util.spec_from_file_location(
    "collect_health_context",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "collect_health_context.py"),
)
chc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chc)


def _report(date, findings):
    return {"generated_at": date, "overall": "warn", "findings": findings}


def test_digest_lists_unresolved_nonok_findings():
    history = [
        _report("2026-05-30", [{"id": "endpoint_fred", "severity": "warn", "title": "fred degraded",
                                 "detail": "429", "remediation": "manual", "evidence": {}}]),
        _report("2026-05-31", [{"id": "endpoint_fred", "severity": "warn", "title": "fred degraded",
                                 "detail": "429", "remediation": "manual", "evidence": {}}]),
    ]
    digest = chc.build_digest(history)
    assert "endpoint_fred" in digest
    assert "2 day(s)" in digest or "recurring" in digest.lower()


def test_digest_omits_consistently_ok():
    history = [_report("2026-05-31", [{"id": "endpoint_spy", "severity": "ok", "title": "ok",
                                       "detail": "", "remediation": "none", "evidence": {}}])]
    digest = chc.build_digest(history)
    assert "endpoint_spy" not in digest


def test_load_recent_history_limit(tmp_path):
    d = tmp_path / "history"
    d.mkdir()
    for day in ("2026-05-25", "2026-05-26", "2026-05-31"):
        (d / f"{day}.json").write_text(json.dumps(_report(day, [])))
    recent = chc.load_recent_history(str(d), limit=2)
    assert len(recent) == 2
    assert recent[-1]["generated_at"] == "2026-05-31"   # newest kept
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_collect_health_context.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `scripts/collect_health_context.py`**

```python
#!/usr/bin/env python3
"""Distill recent health-history into one bounded digest for the weekly self-improve agent.

Reads health/history/*.json (written by the Phase 1 daily workflow), groups non-ok
findings by id, flags recurring ones, and emits a compact Markdown digest to stdout.
"""
import glob
import json
import os
import sys


def load_recent_history(history_dir, limit=7):
    paths = sorted(glob.glob(os.path.join(history_dir, "*.json")))
    reports = []
    for p in paths[-limit:]:
        try:
            with open(p) as fh:
                reports.append(json.load(fh))
        except Exception:
            continue
    return reports


def build_digest(reports):
    by_id = {}
    for rep in reports:
        for f in rep.get("findings", []):
            if f.get("severity") == "ok":
                continue
            entry = by_id.setdefault(f["id"], {"finding": f, "days": 0})
            entry["days"] += 1
            entry["finding"] = f  # keep latest
    if not by_id:
        return "All health-history findings are OK over the window. No action needed."
    lines = ["# Health digest (unresolved findings)\n"]
    for fid, entry in sorted(by_id.items(), key=lambda kv: -kv[1]["days"]):
        f = entry["finding"]
        recurring = " (recurring)" if entry["days"] > 1 else ""
        lines.append(f"## {fid} — {f['severity']}{recurring}, seen {entry['days']} day(s)")
        lines.append(f"- {f.get('title','')}")
        if f.get("detail"):
            lines.append(f"- detail: {f['detail']}")
        if f.get("remediation") and f["remediation"] != "none":
            lines.append(f"- suggested: {f['remediation']}")
        lines.append("")
    return "\n".join(lines)


def main():
    history_dir = os.environ.get("HEALTH_HISTORY_DIR", "health/history")
    digest = build_digest(load_recent_history(history_dir))
    with open("health-digest.md", "w") as fh:
        fh.write(digest)
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_collect_health_context.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/collect_health_context.py tests/test_collect_health_context.py
git commit -m "feat(health): weekly digest of recent health-history for the agent"
```

---

## Task 2: The agent prompt (guardrails as text)

**Files:**
- Create: `.github/self-improve-prompt.md`

- [ ] **Step 1: Write the prompt**

```markdown
You are the weekly maintenance agent for the financial-telegram-bot — the owner's MOST
IMPORTANT tool. Production stability matters more than speed.

READ FIRST: `AGENTS.md` (the authoritative guide) and `health-digest.md` (this week's
unresolved health findings). Also skim recent `git log` and the newest files in
`health/history/`.

YOUR JOB, in priority order:
1. Fix the highest-severity unresolved finding in the digest that you can fix safely.
2. If the digest is clean, look for ONE worthwhile robustness improvement (a missing test,
   a fragile path, a documented known-issue in AGENTS.md §4 you can resolve).
3. If there is genuinely nothing worth doing, make NO changes and say so — do not invent work.

HARD RULES (never violate):
- Open a PULL REQUEST. NEVER push to main. NEVER merge.
- Make ONE small, single-concern change per run. No broad refactors.
- ALWAYS add or update a test for your change, and ensure `python -m pytest tests/ -q`
  passes locally before opening the PR.
- NEVER touch secrets/keys, `aws/template.yaml`, live AWS config, or `.env`.
- If a fix needs information you don't have (e.g. Google-Sheet IDs for the config.py URLS
  gap), do NOT guess — open an issue/PR comment describing exactly what's needed instead.
- When you resolve a documented known issue, update `AGENTS.md` §4 to reflect it.

PR description must: name the finding id you addressed, what you changed and why, and how
you verified it.
```

- [ ] **Step 2: Commit**

```bash
git add .github/self-improve-prompt.md
git commit -m "feat(agent): guardrailed prompt for the weekly self-improve agent"
```

---

## Task 3: The weekly workflow

**Files:**
- Create: `.github/workflows/self-improve.yml`

- [ ] **Step 1: Create the workflow**

```yaml
# yaml-language-server: $schema=https://json.schemastore.org/github-workflow
name: Weekly Self-Improve

on:
  schedule:
    - cron: '0 13 * * 3'   # Wednesday 13:00 UTC (~9 ET) — a PR waiting Wednesday morning
  workflow_dispatch:        # the owner's one-click on-demand override

permissions:
  contents: write
  pull-requests: write

concurrency:
  group: self-improve
  cancel-in-progress: false

jobs:
  self-improve:
    runs-on: ubuntu-latest
    timeout-minutes: 30     # hard wall-clock backstop
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v6
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install deps
        run: pip install -r requirements-dev.txt

      - name: Build the health digest
        run: python scripts/collect_health_context.py

      - name: Run the self-improve agent (opens a PR)
        uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: "@.github/self-improve-prompt.md"
          base_branch: main
          branch_prefix: "self-improve/"
          claude_args: |
            --max-turns 20
            --model claude-opus-4-8
            --allowedTools "Read,Write,Edit,Bash(git:*),Bash(gh:*),Bash(python:*),Bash(pytest:*)"
```

- [ ] **Step 2: Validate YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/self-improve.yml')); print('valid')"`
Expected: `valid`.

- [ ] **Step 3: Confirm action inputs against the current README** (the action evolves)

Open `https://github.com/anthropics/claude-code-action` and verify `prompt`, `claude_args`,
`base_branch`, `branch_prefix`, and the `--max-turns`/`--allowedTools` flag names still match.
Adjust if renamed. (This is a real verification step, not a placeholder — the GA API was
confirmed 2026-05 but may shift.)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/self-improve.yml
git commit -m "feat(ci): weekly self-improve agent (Wednesday) opens guardrailed PRs"
```

---

## Task 4: Telegram summary of the weekly run

Append a step that posts a short Telegram summary with the opened-PR link(s).

**Files:**
- Modify: `.github/workflows/self-improve.yml` (add a final step)

- [ ] **Step 1: Add the summary step** (after the agent step)

```yaml
      - name: Telegram summary
        if: always()
        env:
          TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          GH_TOKEN: ${{ github.token }}
        run: |
          set -uo pipefail
          PRS=$(gh pr list --search "head:self-improve/" --state open --json number,title,url \
                --jq '.[] | "PR #\(.number): \(.title)\n\(.url)"' 2>/dev/null || echo "")
          if [ -n "$PRS" ]; then
            MSG="🛠️ Weekly self-improve opened PR(s) for your review:\n$PRS"
          else
            MSG="✅ Weekly self-improve: nothing to fix this week."
          fi
          python -c "import os; from bot.utils import send_to_telegram; \
            send_to_telegram(os.environ['TELEGRAM_TOKEN'], os.environ['TELEGRAM_CHAT_ID'], \
            caption=os.environ['MSG'].replace('\\\\n','\n'))" || true
        # MSG passed via env to avoid shell-escaping the report body
```

- [ ] **Step 2: Set MSG in the env block** — replace the inline `python -c` with a two-line form that reads `$MSG`:

```yaml
        run: |
          set -uo pipefail
          PRS=$(gh pr list --search "head:self-improve/" --state open --json number,title,url \
                --jq '.[] | "PR #\(.number): \(.title) \(.url)"' 2>/dev/null || echo "")
          export MSG="🛠️ Weekly self-improve: ${PRS:-nothing to fix this week.}"
          python - <<'PY'
          import os
          from bot.utils import send_to_telegram
          send_to_telegram(os.environ['TELEGRAM_TOKEN'], os.environ['TELEGRAM_CHAT_ID'], caption=os.environ['MSG'])
          PY
```

- [ ] **Step 3: Validate YAML + commit**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/self-improve.yml')); print('valid')"`

```bash
git add .github/workflows/self-improve.yml
git commit -m "feat(ci): Telegram summary of the weekly self-improve run"
```

---

## Task 5: Dry-run verification (after prerequisites are met)

- [ ] **Step 1:** Ensure a non-ok finding exists in `health/history/` (the live system usually has one, e.g. the config_urls gap).
- [ ] **Step 2:** Trigger manually: `gh workflow run self-improve.yml --ref main`.
- [ ] **Step 3:** Watch: `gh run watch` — confirm the agent runs within budget, opens a single focused PR, and the Telegram summary arrives.
- [ ] **Step 4:** Confirm the PR triggers `ci.yml` (proves the GitHub App token wiring) and that the agent did NOT touch secrets/`template.yaml`. Review + merge if good.

---

## Self-Review (against the spec, §6 Component C)
- **Weekly Wednesday cron + workflow_dispatch override** → Task 3 (`cron: '0 13 * * 3'`).
- **Reads health history + AGENTS.md + logs** → Task 1 digest + Task 2 prompt.
- **Opens PRs, never self-merges; small single-concern; adds tests; updates AGENTS.md §4** → Task 2 prompt hard rules.
- **Hard token/cost cap** → `--max-turns 20` + `timeout-minutes: 30` (Task 3).
- **PR-only / can't reach main** → `permissions` + branch-protection prerequisite.
- **CI gates the PR** → prerequisite (Claude GitHub App so agent commits trigger `ci.yml`).
- **Telegram summary** → Task 4.
- **Forbids secrets/infra** → Task 2 prompt + `--allowedTools` excludes network/curl.
- **Honest open item:** exact `claude-code-action` input names verified 2026-05 but re-checked at execution (Task 3 Step 3).
