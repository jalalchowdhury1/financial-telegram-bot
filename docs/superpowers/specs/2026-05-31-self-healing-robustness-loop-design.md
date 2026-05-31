# Design: Self-Healing Robustness Loop

**Date:** 2026-05-31
**Status:** Approved (design); pending implementation plan
**Owner:** jalal.chowdhury@gmail.com
**Repo:** `jalalchowdhury1/financial-telegram-bot`

> **Note on this doc's lifecycle.** Per `AGENTS.md`, point-in-time specs under
> `docs/superpowers/specs/` are not kept long-term — `AGENTS.md` is the single source of
> truth. This spec drives the implementation plan; once the work ships, its durable facts
> (how the health-check, daily workflow, and weekly agent operate) get folded into
> `AGENTS.md` and this file can be deleted.

---

## 1. Problem & motivation

The system (one AWS Lambda doing the daily Telegram report + the dashboard data API, plus
the Next.js dashboard on Vercel) is the owner's **most important tool**. It has accrued a
lot of hard-won resilience (never-throw dashboard routes, `/tmp` yfinance caching, NaN
sanitization, FRED freshness logic, deploy concurrency-guard + smoke test). But:

- Robustness improvements happen **reactively and manually** — a human notices something
  broke, diagnoses it, fixes it (exactly the work captured in `AGENTS.md` §4 and recent
  history). There is no programmatic loop that does this continuously.
- The system can **fail silently**. The clearest example, confirmed in code:
  `bot/main.py`'s `__main__` calls `run_report()` and **discards its return value** (no
  `sys.exit(1)`), and `run_report()` returns `True` even when
  `fetch_google_sheet_indicators()` returns empty and **nothing was sent to Telegram**.
  Result: a day with no content → green workflow → `/api/last-run` reports "all good" →
  **the owner gets no report and no alert.** So "did the report actually send?" is not even
  reliably answerable today.
- `scripts/status_check.py` exists but is trivial — it checks only that env vars are
  *present* and prints the git remote. It verifies nothing about whether the live system
  actually works.

**Goal:** make the system check itself daily, alert the owner the moment something is
wrong, auto-fix the safe/known failure classes deterministically, and — once a week — have
a Claude agent diagnose the harder problems and *hunt for new improvements*, opening PRs
the owner approves. Production stays gated behind CI + human merge.

## 2. Goals / non-goals

**Goals**
- Daily, deterministic, zero-cost self-check of the **live** system end-to-end.
- Immediate notification (Telegram) on any real problem; **silence when healthy**.
- Deterministic auto-remediation for well-understood, safe failure classes.
- A weekly LLM agent that diagnoses + writes fixes + opens PRs, plus discovers new
  robustness improvements.
- Strict separation: **plain code where reliability matters, LLM only where judgment
  matters, gates in between.** The LLM can *propose*, never *impose*.

**Non-goals**
- No fully-autonomous self-merge/self-deploy. The owner holds the merge gate.
- No LLM use for "simple stuff" (presence checks, "did it send", re-triggering a failed
  run) — those stay in plain code.
- No infra-as-code changes applied automatically (`aws/template.yaml` is reference-only and
  not applied by CI; the agent must not touch it or live AWS config).
- The agent never touches secrets.

## 3. Architecture

```
┌─ 1. DETECTION (daily) ─────────────┐   PURE CODE — deterministic, free, never flaky
│  scripts/health_check.py probes    │   • emits health-report.json (findings by severity)
│  the LIVE system end-to-end        │   • itself never-throw: a broken probe = warn, not crash
└───────────────┬────────────────────┘
                │  all green → silent.  warn/critical → Telegram alert + safe auto-fix.
                │  every run appended to health/history/ for the weekly agent.
                ▼
┌─ 2. DIAGNOSE + FIX (weekly, Wed AM) ┐  LLM (Claude, headless in GitHub Actions)
│  reads health history + AGENTS.md + │   • root-cause + writes fix + test → opens a PR
│  logs + git; opens PR(s); updates   │   • hunts for new improvements
│  AGENTS.md §4 when it fixes a known │   • healthy week → no-op, near-zero spend
│  issue                              │   • + one-click "Run workflow" override button
└───────────────┬────────────────────┘
                ▼
┌─ 3. GATE ───────────────────────────┐  PURE CODE + HUMAN
│  CI (pytest + jest + live smoke      │   • deterministic pass/fail
│  test) must pass; branch protection  │   • owner reviews + merges
│  on main; owner merges               │   • merge → existing auto-deploy + smoke test
└─────────────────────────────────────┘
```

**Cadence**
- **Daily** (pure code, $0): health-check → alert + safe auto-fix → append history.
- **Weekly, Wednesday morning** (LLM): self-improve agent opens PRs.
- **On-demand**: manual "Run workflow" button on the weekly workflow = the owner's override.

## 4. Component A — Health prober (`scripts/health_check.py`)

Pure Python. Supersedes `scripts/status_check.py` (which is removed). Produces a structured
report and is **itself never-throw**: any probe that errors becomes a finding, never an
unhandled exception that aborts the run.

### Finding model
Each check yields a finding:
```json
{
  "id": "report_delivered_today",
  "severity": "ok | warn | critical",
  "title": "Daily report delivered today",
  "detail": "Heartbeat at 2026-05-31T09:16Z; /api/last-run agrees.",
  "remediation": "auto:redispatch_daily_report | manual | none",
  "evidence": { "...": "..." }
}
```
Report: `{ "generated_at": "...", "overall": "ok|warn|critical", "findings": [ ... ] }`
written to `health-report.json` (artifact) and appended to `health/history/<date>.json`.

### Checks (each → a finding)
1. **Report delivered today** — read `state/last_run.json` *heartbeat* (written only on
   confirmed Telegram delivery; see Component D). Cross-check `/api/last-run`. `critical`
   if no heartbeat for today after report time.
2. **Dashboard `/api/*` endpoints** — for each of `spy`, `spy-daily-move`, `market-extra`,
   `polymarket`, `fred`, `assessment`, `sheets`, `fear-greed`, via the real Vercel base
   URL: HTTP 200, valid JSON, **no bare `NaN`/`Infinity`** in the body,
   `_meta.hasErrors !== true`. `critical` on 500/invalid-JSON/NaN; `warn` on
   `hasErrors`/degraded.
3. **"Green but on fallback"** — inspect `_meta` / `stale` / `unavailable` flags to detect
   the dashboard serving last-known-good or fallback instead of live data. `warn`.
4. **Known §4 issues** — probe each so regressions *and fixes* are visible:
   missing `config.py` `URLS` keys (`SPY_DAILY_MOVE`, `SPY_INDICATORS`, `STOOQ_SPY`);
   silent-failure exit code; Telegram-Markdown fragility. Each `warn`/`ok` as appropriate.
5. **Secret-leak scan** — run `gitleaks` (pinned) over the repo. Any hit = `critical`.
6. **CI / deploy health** — via `gh api`, confirm the most recent `deploy-lambda.yml` and
   `daily_report.yml` runs concluded `success`. `warn`/`critical` otherwise.
7. **Test suites** — run `pytest -q` and (in the dashboard) `npm test` as a cheap daily
   regression signal. Failures = `warn` (a fix-worthy signal for the weekly agent).

Config (endpoint list, Vercel base URL, freshness deadlines, repo slug) lives in a small
constants block at the top of the script, not hard-coded throughout.

## 5. Component B — Daily workflow (`.github/workflows/health-check.yml`)

- **Trigger:** `schedule` (cron) shortly *after* the morning report time, plus
  `workflow_dispatch`. Keepalive-safe (this repo is on GitHub Actions cron; see the GHA
  cron hardening playbook — pinned `actions/*@v6`, etc.).
- **Steps:** checkout → setup Python + Node → install deps → run `scripts/health_check.py`
  → branch on `overall`:
  - **`ok`** → do nothing (no Telegram message; silence == healthy).
  - **`warn`/`critical`** → send **one** concise Telegram alert summarizing the findings.
  - **Safe auto-remediation** for known classes, deterministic, no LLM. Initial set:
    - report not delivered today → re-dispatch `daily_report.yml` once.
    (Auto-fixes are conservative and individually logged; anything not on the safe list is
    left for the weekly agent / the owner.)
  - Always upload `health-report.json` as an artifact and **append** to `health/history/`
    (committed via a `[skip ci]` commit, or stored as an artifact — see Open Items §10).
- **Never-throw:** the workflow must not fail the run just because a probe found a problem;
  a non-green *finding* is data, not a workflow error. (The job only "fails" on
  infrastructure errors.)

## 6. Component C — Weekly self-improve agent (`.github/workflows/self-improve.yml`)

- **Trigger:** `schedule` Wednesday morning (cron) + `workflow_dispatch` (the one-click
  override button).
- **Runtime:** Claude Code headless via GitHub Actions, authenticated with the
  `ANTHROPIC_API_KEY` repo secret. Strong model (configurable; quality matters, frequency
  is low).
- **Inputs the agent reads:** `AGENTS.md`, the week's `health/history/`, current
  `health-report.json`, unresolved findings, recent failed-run logs (`gh api` /
  CloudWatch where available), and `git log`.
- **Behavior:** for each real problem or worthwhile improvement: create a branch, write the
  **fix + a test**, open a **PR** with a clear description tying it to the finding; update
  `AGENTS.md` §4 when it resolves a known issue. A healthy week with nothing worth doing →
  **no-op** (near-zero spend). Posts a Telegram summary with PR links.
- **Guardrails (hard requirements):**
  - **Hard token/cost cap** per run (fail-safe stop).
  - **PRs only** — workflow token scoped to `contents: write` (branches) +
    `pull-requests: write`; **branch protection on `main`** so it physically cannot merge
    its own PR.
  - Every PR must pass existing CI (pytest + jest + the live API-Gateway smoke test) to be
    mergeable. **Owner merges.** Merge → existing auto-deploy + smoke test.
  - **Prompt forbids:** touching secrets/keys, editing `aws/template.yaml` or live AWS
    config, large refactors. Small, single-concern PRs only.
  - Scoped, read-mostly tooling; no destructive shell.

## 7. Component D — Foundational fixes (Phase 1, no LLM, done during build)

The detection layer is only honest if these hold, so they are fixed up front:

1. **Silent-failure exit code** (`bot/main.py`): `run_report()` returns `False` on *any*
   failure, **including empty/missing content** (today it returns `True` after sending
   nothing). `__main__` does `sys.exit(1)` on `False`. Now "workflow green == report
   sent," and the `daily_report.yml` retry harness actually fires on real failures.
2. **Telegram delivery hardening** (`bot/utils.py:send_to_telegram`): on a `parse_mode`
   400, **retry as plain text**; **chunk** messages > 4096 chars. A stray ``_ * [ ] ` ``
   in sheet content can no longer silently drop the whole report.
3. **Delivery heartbeat** (`state/last_run.json`): written **only after** Telegram confirms
   `ok:true`. This is the prober's source of truth for "did it send today."

**Carried, not auto-fixed** (handed to the owner as a checklist; agent forbidden from
secrets):
- **`config.py` missing Sheet URL keys** — needs the owner's Google-Sheet IDs/gids. Leave a
  clearly-marked TODO; the weekly agent surfaces it but cannot invent the IDs.
  (`STOOQ_SPY = https://stooq.com/q/d/l/?s=spy.us&i=d` is known.)
- **Rotate the leaked RapidAPI key** (it's in git history) and **deactivate/delete the old
  root AWS access keys** (CI already uses the `github-deploy-bot` IAM user). Console-only;
  owner action.

## 8. Phasing

- **Phase 1 — no LLM, no API key needed:** Component A (prober) + Component B (daily
  workflow + alerts + safe auto-fix) + Component D foundational fixes + tests. Delivers the
  robustness + self-checking immediately and risk-free. Run ~1 week; read real reports.
- **Phase 2 — the LLM:** Component C (Wednesday agent) + `ANTHROPIC_API_KEY` secret +
  branch protection on `main`. Added once Phase 1's signal is trusted.

## 9. Testing strategy

- **Foundational fixes:** unit tests — `run_report()` returns `False` on empty content and
  on send failure; `__main__` exits non-zero; `send_to_telegram` falls back to plain text on
  a simulated 400 and chunks > 4096; heartbeat written only on confirmed send.
- **Prober:** unit tests with mocked HTTP — each check maps inputs → correct severity; the
  prober never raises (inject a probe that throws → becomes a `warn` finding).
- **Workflows:** validate YAML; dry-run the daily workflow via `workflow_dispatch` against
  the live system and confirm the alert path + the silent-when-healthy path.
- **Weekly agent:** first run via the manual button, watched, on a week with a known issue
  present — confirm it opens a sensible PR and respects the guardrails.

## 10. Open items / decisions

- **Decided:** alerts fire only on `warn`/`critical` (silence == healthy); branch
  protection on `main` enabled in Phase 2.
- **History storage:** `health/history/` committed via `[skip ci]` vs. run artifacts +
  a small rolling summary. Lean toward committed JSON (the weekly agent reads it trivially);
  finalize in the plan.
- **Alert destination:** same Telegram chat as the market report (only-on-problem, so low
  noise) vs. a dedicated chat. Default: same chat; revisit if noisy.
- **Agent model + token cap values:** finalize in the plan.

## 11. Success criteria

- A silent report failure (empty content or a Telegram 400) now (a) does **not** go green
  and (b) produces a same-day Telegram alert.
- The daily health-check runs every day, costs $0, stays silent when healthy, and alerts on
  real problems.
- The Wednesday agent opens at least one sensible, CI-passing PR when a real issue or
  improvement exists, no-ops on a clean week, and never reaches `main` without the owner's
  merge.
- `AGENTS.md` §4 shrinks over time as the loop resolves known issues.
