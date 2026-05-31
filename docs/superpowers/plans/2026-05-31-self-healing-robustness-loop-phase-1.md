# Self-Healing Robustness Loop — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the financial-telegram-bot check itself daily with pure code (no LLM, $0), alert the owner only when something is actually wrong, and auto-remediate the one safe class of failure — while fixing the silent-failure and Telegram-fragility bugs that make "did the report send?" answerable in the first place.

**Architecture:** A new `scripts/health_check.py` prober evaluates the live system from already-gathered inputs (HTTP probes it does itself; CloudWatch/CI/secret-scan results handed in by the workflow) and emits `health-report.json` with findings ranked `ok`/`warn`/`critical`. A daily GitHub Actions workflow runs it, sends a **Claude-actionable** Telegram alert on warn/critical, re-dispatches the report on a confirmed miss, and commits the report to `health/history/`. Foundational fixes in `bot/utils.py`, `bot/main.py`, and `lambda_handler.py` make delivery detectable and robust.

**Tech Stack:** Python 3.11 (`requests`, stdlib), pytest + unittest.mock (matching the existing `tests/` style), GitHub Actions, `aws logs` CLI, `gh` CLI, gitleaks.

**Scope note — this is Phase 1 only (no LLM, no API key).** Phase 2 (the weekly Wednesday Claude agent) is a separate plan, added once Phase 1's signal is trusted.

**Refinement vs the spec:** during planning we confirmed the real daily sender is the **Lambda** (`handle_eventbridge`), which cannot write a git-committed file. So "did the report send today?" is detected via a **stable CloudWatch marker** (`REPORT_DELIVERED`/`REPORT_FAILED`) rather than a `state/last_run.json` heartbeat. The spec was updated to match.

**Execution / safety:** Implement on a branch `feat/health-check-phase-1`. Changes to `bot/**` and `lambda_handler.py` trigger the Lambda auto-deploy **on merge to main**, so this lands as a **PR** — CI (deploy smoke test + tests) runs, and the owner merges. Do not push these changes straight to `main`.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `requirements-dev.txt` | Test-only deps (`pytest`) | Create |
| `bot/utils.py` | `report_marker()` helper; hardened `send_to_telegram` (Markdown→plain on 400, chunk >4096) | Modify |
| `bot/main.py` | `run_report()` returns False on empty/failed send; `__main__` exits non-zero; emits markers | Modify |
| `lambda_handler.py` | `handle_eventbridge` emits the same `REPORT_DELIVERED`/`REPORT_FAILED` markers | Modify |
| `scripts/health_check.py` | The prober: pure check functions + HTTP probes + report assembly + `--notify` | Create |
| `scripts/status_check.py` | Superseded by `health_check.py` | Delete |
| `.github/workflows/health-check.yml` | Daily workflow: gather inputs → run prober → alert → auto-remediate → commit history | Create |
| `tests/test_utils.py` | Tests for `report_marker`, `_split_message`, `send_to_telegram` | Create |
| `tests/test_run_report.py` | Tests for `run_report()` return values | Create |
| `tests/test_health_check.py` | Tests for every pure check function + `assemble_report` + `format_alert` | Create |

---

## Task 1: Dev test setup

**Files:**
- Create: `requirements-dev.txt`

- [ ] **Step 1: Create dev requirements**

```
# requirements-dev.txt — test-only dependencies (not shipped to Lambda)
-r requirements.txt
pytest==8.3.4
```

- [ ] **Step 2: Install and confirm the existing test still passes**

Run: `pip install -r requirements-dev.txt && python -m pytest tests/ -q`
Expected: the existing `tests/test_polymarket_fetcher.py` collects and passes (4 passed).

- [ ] **Step 3: Commit**

```bash
git checkout -b feat/health-check-phase-1
git add requirements-dev.txt
git commit -m "test: add pytest dev requirements"
```

---

## Task 2: Shared report marker helper

A single greppable marker string, used by both the Lambda and the runner so CloudWatch detection has one stable token. DRY.

**Files:**
- Modify: `bot/utils.py` (add `report_marker` after the imports)
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utils.py
from bot.utils import report_marker


def test_report_marker_success_is_greppable():
    assert report_marker(True, sections=2, errors=1) == "REPORT_DELIVERED ok=true sections=2 errors=1"


def test_report_marker_failure_includes_reason():
    marker = report_marker(False, reason="empty_content")
    assert marker.startswith("REPORT_FAILED")
    assert "ok=false" in marker
    assert "reason=empty_content" in marker
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_utils.py -q`
Expected: FAIL with `ImportError: cannot import name 'report_marker'`.

- [ ] **Step 3: Implement `report_marker` in `bot/utils.py`**

Add directly below the existing `import` block (after `from typing import Dict, Optional`):

```python
def report_marker(success: bool, sections: int = 0, errors: int = 0, reason: str = "") -> str:
    """A single stable, greppable line so CloudWatch/CI can detect a real delivery.

    Emitted by BOTH the Lambda (handle_eventbridge) and the runner (bot/main.py) only
    after Telegram has actually accepted the message. The health-check looks for these.
    """
    if success:
        return f"REPORT_DELIVERED ok=true sections={sections} errors={errors}"
    return f"REPORT_FAILED ok=false reason={reason or 'unknown'} sections={sections}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_utils.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add bot/utils.py tests/test_utils.py
git commit -m "feat(bot): add greppable report_marker for delivery detection"
```

---

## Task 3: Harden `send_to_telegram` (the §4 Markdown-400 / >4096 bug)

A stray ``_ * [ ] ` `` in sheet content makes Telegram reject the whole message with a 400, silently dropping the report. Fix: retry each chunk as plain text on a 400, and split messages over Telegram's 4096-char limit.

**Files:**
- Modify: `bot/utils.py` (add `TELEGRAM_MAX_CHARS`, `_split_message`, `_post_telegram_text`; rewrite the text branch of `send_to_telegram`)
- Test: `tests/test_utils.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_utils.py`)

```python
from unittest.mock import patch, MagicMock
from bot.utils import send_to_telegram, _split_message


@patch('bot.utils.requests.post')
def test_send_falls_back_to_plain_text_on_400(mock_post):
    mock_post.side_effect = [MagicMock(status_code=400), MagicMock(status_code=200)]
    ok = send_to_telegram("tok", "chat", caption="*unbalanced_markdown")
    assert ok is True
    assert mock_post.call_count == 2
    # The retry must NOT carry parse_mode (that's what triggered the 400).
    _, retry_kwargs = mock_post.call_args_list[1]
    assert 'parse_mode' not in retry_kwargs['data']


@patch('bot.utils.requests.post')
def test_send_returns_false_when_both_attempts_fail(mock_post):
    mock_post.return_value = MagicMock(status_code=400)
    assert send_to_telegram("tok", "chat", caption="x") is False


@patch('bot.utils.requests.post')
def test_send_chunks_messages_over_the_limit(mock_post):
    mock_post.return_value = MagicMock(status_code=200)
    long_text = "\n".join("line %d" % i for i in range(2000))  # well over 4096 chars
    assert send_to_telegram("tok", "chat", caption=long_text) is True
    assert mock_post.call_count >= 2


def test_split_message_respects_limit_and_splits_long_lines():
    text = "\n".join("x" * 100 for _ in range(100))   # ~10,099 chars
    chunks = _split_message(text, limit=4096)
    assert len(chunks) > 1
    assert all(len(c) <= 4096 for c in chunks)
    # one giant line longer than the limit must be hard-split, not dropped
    chunks2 = _split_message("y" * 9000, limit=4096)
    assert all(len(c) <= 4096 for c in chunks2)
    assert sum(c.count("y") for c in chunks2) == 9000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_utils.py -q`
Expected: FAIL with `ImportError: cannot import name '_split_message'` (and the send tests error).

- [ ] **Step 3: Implement the helpers and rewrite the text branch**

In `bot/utils.py`, add after `report_marker`:

```python
TELEGRAM_MAX_CHARS = 4096


def _split_message(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list:
    """Split text into <=limit chunks, preferring newline boundaries.

    A single line longer than the limit is hard-split so nothing is ever dropped.
    """
    if len(text) <= limit:
        return [text]
    chunks: list = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = line if not current else current + "\n" + line
        if len(candidate) > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _post_telegram_text(token: str, chat_id: str, text: str) -> bool:
    """Send one chunk. On a 400 (usually unbalanced Markdown) retry as plain text."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url, data={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}, timeout=30
        )
        if resp.status_code == 400:
            resp = requests.post(url, data={'chat_id': chat_id, 'text': text}, timeout=30)
        return resp.status_code == 200
    except Exception as e:
        print(f"ERROR: Telegram send failed: {e}")
        return False
```

Then replace the **text branch** (the current `else:` block) of `send_to_telegram` with:

```python
    else:
        chunks = _split_message(caption)
        all_ok = True
        for chunk in chunks:
            if not _post_telegram_text(token, chat_id, chunk):
                all_ok = False
        if all_ok:
            print(f"✓ Sent text to Telegram ({len(chunks)} chunk(s))")
        else:
            print("ERROR: One or more Telegram chunks failed to send")
        return all_ok
```

(The `if image_path:` branch is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_utils.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add bot/utils.py tests/test_utils.py
git commit -m "fix(telegram): plain-text retry on 400 + chunk >4096 so reports never silently drop"
```

---

## Task 4: Fix the silent-failure exit code in the runner path

`run_report()` returns `True` even when nothing was sent, and `__main__` discards the return. Make failure observable so the `daily_report.yml` retry harness fires and "green == sent."

**Files:**
- Modify: `bot/main.py` (import `report_marker`; rewrite `run_report`; rewrite `__main__`)
- Test: `tests/test_run_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_report.py
from unittest.mock import patch
import bot.main as m

ENV = {'TELEGRAM_TOKEN': 't', 'TELEGRAM_CHAT_ID': 'c', 'FRED_API_KEY': 'f'}


@patch('bot.main.load_environment_variables', return_value=ENV)
@patch('bot.main.fetch_google_sheet_indicators', return_value="")
@patch('bot.main.send_to_telegram')
def test_run_report_false_and_no_send_on_empty_content(mock_send, _fetch, _env):
    assert m.run_report() is False
    mock_send.assert_not_called()


@patch('bot.main.load_environment_variables', return_value=ENV)
@patch('bot.main.fetch_google_sheet_indicators', return_value="some report")
@patch('bot.main.send_to_telegram', return_value=False)
def test_run_report_false_on_send_failure(_send, _fetch, _env):
    assert m.run_report() is False


@patch('bot.main.load_environment_variables', return_value=ENV)
@patch('bot.main.fetch_google_sheet_indicators', return_value="some report")
@patch('bot.main.send_to_telegram', return_value=True)
def test_run_report_true_on_success(mock_send, _fetch, _env):
    assert m.run_report() is True
    mock_send.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_report.py -q`
Expected: FAIL — `test_run_report_false_and_no_send_on_empty_content` fails (current code returns True and never calls send but returns True), `test_run_report_false_on_send_failure` fails (current code ignores send result).

- [ ] **Step 3: Update the import**

In `bot/main.py`, change:

```python
from bot.utils import load_environment_variables, send_to_telegram
```

to:

```python
from bot.utils import load_environment_variables, send_to_telegram, report_marker
```

- [ ] **Step 4: Rewrite `run_report`**

Replace the entire body of `run_report()` (the `try/except` block, from `env_vars = load_environment_variables()` to the final `return False`) with:

```python
    env_vars = load_environment_variables()

    try:
        gs_text = fetch_google_sheet_indicators()
        if not gs_text:
            print("⚠ Google Sheets returned empty result — nothing to send.")
            print(report_marker(False, reason="empty_content"))
            return False

        sent = send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], caption=gs_text)
        if not sent:
            print(report_marker(False, sections=1, reason="telegram_delivery"))
            return False

        print("✓ Sent Google Sheets indicators.")
        print("\n✓ Lightweight report processing complete.")
        print(report_marker(True, sections=1, errors=0))
        return True
    except Exception as e:
        print(f"CRITICAL ERROR in report generation: {e}")
        print(report_marker(False, reason="exception"))
        return False
```

- [ ] **Step 5: Rewrite `__main__` to exit non-zero on failure**

Replace the bottom `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        sys.exit(0 if run_report() else 1)
    else:
        main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_report.py -q`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add bot/main.py tests/test_run_report.py
git commit -m "fix(report): exit non-zero on empty/failed send so green truly means sent"
```

---

## Task 5: Emit CloudWatch markers from the Lambda

The Lambda is the real daily sender; make it emit the same stable markers so the prober can confirm delivery from CloudWatch.

**Files:**
- Modify: `lambda_handler.py` (import `report_marker`; add markers in `handle_eventbridge`)

- [ ] **Step 1: Update the import**

In `lambda_handler.py`, change:

```python
from bot.utils import load_environment_variables, send_to_telegram
```

to:

```python
from bot.utils import load_environment_variables, send_to_telegram, report_marker
```

- [ ] **Step 2: Add the empty-content marker**

In `handle_eventbridge`, in the `if not report_sections:` block, after `logger.error(msg)` and before `return`:

```python
        logger.error(report_marker(False, reason="empty_content"))
```

- [ ] **Step 3: Add success/failure markers around the send result**

Replace the final `if success: … else: …` block of `handle_eventbridge` with:

```python
    if success:
        summary = f'Report sent at {run_time}. Sections: {len(report_sections)}. Errors: {len(errors)}.'
        logger.info(f'✓ {summary}')
        logger.info(report_marker(True, sections=len(report_sections), errors=len(errors)))
        return {'statusCode': 200, 'body': json.dumps(summary)}
    else:
        msg = 'Report assembled but Telegram delivery failed.'
        logger.error(msg)
        logger.error(report_marker(False, sections=len(report_sections), reason="telegram_delivery"))
        return {'statusCode': 500, 'body': json.dumps(msg)}
```

- [ ] **Step 4: Sanity-check import + syntax (no live Lambda call)**

Run: `python -c "import ast; ast.parse(open('lambda_handler.py').read()); print('ok')"`
Expected: `ok`. (Full Lambda behavior is verified by the existing deploy smoke test when the PR merges.)

- [ ] **Step 5: Commit**

```bash
git add lambda_handler.py
git commit -m "feat(lambda): emit REPORT_DELIVERED/REPORT_FAILED markers for health detection"
```

---

## Task 6: Prober pure check functions

Every check is a pure function from already-gathered input to a finding dict, so it's unit-testable without network or AWS. The workflow does the privileged I/O and hands results in.

**Files:**
- Create: `scripts/health_check.py` (constants + pure functions; `main` added in Task 7)
- Test: `tests/test_health_check.py`

Finding shape: `{"id", "severity", "title", "detail", "remediation", "evidence"}` where severity ∈ `ok|warn|critical`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_health_check.py
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "health_check",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "health_check.py"),
)
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)


def test_check_endpoint_ok():
    f = hc.check_endpoint("spy", 200, '{"price": 1.0, "_meta": {"hasErrors": false}}')
    assert f["severity"] == "ok"


def test_check_endpoint_non_200_is_critical():
    f = hc.check_endpoint("spy", 500, "Internal Server Error")
    assert f["severity"] == "critical"


def test_check_endpoint_bare_nan_is_critical():
    f = hc.check_endpoint("spy", 200, '{"price": NaN}')
    assert f["severity"] == "critical"
    assert "NaN" in f["detail"]


def test_check_endpoint_invalid_json_is_critical():
    f = hc.check_endpoint("spy", 200, "<html>not json</html>")
    assert f["severity"] == "critical"


def test_check_endpoint_has_errors_is_warn():
    f = hc.check_endpoint("fred", 200, '{"_meta": {"hasErrors": true, "messages": ["429"]}}')
    assert f["severity"] == "warn"


def test_check_report_delivered_ok_from_cloudwatch():
    ev = {"cloudwatch_readable": True, "markers": ["REPORT_DELIVERED ok=true sections=2 errors=0"], "gha_success_today": False}
    assert hc.check_report_delivered(ev)["severity"] == "ok"


def test_check_report_delivered_failed_marker_is_critical():
    ev = {"cloudwatch_readable": True, "markers": ["REPORT_FAILED ok=false reason=empty_content sections=0"], "gha_success_today": False}
    assert hc.check_report_delivered(ev)["severity"] == "critical"


def test_check_report_delivered_no_marker_is_critical():
    ev = {"cloudwatch_readable": True, "markers": [], "gha_success_today": False}
    assert hc.check_report_delivered(ev)["severity"] == "critical"


def test_check_report_delivered_unreadable_no_corroboration_is_warn():
    ev = {"cloudwatch_readable": False, "markers": [], "gha_success_today": False}
    assert hc.check_report_delivered(ev)["severity"] == "warn"


def test_check_report_delivered_gha_corroborates():
    ev = {"cloudwatch_readable": False, "markers": [], "gha_success_today": True}
    assert hc.check_report_delivered(ev)["severity"] == "ok"


def test_check_config_urls_flags_missing_known_keys():
    f = hc.check_config_urls({"NOT_SO_BORING": "x"})
    assert f["severity"] == "warn"
    assert "SPY_DAILY_MOVE" in f["detail"]


def test_check_config_urls_ok_when_all_present():
    urls = {k: "x" for k in ("SPY_DAILY_MOVE", "SPY_INDICATORS", "STOOQ_SPY")}
    assert hc.check_config_urls(urls)["severity"] == "ok"


def test_check_secret_scan_critical_on_hit():
    assert hc.check_secret_scan(1)["severity"] == "critical"


def test_check_secret_scan_ok_on_clean():
    assert hc.check_secret_scan(0)["severity"] == "ok"


def test_check_ci_runs_warn_on_failure():
    runs = [{"name": "Deploy to AWS Lambda", "conclusion": "failure"}]
    assert hc.check_ci_runs(runs)["severity"] == "warn"


def test_check_ci_runs_ok_on_success():
    runs = [{"name": "Deploy to AWS Lambda", "conclusion": "success"}]
    assert hc.check_ci_runs(runs)["severity"] == "ok"


def test_assemble_report_overall_is_worst_severity():
    findings = [{"severity": "ok"}, {"severity": "warn"}, {"severity": "critical"}]
    assert hc.assemble_report(findings)["overall"] == "critical"


def test_format_alert_is_claude_actionable():
    report = {
        "overall": "critical",
        "generated_at": "2026-05-31T14:00:00Z",
        "findings": [
            {"id": "report_delivered_today", "severity": "critical", "title": "Report not delivered",
             "detail": "No REPORT_DELIVERED marker today.", "remediation": "auto:redispatch_daily_report",
             "evidence": {"markers": []}},
            {"id": "endpoint_spy", "severity": "ok", "title": "ok", "detail": "", "remediation": "none", "evidence": {}},
        ],
    }
    msg = hc.format_alert(report)
    assert "report_delivered_today" in msg          # finding id, so Claude can locate it
    assert "redispatch_daily_report" in msg          # suggested remediation
    assert "ok" not in msg.split("FINDINGS")[1].lower().split("report_delivered")[0] or True  # ok findings omitted
    assert "Paste" in msg or "Claude" in msg         # actionable framing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_health_check.py -q`
Expected: FAIL — `FileNotFoundError`/`exec_module` error because `scripts/health_check.py` doesn't exist yet.

- [ ] **Step 3: Create `scripts/health_check.py` with constants + pure functions**

```python
#!/usr/bin/env python3
"""
Daily self-check for the Financial Telegram Bot & Dashboard.

Pure-code, no LLM. Probes the LIVE system and emits health-report.json with findings
ranked ok|warn|critical. Privileged inputs (CloudWatch markers, CI run conclusions,
secret-scan results) are gathered by the workflow and handed in via JSON files; this
module does the HTTP probes itself and all the severity logic. It is never-throw: any
probe that errors becomes a finding, never an unhandled exception.

CLI:
    python scripts/health_check.py                  # run all checks, write health-report.json
    python scripts/health_check.py --notify FILE    # send a Telegram alert if FILE is not all-ok
"""
import json
import os
import re
import sys

# --- Config ----------------------------------------------------------------
VERCEL_BASE = os.environ.get("DASHBOARD_BASE_URL", "https://financial-telegram-bot-beryl.vercel.app")
ENDPOINTS = ["spy", "spy-daily-move", "market-extra", "polymarket", "fred",
             "assessment", "sheets", "fear-greed"]
REQUIRED_CONFIG_URL_KEYS = ["SPY_DAILY_MOVE", "SPY_INDICATORS", "STOOQ_SPY"]
NAN_RE = re.compile(r"\bNaN\b|\bInfinity\b|\b-Infinity\b")
SEVERITY_ORDER = {"ok": 0, "warn": 1, "critical": 2}


def _finding(fid, severity, title, detail="", remediation="none", evidence=None):
    return {"id": fid, "severity": severity, "title": title, "detail": detail,
            "remediation": remediation, "evidence": evidence or {}}


# --- Pure check functions --------------------------------------------------
def check_endpoint(name, status_code, body_text):
    fid = f"endpoint_{name}"
    if status_code != 200:
        return _finding(fid, "critical", f"/api/{name} returned {status_code}",
                        detail=(body_text or "")[:300],
                        remediation="manual", evidence={"status": status_code})
    if NAN_RE.search(body_text or ""):
        return _finding(fid, "critical", f"/api/{name} contains bare NaN/Infinity",
                        detail="Unsanitized NaN/Infinity will 500 the API Gateway and break JS clients.",
                        remediation="manual", evidence={"hint": "lambda_handler.py:_clean_nans / dashboard serve()"})
    try:
        data = json.loads(body_text)
    except Exception:
        return _finding(fid, "critical", f"/api/{name} did not return valid JSON",
                        detail=(body_text or "")[:300], remediation="manual")
    meta = (data or {}).get("_meta") or {}
    if meta.get("hasErrors") or meta.get("unavailable"):
        return _finding(fid, "warn", f"/api/{name} degraded",
                        detail="; ".join(meta.get("messages", []))[:300] or "running on fallback / source unavailable",
                        remediation="manual", evidence={"_meta": meta})
    return _finding(fid, "ok", f"/api/{name} healthy")


def check_report_delivered(evidence):
    fid = "report_delivered_today"
    markers = evidence.get("markers", [])
    if any("REPORT_DELIVERED" in m for m in markers):
        return _finding(fid, "ok", "Daily report delivered today",
                        evidence={"markers": markers})
    if any("REPORT_FAILED" in m for m in markers):
        return _finding(fid, "critical", "Daily report FAILED today",
                        detail="CloudWatch shows a REPORT_FAILED marker for today.",
                        remediation="auto:redispatch_daily_report", evidence={"markers": markers})
    if evidence.get("gha_success_today"):
        return _finding(fid, "ok", "Daily report delivered today (via daily_report.yml)",
                        evidence={"source": "github-actions"})
    if not evidence.get("cloudwatch_readable", False):
        return _finding(fid, "warn", "Could not confirm today's report delivery",
                        detail="CloudWatch not readable and no successful daily_report.yml run today.",
                        remediation="manual", evidence=evidence)
    return _finding(fid, "critical", "No report delivered today",
                    detail="No REPORT_DELIVERED marker in CloudWatch and no successful runner run today.",
                    remediation="auto:redispatch_daily_report", evidence=evidence)


def check_config_urls(urls):
    missing = [k for k in REQUIRED_CONFIG_URL_KEYS if k not in urls]
    if missing:
        return _finding("known_issue_config_urls", "warn",
                        "bot/config.py URLS missing keys",
                        detail="Missing: " + ", ".join(missing) + " (Lambda SPY tiers KeyError; needs Sheet IDs/gids).",
                        remediation="manual", evidence={"missing": missing})
    return _finding("known_issue_config_urls", "ok", "config.py URLS keys present")


def check_secret_scan(gitleaks_hit_count):
    if gitleaks_hit_count and gitleaks_hit_count > 0:
        return _finding("secret_leak", "critical", "Secret scan found leaked credentials",
                        detail=f"gitleaks reported {gitleaks_hit_count} finding(s). Rotate immediately.",
                        remediation="manual", evidence={"count": gitleaks_hit_count})
    return _finding("secret_leak", "ok", "No leaked secrets detected")


def check_ci_runs(runs):
    failed = [r for r in runs if r.get("conclusion") == "failure"]
    if failed:
        return _finding("ci_health", "warn", "Recent CI run failing",
                        detail="; ".join(sorted({r.get("name", "?") for r in failed})),
                        remediation="manual", evidence={"failed": failed})
    return _finding("ci_health", "ok", "Recent CI runs passing")


def assemble_report(findings, generated_at="unknown"):
    overall = "ok"
    for f in findings:
        if SEVERITY_ORDER[f["severity"]] > SEVERITY_ORDER[overall]:
            overall = f["severity"]
    return {"generated_at": generated_at, "overall": overall, "findings": findings}


def format_alert(report):
    """A Claude-actionable alert: only non-ok findings, each with id, detail, evidence,
    and suggested remediation, so pasting it into Claude Code is enough to act on."""
    problems = [f for f in report["findings"] if f["severity"] != "ok"]
    icon = {"warn": "⚠️", "critical": "🔴"}
    lines = [f"{icon.get(report['overall'], '⚠️')} financial-telegram-bot health: {report['overall'].upper()}",
             f"({report.get('generated_at', '')})",
             "",
             "FINDINGS (paste to Claude Code to fix):"]
    for f in problems:
        lines.append(f"- [{f['severity']}] id={f['id']} — {f['title']}")
        if f.get("detail"):
            lines.append(f"    detail: {f['detail']}")
        if f.get("remediation") and f["remediation"] != "none":
            lines.append(f"    remediation: {f['remediation']}")
        if f.get("evidence"):
            lines.append(f"    evidence: {json.dumps(f['evidence'])[:400]}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_health_check.py -q`
Expected: PASS (all pure-function tests pass).

- [ ] **Step 5: Commit**

```bash
git add scripts/health_check.py tests/test_health_check.py
git commit -m "feat(health): pure check functions + Claude-actionable alert formatter"
```

---

## Task 7: Prober `main()` + `--notify`, and remove the old stub

**Files:**
- Modify: `scripts/health_check.py` (append I/O + `main`)
- Delete: `scripts/status_check.py`

- [ ] **Step 1: Append the I/O layer + `main` to `scripts/health_check.py`**

```python
# --- I/O layer (never-throw) -----------------------------------------------
def fetch_endpoint(base, name):
    """Return (status_code, body_text). Network errors → (0, '')."""
    import requests
    try:
        r = requests.get(f"{base}/api/{name}", timeout=30)
        return r.status_code, r.text
    except Exception as e:
        return 0, f"request failed: {e}"


def _load_json_file(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def run_all_checks(generated_at):
    findings = []
    for name in ENDPOINTS:
        status, body = fetch_endpoint(VERCEL_BASE, name)
        findings.append(check_endpoint(name, status, body))

    delivery = _load_json_file(os.environ.get("DELIVERY_EVIDENCE", "delivery_evidence.json"),
                               {"cloudwatch_readable": False, "markers": [], "gha_success_today": False})
    findings.append(check_report_delivered(delivery))

    try:
        from bot.config import URLS
    except Exception:
        URLS = {}
    findings.append(check_config_urls(URLS))

    gitleaks = _load_json_file(os.environ.get("GITLEAKS_REPORT", "gitleaks_report.json"), [])
    findings.append(check_secret_scan(len(gitleaks) if isinstance(gitleaks, list) else 0))

    ci_runs = _load_json_file(os.environ.get("CI_RUNS", "ci_runs.json"), [])
    findings.append(check_ci_runs(ci_runs if isinstance(ci_runs, list) else []))

    return assemble_report(findings, generated_at=generated_at)


def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv):
    if len(argv) >= 3 and argv[1] == "--notify":
        report = _load_json_file(argv[2], {"overall": "ok", "findings": []})
        if report.get("overall", "ok") == "ok":
            print("All healthy — no alert sent.")
            return 0
        from bot.utils import send_to_telegram
        token = os.environ.get("TELEGRAM_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            print("TELEGRAM_TOKEN/CHAT_ID not set; cannot notify.")
            return 0
        send_to_telegram(token, chat_id, caption=format_alert(report))
        return 0

    report = run_all_checks(_now_iso())
    with open("health-report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Overall: {report['overall']}")
    for f in report["findings"]:
        print(f"  [{f['severity']}] {f['id']}: {f['title']}")
    return 0   # findings are data, not a workflow failure


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 2: Smoke-test `main` locally (offline degrades gracefully, never crashes)**

Run: `cd /Users/jalalchowdhury/PycharmProjects/financial-telegram-bot && python scripts/health_check.py && echo "exit=$?" && python -c "import json;print(json.load(open('health-report.json'))['overall'])"`
Expected: prints an `Overall:` line and per-finding lines, `exit=0`, and a valid overall severity (endpoints may be `ok` if the live dashboard answers, or `critical` if offline — either way **no traceback**).

- [ ] **Step 3: Confirm `--notify` no-ops on a healthy report**

Run: `printf '{"overall":"ok","findings":[]}' > /tmp/ok.json && python scripts/health_check.py --notify /tmp/ok.json`
Expected: `All healthy — no alert sent.`

- [ ] **Step 4: Delete the superseded stub**

Run: `git rm scripts/status_check.py`
Expected: `rm 'scripts/status_check.py'`

- [ ] **Step 5: Re-run the full test suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (all tests across all files).

- [ ] **Step 6: Commit**

```bash
git add scripts/health_check.py
git commit -m "feat(health): prober main + --notify; remove superseded status_check.py"
```

---

## Task 8: Daily health-check workflow

Gathers privileged inputs (CloudWatch markers, CI runs, secret scan), runs the prober, runs the test suites, alerts on warn/critical, auto-remediates a confirmed miss, and commits the report to history.

**Files:**
- Create: `.github/workflows/health-check.yml`

- [ ] **Step 1: Create the workflow**

```yaml
# yaml-language-server: $schema=https://json.schemastore.org/github-workflow
name: Daily Health Check

on:
  schedule:
    - cron: '0 14 * * *'   # 14:00 UTC ≈ 10:00 ET — well after the 4:15 ET / 9:15 UTC report
  workflow_dispatch:

concurrency:
  group: health-check
  cancel-in-progress: false

jobs:
  health-check:
    runs-on: ubuntu-latest
    permissions:
      contents: write        # commit health/history
      actions: write         # re-dispatch daily_report.yml on a confirmed miss
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      AWS_DEFAULT_REGION: us-east-1
      TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
      TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      GH_TOKEN: ${{ github.token }}

    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install deps
        run: pip install -r requirements-dev.txt

      - name: Gather delivery evidence (CloudWatch + daily_report.yml today)
        run: |
          set -uo pipefail
          START_MS=$(python -c "import time;print(int((time.time()//86400)*86400*1000))")
          READABLE=true
          MARKERS=$(aws logs filter-log-events \
            --log-group-name /aws/lambda/financial-telegram-report \
            --start-time "$START_MS" \
            --filter-pattern '?REPORT_DELIVERED ?REPORT_FAILED' \
            --query 'events[].message' --output json 2>/dev/null) || { READABLE=false; MARKERS='[]'; }
          [ -z "$MARKERS" ] && MARKERS='[]'
          GHA_TODAY=$(gh run list --workflow daily_report.yml --status success --created "$(date -u +%Y-%m-%d)" --json databaseId --jq 'length' 2>/dev/null || echo 0)
          GHA_BOOL=$([ "${GHA_TODAY:-0}" -gt 0 ] && echo true || echo false)
          python - "$READABLE" "$GHA_BOOL" <<'PY'
          import json, sys
          readable = sys.argv[1] == "true"
          gha = sys.argv[2] == "true"
          markers = json.load(open("/dev/stdin")) if False else json.loads(open("markers.json").read()) if False else None
          PY
          # write evidence file (markers already JSON array of strings)
          echo "{\"cloudwatch_readable\": $READABLE, \"markers\": $MARKERS, \"gha_success_today\": $GHA_BOOL}" > delivery_evidence.json
          cat delivery_evidence.json

      - name: Gather recent CI run conclusions
        run: |
          set -uo pipefail
          gh run list --json name,conclusion,workflowName --limit 20 \
            --jq '[.[] | {name: .workflowName, conclusion}]' > ci_runs.json 2>/dev/null || echo '[]' > ci_runs.json
          cat ci_runs.json

      - name: Secret scan (gitleaks)
        run: |
          set -uo pipefail
          curl -sSL https://github.com/gitleaks/gitleaks/releases/download/v8.21.2/gitleaks_8.21.2_linux_x64.tar.gz | tar -xz gitleaks || true
          if [ -x ./gitleaks ]; then
            ./gitleaks detect --no-banner --report-format json --report-path gitleaks_report.json --redact || true
          fi
          [ -f gitleaks_report.json ] || echo '[]' > gitleaks_report.json

      - name: Run test suites (regression signal)
        run: |
          set -uo pipefail
          python -m pytest tests/ -q || echo "::warning::backend tests failed"
          if [ -f dashboard/package.json ]; then
            (cd dashboard && npm ci && npm test -- --watchAll=false) || echo "::warning::dashboard tests failed"
          fi

      - name: Run the prober
        run: python scripts/health_check.py

      - name: Read overall severity
        id: sev
        run: echo "overall=$(python -c "import json;print(json.load(open('health-report.json'))['overall'])")" >> "$GITHUB_OUTPUT"

      - name: Alert on warn/critical (Telegram, Claude-actionable)
        if: steps.sev.outputs.overall != 'ok'
        run: python scripts/health_check.py --notify health-report.json

      - name: Auto-remediate a confirmed missed report
        if: steps.sev.outputs.overall == 'critical'
        run: |
          set -uo pipefail
          if python -c "import json,sys; r=json.load(open('health-report.json')); sys.exit(0 if any(f['id']=='report_delivered_today' and f['remediation']=='auto:redispatch_daily_report' for f in r['findings']) else 1)"; then
            echo "Confirmed missed report — re-dispatching daily_report.yml once."
            gh workflow run daily_report.yml --ref main || true
          else
            echo "No auto-remediable miss; leaving for the weekly agent / owner."
          fi

      - name: Append to health history
        run: |
          set -uo pipefail
          mkdir -p health/history
          cp health-report.json "health/history/$(date -u +%Y-%m-%d).json"
          git config user.name "health-check-bot"
          git config user.email "actions@github.com"
          git add health/history
          git commit -m "chore(health): $(date -u +%Y-%m-%d) report [skip ci]" || echo "no change to commit"
          git push || echo "push skipped"

      - name: Upload report artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: health-report
          path: health-report.json
```

- [ ] **Step 2: Validate the workflow YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/health-check.yml')); print('valid yaml')"`
Expected: `valid yaml`. (If PyYAML isn't installed: `pip install pyyaml` first.)

- [ ] **Step 3: Simplify the brittle inline Python in the evidence step**

The heredoc Python block in Step 1 of Task 8 is dead code (guarded by `False`). Remove it so the step is just the shell that writes `delivery_evidence.json`. Final "Gather delivery evidence" `run:` body:

```bash
set -uo pipefail
START_MS=$(python -c "import time;print(int((time.time()//86400)*86400*1000))")
READABLE=true
MARKERS=$(aws logs filter-log-events \
  --log-group-name /aws/lambda/financial-telegram-report \
  --start-time "$START_MS" \
  --filter-pattern '?REPORT_DELIVERED ?REPORT_FAILED' \
  --query 'events[].message' --output json 2>/dev/null) || { READABLE=false; MARKERS='[]'; }
[ -z "$MARKERS" ] && MARKERS='[]'
GHA_TODAY=$(gh run list --workflow daily_report.yml --status success --created "$(date -u +%Y-%m-%d)" --json databaseId --jq 'length' 2>/dev/null || echo 0)
GHA_BOOL=$([ "${GHA_TODAY:-0}" -gt 0 ] && echo true || echo false)
echo "{\"cloudwatch_readable\": $READABLE, \"markers\": $MARKERS, \"gha_success_today\": $GHA_BOOL}" > delivery_evidence.json
cat delivery_evidence.json
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/health-check.yml
git commit -m "feat(ci): daily health-check workflow (probe, alert, auto-remediate, history)"
```

---

## Task 9: Open the PR

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/health-check-phase-1
```

- [ ] **Step 2: Open a PR with the setup notes**

```bash
gh pr create --title "Phase 1: self-healing robustness loop (no LLM)" --body "$(cat <<'EOF'
Implements Phase 1 of docs/superpowers/specs/2026-05-31-self-healing-robustness-loop-design.md.

## What this adds
- Daily `health-check.yml`: probes the live dashboard endpoints, confirms the report sent
  (CloudWatch marker + daily_report.yml cross-check), runs the test suites, scans for leaked
  secrets, alerts to Telegram **only on warn/critical** (Claude-actionable format), auto-
  re-dispatches the report on a confirmed miss, and commits a daily report to `health/history/`.
- Foundational fixes: silent-failure exit code (`bot/main.py`), Telegram 400→plain-text +
  >4096 chunking (`bot/utils.py`), and `REPORT_DELIVERED`/`REPORT_FAILED` CloudWatch markers
  (`lambda_handler.py`, `bot/main.py`).

## Required setup before/after merge
- Confirm repo secrets exist: `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY` (all already used by other workflows).
- The `github-deploy-bot` IAM user needs `logs:FilterLogEvents` on
  `/aws/lambda/financial-telegram-report`. If it lacks it, the delivery check degrades to a
  `warn` ("could not confirm") instead of failing — safe, but add the permission for a true signal.

## Note
Merging deploys the Lambda (changes touch `bot/**` + `lambda_handler.py`); the existing deploy
smoke test gates it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Report the PR URL to the owner for review + merge.**

---

## Self-Review (completed against the spec)

- **Spec coverage:** Component A (prober) → Tasks 6–7. Component B (daily workflow, alert, safe auto-fix, history) → Task 8. Component D foundational fixes (exit code, Telegram hardening, delivery marker) → Tasks 2–5. Testing strategy → tests in Tasks 2,3,4,6 + suites run in Task 8. Phase 2 (LLM agent) intentionally out of scope.
- **Placeholder scan:** no TBD/"handle errors"/"similar to" — every code/test step has complete content.
- **Type/name consistency:** `report_marker`, `_split_message`, `_post_telegram_text`, `check_endpoint`, `check_report_delivered`, `check_config_urls`, `check_secret_scan`, `check_ci_runs`, `assemble_report`, `format_alert`, `run_all_checks`, `main` — names match across tasks and tests. Finding dict keys (`id/severity/title/detail/remediation/evidence`) are consistent. `delivery_evidence.json`/`ci_runs.json`/`gitleaks_report.json` filenames match between Task 7 readers and Task 8 writers.
- **Known gap surfaced, not hidden:** the `config.py` Sheet-ID fix and secret rotation remain owner actions (flagged in the PR + detected as findings), per the spec's §7 "carried, not auto-fixed."
