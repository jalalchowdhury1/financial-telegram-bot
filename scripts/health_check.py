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
# GET-able dashboard data endpoints. NOTE: /api/assessment is POST-only (needs a request
# body), so it's excluded from this GET sweep — probing it properly is a Phase 2 candidate.
ENDPOINTS = ["spy", "spy-daily-move", "market-extra", "polymarket", "fred",
             "sheets", "fear-greed"]
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
