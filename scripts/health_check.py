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

# Make the repo root importable so `from bot.* import ...` works when this is run as
# `python scripts/health_check.py` (Python puts scripts/ on sys.path, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Config ----------------------------------------------------------------
VERCEL_BASE = os.environ.get("DASHBOARD_BASE_URL", "https://financial-telegram-bot-beryl.vercel.app")
# GET-able dashboard data endpoints. NOTE: /api/assessment is POST-only (needs a request
# body), so it's excluded from this GET sweep — probing it properly is a Phase 2 candidate.
ENDPOINTS = ["spy", "spy-daily-move", "market-extra", "polymarket", "fred",
             "sheets", "fear-greed", "vol"]
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


# Indicators that are N/A ON PURPOSE — their upstream series is dead/frozen, so they are
# NEVER alarmed (alarming them is the "crying wolf" we must avoid). Keyed by the field name
# in /api/fred's `indicators`. NOTE: `lei` is here only until it's replaced by a live,
# multi-sourced indicator (owner decision 2026-05-31) — remove it once that ships.
KNOWN_DISCONTINUED = {
    "lei": "USSLIND (Leading Economic Index) — discontinued/frozen by FRED since 2020",
}


# Cards that live at the ROOT of /api/fred rather than under `indicators`/`checklist`.
# Like the horsemen they expose `current`, not `value`.
TOP_LEVEL_FRED_CARDS = ["yieldCurve", "profitMargin", "spEps"]


def fred_metrics_for_na_check(fred):
    """Collect every dashboard metric the N/A sweep should inspect: the indicator
    tiles, the bull-checklist metrics, the Four Horsemen series, and the top-level
    cards. Horsemen and top-level entries use `current` rather than `value` (they
    carry chart history), so they are mapped into the common
    {value, unavailable, staleDays, asOf} shape — an unexpectedly N/A or overdue
    metric (e.g. the uscourts bankruptcies feed drifting after a file-naming
    change) then warns like any other indicator.

    `yieldCurve`, `profitMargin` and `spEps` were OUTSIDE this sweep until
    2026-08-06. That mattered most for yieldCurve, which IS the 10Y-2Y horseman
    (AGENTS §3: the card reuses `fred.yieldCurve` instead of duplicating it under
    `horsemen`). If FRED froze T10Y2Y and the repair cascade adopted nothing, the
    route's `failed` list stayed empty, `hasErrors` stayed false and endpoint_fred
    stayed green forever — while claims and unemployment, the two lines drawn
    right next to it, would have warned via staleDays > 3.

    (`peRatio` is deliberately NOT here: it is covered by check_pe_source, which
    reports the more specific CAPE-for-TTM substitution.)
    """
    metrics = {**(fred.get("indicators") or {}), **(fred.get("checklist") or {})}
    for hk, hv in (fred.get("horsemen") or {}).items():
        if isinstance(hv, dict):
            metrics[f"horsemen_{hk}"] = {**hv, "value": hv.get("current")}
    for key in TOP_LEVEL_FRED_CARDS:
        card = fred.get(key)
        if isinstance(card, dict):
            metrics[key] = {**card, "value": card.get("current")}
    return metrics


def check_pe_source(fred):
    """The P/E tile is a layered scrape: multpl -> Yahoo -> FRED `PE10`. `PE10` is
    Shiller CAPE, a 10-yr smoothed ratio (~40) — NOT the trailing-twelve-month P/E
    (~30) the tile is labelled as. Tier 2 (Yahoo) is dead from Vercel datacenter IPs,
    so a multpl markup change drops STRAIGHT to CAPE and overstates the tile by ~40%
    while `peRatioAsOf` still reads `now`. Nothing surfaced that substitution before
    2026-08-06: `peSource` was computed and discarded, and the P/E sits outside the
    indicator sweep, so `endpoint_fred` stayed green."""
    fid = "pe_source_substituted"
    source = (fred or {}).get("peSource")
    if source is None:
        # Payload predates the peSource field (deploy lag) — unknowable, not known-bad.
        return _finding(fid, "ok", "P/E source not reported (payload predates peSource)")
    if source == "cape":
        return _finding(
            fid, "warn", "P/E tile is showing Shiller CAPE, not TTM",
            detail=(f"peSource='cape' (FRED PE10, 10-yr smoothed) is standing in for the "
                    f"trailing-twelve-month P/E the tile claims; current value "
                    f"{(fred or {}).get('peRatio')}. The multpl scrape failed and the Yahoo "
                    f"tier is dead from Vercel. Repair the multpl regex in "
                    f"dashboard/app/api/fred/route.js."),
            remediation="manual", evidence={"peSource": source, "peRatio": (fred or {}).get("peRatio")})
    return _finding(fid, "ok", f"P/E from {source} (trailing-twelve-month)",
                    evidence={"peSource": source})


# The four Lambda-primary routes (AGENTS §1): each hits API Gateway -> Lambda first and
# falls back to direct public sources. Every OTHER /api/* route is dashboard-only.
LAMBDA_PRIMARY_ROUTES = ["spy", "market-extra", "spy-daily-move", "polymarket"]

# Every dashboard fallback builder labels itself "... (fallback)" — verified against the
# live endpoints 2026-08-06 by injecting ?_fail=lambda:
#   spy            -> "Polygon + Finnhub (fallback)" | "Polygon (fallback)" | "Yahoo Finance (fallback)"
#   market-extra   -> "Direct sources (fallback)"
#   spy-daily-move -> "Finnhub (fallback)" | "Polygon (fallback)" | "Yahoo Finance (fallback)"
#   polymarket     -> "Polymarket Gamma API (fallback)"
# The Lambda's own labels (bot/fetchers.py: "<tier> + Finnhub Spot",
# "yfinance/Polygon/Finnhub/FRED/ER-API", "Google Sheets") never contain it, so this one
# marker separates the two paths for all four routes without a per-route allowlist that
# would rot the next time a waterfall tier is renamed.
FALLBACK_SOURCE_MARKER = "(fallback)"


def _route_source(payload):
    """The winning-source label, wherever the route puts it. spy/market-extra use
    `_meta.source`; spy-daily-move/polymarket carry a TOP-LEVEL `source` and no `_meta`."""
    if not isinstance(payload, dict):
        return None
    meta = payload.get("_meta")
    if isinstance(meta, dict) and isinstance(meta.get("source"), str):
        return meta["source"]
    src = payload.get("source")
    return src if isinstance(src, str) else None


def check_lambda_path(payloads):
    """Is the Lambda still serving the dashboard's HTTP data path, or is every route
    quietly running on its fallback?

    WHY THIS EXISTS. /api/spy, /api/market-extra, /api/polymarket and /api/spy-daily-move
    are Lambda-primary. When the Lambda does not answer they fall back to direct public
    sources and return `hasErrors:false` ON PURPOSE — a full direct build IS healthy data
    (hard-coding hasErrors:true once made SPY perpetually red). The Lambda failure is
    recorded only as a `_meta.messages` string that nothing reads. And scripts/health_check
    probes Vercel, never the Lambda. So if the Lambda lost its apigateway invoke grant —
    the EXACT failure mode that hid on the EventBridge grant for two months (AGENTS §2
    gotcha #0) — all four routes would switch to fallbacks and every green check would
    stay green.

    A green check must prove the PRIMARY path ran. Here that means: the route's winning
    source is a Lambda label, not a dashboard fallback label.

    Unreadable payloads are SKIPPED, not counted against the Lambda: an endpoint that is
    down is already `endpoint_<name>` critical, and unknowable must never become
    known-bad (same rule as CloudWatch in check_report_delivered).
    """
    fid = "lambda_primary_path"
    fallback, lambda_served, sources = [], [], {}
    for name in LAMBDA_PRIMARY_ROUTES:
        source = _route_source((payloads or {}).get(name))
        if source is None:
            continue
        sources[name] = source
        if FALLBACK_SOURCE_MARKER in source.lower():
            fallback.append(name)
        else:
            lambda_served.append(name)

    if fallback:
        fell = ", ".join(f"{n} ({sources[n]})" for n in fallback)
        if lambda_served:
            # The Lambda answered OTHER routes in this same run, so the gateway hop and
            # the invoke grant are fine -- it's this route's handler. In practice that
            # means it ran past API Gateway's 30 s integration cap (gateway answers 503,
            # Vercel falls back): 2026-09-01, /api/market-extra took 49-59 s while
            # /api/spy took 0.7 s. Don't send the reader to the resource policy.
            title = "Dashboard route(s) on FALLBACK — a Lambda handler is failing or too slow"
            detail = (
                f"Served from direct sources instead: {fell}. But the Lambda IS reachable — "
                f"it served {', '.join(lambda_served)} in this same run — so this is NOT the "
                "API Gateway invoke grant or LAMBDA_URL. The handler behind the fallen-back "
                "route is erroring or exceeding API Gateway's 30 s integration cap (the "
                "gateway then answers 503 and Vercel falls back). Check CloudWatch for that "
                "route: 'Trigger: HTTP /api/<route>' → the REPORT line's Duration (>30000 ms "
                "= timeout) and any upstream failures logged between them.")
        else:
            title = "Dashboard is running on FALLBACKS — the Lambda data path is down"
            detail = (
                f"These Lambda-primary routes served from direct sources instead: {fell}. "
                "The dashboard still shows correct numbers, so nothing looks broken — "
                "but the API Gateway -> Lambda hop is failing. Check the Lambda's "
                "resource policy for an apigateway.amazonaws.com invoke grant "
                "(AGENTS §2 'Wiring a new API Gateway'), that LAMBDA_URL still points "
                "at the gateway base, and the function's recent CloudWatch errors.")
        return _finding(
            fid, "warn", title, detail=detail, remediation="manual",
            evidence={"fallback_routes": fallback, "lambda_routes": lambda_served, "sources": sources})

    if not sources:
        # Nothing readable — the endpoint probes own that failure, not this check.
        return _finding(fid, "ok", "Lambda data path not checkable (no readable route payload)")

    return _finding(fid, "ok", f"Lambda serving {len(lambda_served)}/{len(sources)} primary routes",
                    evidence={"sources": sources})


def check_indicators_na(metrics):
    """Flag dashboard metrics that are N/A (fetch failed) or genuinely overdue
    (stale > 3 days past their deadline). Normal reporting lag (a value that is present
    and stale <= 3 days, or not stale at all) is fine and never alarmed. A metric on the
    KNOWN_DISCONTINUED allowlist is expected — reported ok with a note."""
    if not isinstance(metrics, dict):
        return _finding("indicators_na", "warn", "Could not read dashboard indicators",
                        detail="/api/fred returned no metrics object to inspect.",
                        remediation="manual")
    unexpected, expected_na = [], []
    for key, ind in metrics.items():
        if not isinstance(ind, dict):
            continue
        overdue = (ind.get("staleDays") or 0) > 3
        missing = ind.get("unavailable") is True or ind.get("value") is None
        if not (overdue or missing):
            continue
        if key in KNOWN_DISCONTINUED:
            expected_na.append((key, KNOWN_DISCONTINUED[key]))
        elif missing:
            unexpected.append((key, "N/A (fetch failed)"))
        else:
            unexpected.append((key, f"overdue {ind.get('staleDays')}d past deadline (asOf={ind.get('asOf')})"))
    if unexpected:
        return _finding("indicators_na", "warn", "Dashboard indicator(s) N/A or overdue",
                        detail="; ".join(f"{k}: {why}" for k, why in unexpected),
                        remediation="manual",
                        evidence={"unexpected": dict(unexpected), "expected_na": dict(expected_na)})
    note = "; ".join(f"{k} ({why})" for k, why in expected_na) or "all indicators fresh or within lag"
    return _finding("indicators_na", "ok", "Dashboard indicators present (fresh or within lag)",
                    detail=f"Expected-N/A: {note}", evidence={"expected_na": dict(expected_na)})


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
        # A delivered report is NOT proof the system is healthy. There are two
        # senders: the Lambda (primary, EventBridge ~08:15 UTC) and the runner
        # (backstop, daily_report.yml ~09:45 UTC). If CloudWatch was readable and
        # showed no Lambda marker, the primary did NOT deliver and the backstop
        # covered — the owner still gets the report, so this is a warn, not a
        # critical, but it MUST NOT read as `ok`. Returning `ok` here is why the
        # EventBridge outage of 2026-06-01 → 2026-08-06 stayed invisible for two
        # months (the recreated Lambda lost its invoke grant AND its Telegram env
        # vars; AWS/Events FailedInvocations sat at 1/day the whole time).
        if evidence.get("cloudwatch_readable", False):
            return _finding(
                fid, "warn", "Report delivered by BACKSTOP only — Lambda path is down",
                detail=("No REPORT_DELIVERED marker from the Lambda in the last 24h, but "
                        "daily_report.yml succeeded. The EventBridge→Lambda primary path is "
                        "failing and the runner is masking it. Check the rule's AWS/Events "
                        "FailedInvocations metric, the Lambda's resource policy (needs an "
                        "events.amazonaws.com invoke grant) and its env vars "
                        "(TELEGRAM_TOKEN/TELEGRAM_CHAT_ID)."),
                remediation="manual", evidence=evidence)
        # CloudWatch unreadable → the Lambda path is unknowable, not known-bad.
        return _finding(fid, "ok", "Daily report delivered today (via daily_report.yml)",
                        detail="CloudWatch unreadable, so the Lambda path could not be checked.",
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
    """Healthy = each workflow's LATEST run passed. A workflow that failed earlier but
    whose newest run succeeded is healthy (don't nag about already-fixed history). `runs`
    should already be limited to ACTIVE workflows (the gather step drops deleted one-offs);
    each item has name + conclusion + (optional) createdAt. In-progress runs are ignored."""
    latest = {}
    for r in runs:
        name = r.get("name", "?")
        if not r.get("conclusion"):       # skip in-progress / queued (no conclusion yet)
            continue
        cur = latest.get(name)
        if cur is None or (r.get("createdAt", "") >= cur.get("createdAt", "")):
            latest[name] = r
    failing = sorted(n for n, r in latest.items() if r.get("conclusion") == "failure")
    if failing:
        return _finding("ci_health", "warn", "A workflow's latest run is failing",
                        detail="; ".join(failing),
                        remediation="manual", evidence={"failing": failing})
    return _finding("ci_health", "ok", "Active workflows' latest runs passing")


def assemble_report(findings, generated_at="unknown"):
    overall = "ok"
    for f in findings:
        if SEVERITY_ORDER[f["severity"]] > SEVERITY_ORDER[overall]:
            overall = f["severity"]
    return {"generated_at": generated_at, "overall": overall, "findings": findings}


def _plain_english(f):
    """A non-technical, reassuring one-liner per finding, so a scary 'CRITICAL' doesn't read
    as 'the world is on fire' when it's usually something the system handles itself."""
    fid = f.get("id", "")
    auto = (f.get("remediation") or "").startswith("auto:")
    if fid == "report_delivered_today":
        return ("Today's market report didn't send on its own — but the system is auto-re-sending "
                "it, so you'll still get it. Only worry if your report never shows up."
                if auto else
                "Today's market report may not have gone out — check whether it arrived.")
    if fid.startswith("endpoint_"):
        return (f"A dashboard data feed ({fid[len('endpoint_'):]}) hiccuped. The dashboard falls "
                "back to backups and keeps working — this is just a heads-up; it usually self-clears.")
    if fid == "indicators_na":
        return "A dashboard number is blank that normally has data. Worth a peek, not an emergency."
    if fid == "lambda_primary_path":
        return ("The dashboard is quietly running on its backup data sources — the numbers you see "
                "are still right, but the main AWS engine behind them isn't answering. Worth fixing "
                "before something else leans on it.")
    if fid == "known_issue_config_urls":
        return "A small, already-known config gap. Not new and not urgent."
    if fid == "secret_leak":
        return "⚠️ A password / API key may be exposed in the code. THIS one matters — rotate it soon."
    if fid == "ci_health":
        return "An automated workflow's last run failed. Worth a look, usually a quick fix."
    return "Something to glance at — paste this to Claude and it'll handle it."


def format_alert(report):
    """A Claude-actionable alert: only non-ok findings, each with id, detail, evidence, and
    suggested remediation, so pasting it into Claude Code is enough to act on. Each finding
    also gets a plain-English '💬' line, and the alert ends with an ELI10 BOTTOM LINE saying
    whether the system is self-healing this or it genuinely needs you."""
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
        lines.append(f"    💬 {_plain_english(f)}")

    # ELI10 bottom line — is this "the system's got it" or "this actually needs you"?
    secret = any(f["id"] == "secret_leak" for f in problems)
    all_auto = bool(problems) and all((f.get("remediation") or "").startswith("auto:") for f in problems)
    if secret:
        bottom = "⚠️ BOTTOM LINE: mostly routine, but a possible leaked key needs your attention soon."
    elif all_auto:
        bottom = ("🟢 BOTTOM LINE: the system is already auto-fixing this — you almost certainly don't "
                  "need to do anything (just confirm your report arrived). Not a major issue.")
    else:
        bottom = ("🟡 BOTTOM LINE: likely minor and often self-healing — not an emergency. If you want "
                  "it handled, paste this whole message to Claude.")
    lines += ["", "─" * 28, bottom]
    return "\n".join(lines)


def format_summary(report):
    """A positive 'all green' confirmation when healthy, otherwise the actionable alert.
    Used by the --summary mode so a manual run always shows the owner a clean check."""
    findings = report.get("findings", [])
    total = len(findings)
    ok = sum(1 for f in findings if f.get("severity") == "ok")
    if report.get("overall") == "ok":
        return (f"✅ financial-telegram-bot health: ALL GREEN — {ok}/{total} checks ok "
                f"({report.get('generated_at', '')})")
    return format_alert(report)


# --- I/O layer (never-throw) -----------------------------------------------
# Cold-start tolerance: a cold Vercel/Lambda stack can be slow or 5xx on the FIRST hit.
# We warm it up once, then retry each probe with backoff, so a slow first load is never
# mistaken for an outage. Tunable via env without a code change.
PROBE_TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "45"))
PROBE_ATTEMPTS = int(os.environ.get("PROBE_ATTEMPTS", "3"))
PROBE_BACKOFF = (3, 10, 20)


def warm_up(base):
    """Wake a cold stack once before probing so the first real probe isn't a false
    failure. Best-effort; errors are ignored."""
    import requests
    try:
        requests.get(f"{base}/api/spy", timeout=max(PROBE_TIMEOUT, 60))
    except Exception:
        pass


def _body_is_degraded(text):
    """True if a 200 body reports itself degraded (_meta.hasErrors/unavailable). These are
    usually transient source hiccups (a feed momentarily rate-limited) that clear on a
    re-fetch, so we retry them rather than alarm on the first blip."""
    try:
        meta = (json.loads(text) or {}).get("_meta") or {}
        return bool(meta.get("hasErrors") or meta.get("unavailable"))
    except Exception:
        return False


def fetch_endpoint(base, name, attempts=None, sleeper=None):
    """Return (status_code, body_text), retrying transient problems so neither a cold
    start, a slow first load, NOR a momentary degraded response is mistaken for a real
    fault. Returns immediately on the first HTTP 200 that is NOT self-reported-degraded;
    otherwise retries with backoff and returns the last result, so a genuinely-down or
    persistently-degraded endpoint is still flagged."""
    import requests
    import time
    attempts = attempts or PROBE_ATTEMPTS
    sleep = sleeper or time.sleep
    last = (0, "")
    for i in range(attempts):
        try:
            r = requests.get(f"{base}/api/{name}", timeout=PROBE_TIMEOUT)
            last = (r.status_code, r.text)
            if r.status_code == 200 and not _body_is_degraded(r.text):
                return last
        except Exception as e:
            last = (0, f"request failed: {e}")
        if i < attempts - 1:
            sleep(PROBE_BACKOFF[min(i, len(PROBE_BACKOFF) - 1)])
    return last


def _load_json_file(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def run_all_checks(generated_at):
    findings = []
    warm_up(VERCEL_BASE)   # absorb the cold-start cost once before probing
    route_payloads = {}    # name -> parsed body, for the Lambda-path check below
    for name in ENDPOINTS:
        status, body = fetch_endpoint(VERCEL_BASE, name)
        findings.append(check_endpoint(name, status, body))
        if status == 200:
            try:
                route_payloads[name] = json.loads(body)
            except Exception:
                pass
        # /api/fred carries the dashboard's economic indicators — inspect each for an
        # unexpected N/A (a real break) vs an expected one (discontinued series).
        if name == "fred" and status == 200:
            try:
                fred = json.loads(body) or {}
                # Sweep the indicator tiles, bull-checklist metrics AND the Four
                # Horsemen series (see fred_metrics_for_na_check).
                metrics = fred_metrics_for_na_check(fred)
            except Exception:
                metrics = None
                fred = {}
            findings.append(check_indicators_na(metrics))
            # peRatio is top-level, so it is NOT in the sweep above — check it here or a
            # CAPE-for-TTM substitution stays invisible (see check_pe_source).
            findings.append(check_pe_source(fred))

    # The four Lambda-primary routes all degrade SILENTLY to direct sources, so the
    # only way to see the Lambda's HTTP path die is to read which source won.
    findings.append(check_lambda_path(route_payloads))

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
    # --notify FILE  → send ONLY if not all-ok (scheduled runs: silence == healthy)
    # --summary FILE → ALWAYS send (manual runs: a clean ✅ green check, or the alert)
    if len(argv) >= 3 and argv[1] in ("--notify", "--summary"):
        report = _load_json_file(argv[2], {"overall": "ok", "findings": []})
        if argv[1] == "--notify" and report.get("overall", "ok") == "ok":
            print("All healthy — no alert sent.")
            return 0
        from bot.utils import send_to_telegram
        token = os.environ.get("TELEGRAM_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            print("TELEGRAM_TOKEN/CHAT_ID not set; cannot notify.")
            return 0
        send_to_telegram(token, chat_id, caption=format_summary(report))
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
