import importlib.util
import json
import os
import subprocess
import sys

_spec = importlib.util.spec_from_file_location(
    "health_check",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "health_check.py"),
)
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)


def test_notify_runs_as_script_without_bot_import_error(tmp_path):
    """Regression: running `python scripts/health_check.py --notify <warn report>` must
    resolve `from bot.utils import ...` (repo root on sys.path). With no TELEGRAM env it
    no-ops the send and exits 0 — but it MUST reach that import without ModuleNotFoundError."""
    report = {"overall": "warn", "findings": [
        {"id": "x", "severity": "warn", "title": "t", "detail": "", "remediation": "manual", "evidence": {}}]}
    f = tmp_path / "r.json"
    f.write_text(json.dumps(report))
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "health_check.py")
    env = {k: v for k, v in os.environ.items() if not k.startswith("TELEGRAM")}
    proc = subprocess.run([sys.executable, script, "--notify", str(f)],
                          capture_output=True, text=True, env=env)
    assert "No module named 'bot'" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stderr


def test_check_endpoint_ok():
    f = hc.check_endpoint("spy", 200, '{"price": 1.0, "_meta": {"hasErrors": false}}')
    assert f["severity"] == "ok"


def test_check_endpoint_non_200_is_critical():
    f = hc.check_endpoint("spy", 500, "Internal Server Error")
    assert f["severity"] == "critical"


def test_check_endpoint_bare_nan_is_critical():
    f = hc.check_endpoint("spy", 200, '{"price": NaN}')
    assert f["severity"] == "critical"
    assert "NaN" in f["title"] or "NaN" in f["detail"]


def test_check_endpoint_invalid_json_is_critical():
    f = hc.check_endpoint("spy", 200, "<html>not json</html>")
    assert f["severity"] == "critical"


def test_check_endpoint_has_errors_is_warn():
    f = hc.check_endpoint("fred", 200, '{"_meta": {"hasErrors": true, "messages": ["429"]}}')
    assert f["severity"] == "warn"


def test_check_indicators_na_lei_is_expected_not_alarmed():
    indicators = {
        "lei": {"value": None, "asOf": "2020-02-01", "stale": True, "unavailable": False},
        "sentiment": {"value": 49.8, "stale": False, "unavailable": False},
    }
    f = hc.check_indicators_na(indicators)
    assert f["severity"] == "ok"
    assert "lei" in f["evidence"]["expected_na"]


def test_check_indicators_na_unexpected_null_is_warn():
    indicators = {
        "sentiment": {"value": None, "asOf": "2026-04-01", "stale": True, "unavailable": False},
        "lei": {"value": None, "stale": True, "unavailable": False},
    }
    f = hc.check_indicators_na(indicators)
    assert f["severity"] == "warn"
    assert "sentiment" in f["detail"]      # the unexpected one is named
    assert "lei" not in f["detail"]        # the discontinued one is NOT alarmed


def test_check_indicators_na_unavailable_is_warn():
    indicators = {"claims": {"value": None, "unavailable": True}}
    f = hc.check_indicators_na(indicators)
    assert f["severity"] == "warn"
    assert "claims" in f["detail"]


def test_check_indicators_na_all_present_is_ok():
    indicators = {"claims": {"value": 209, "stale": False, "unavailable": False}}
    assert hc.check_indicators_na(indicators)["severity"] == "ok"


def test_check_indicators_na_missing_object_is_warn():
    assert hc.check_indicators_na(None)["severity"] == "warn"


def test_check_indicators_na_overdue_more_than_3_days_is_warn():
    indicators = {"m2": {"value": 2.1, "asOf": "2026-04-01", "stale": True,
                         "unavailable": False, "staleDays": 5}}
    f = hc.check_indicators_na(indicators)
    assert f["severity"] == "warn"
    assert "m2" in f["detail"]


def test_check_indicators_na_stale_within_3_days_is_ok():
    indicators = {"m2": {"value": 2.1, "asOf": "2026-04-01", "stale": True,
                         "unavailable": False, "staleDays": 2}}
    assert hc.check_indicators_na(indicators)["severity"] == "ok"


def test_check_indicators_na_sweeps_checklist_keys():
    # checklist-style metric (durable) overdue -> warn
    metrics = {"durable": {"value": 1.0, "asOf": "2026-03-01", "stale": True,
                           "unavailable": False, "staleDays": 10}}
    f = hc.check_indicators_na(metrics)
    assert f["severity"] == "warn"
    assert "durable" in f["detail"]


def test_check_report_delivered_ok_from_cloudwatch():
    ev = {"cloudwatch_readable": True, "markers": ["REPORT_DELIVERED ok=true sections=2 errors=0"], "gha_success_today": False}
    assert hc.check_report_delivered(ev)["severity"] == "ok"


def test_check_report_delivered_failed_marker_is_critical():
    ev = {"cloudwatch_readable": True, "markers": ["REPORT_FAILED ok=false reason=empty_content sections=0"], "gha_success_today": False}
    f = hc.check_report_delivered(ev)
    assert f["severity"] == "critical"
    assert f["remediation"] == "auto:redispatch_daily_report"


def test_check_report_delivered_no_marker_is_critical():
    ev = {"cloudwatch_readable": True, "markers": [], "gha_success_today": False}
    assert hc.check_report_delivered(ev)["severity"] == "critical"


def test_check_report_delivered_unreadable_no_corroboration_is_warn():
    ev = {"cloudwatch_readable": False, "markers": [], "gha_success_today": False}
    assert hc.check_report_delivered(ev)["severity"] == "warn"


def test_check_report_delivered_gha_corroborates():
    # CloudWatch unreadable → we cannot tell whether the Lambda delivered, so a
    # successful runner run is the only evidence available and must stay `ok`
    # (never a false alarm on a missing IAM grant).
    ev = {"cloudwatch_readable": False, "markers": [], "gha_success_today": True}
    assert hc.check_report_delivered(ev)["severity"] == "ok"


def test_check_report_delivered_backstop_only_warns_primary_is_down():
    """CloudWatch readable + NO Lambda marker + runner succeeded = the backstop
    is silently covering for a dead primary. This exact state ran undetected
    from 2026-06-01 to 2026-08-06 (the recreated Lambda lost its EventBridge
    invoke grant) because this case returned `ok`."""
    ev = {"cloudwatch_readable": True, "markers": [], "gha_success_today": True}
    f = hc.check_report_delivered(ev)
    assert f["severity"] == "warn"
    assert "backstop" in f["title"].lower() or "backstop" in f["detail"].lower()


def test_check_pe_source_ok_when_ttm():
    assert hc.check_pe_source({"peRatio": 29.69, "peSource": "multpl"})["severity"] == "ok"


def test_check_pe_source_warns_when_cape_substituted():
    """FRED PE10 is Shiller CAPE (10-yr smoothed, ~40) standing in for the TTM P/E
    (~30) the tile claims to show. Tier 2 (Yahoo) is dead from Vercel, so a multpl
    HTML change drops straight to CAPE and overstates the tile ~40% with asOf=now."""
    f = hc.check_pe_source({"peRatio": 42.12, "peSource": "cape", "peIsCape": True})
    assert f["severity"] == "warn"
    assert "cape" in f["detail"].lower()


def test_check_pe_source_silent_on_old_payload_without_field():
    """A deploy lag where /api/fred predates the peSource field must not false-alarm."""
    assert hc.check_pe_source({"peRatio": 29.69})["severity"] == "ok"


def test_check_config_urls_flags_missing_known_keys():
    f = hc.check_config_urls({"NOT_SO_BORING": "x"})
    assert f["severity"] == "warn"
    assert "SPY_DAILY_MOVE" in f["detail"]


def test_check_config_urls_ok_when_all_present():
    urls = {k: "x" for k in ("SPY_DAILY_MOVE", "SPY_INDICATORS")}
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


def test_check_ci_runs_ignores_fixed_history():
    # An older failure + a newer success for the SAME workflow → healthy.
    runs = [
        {"name": "CI", "conclusion": "success", "createdAt": "2026-06-01T02:00:00Z"},
        {"name": "CI", "conclusion": "failure", "createdAt": "2026-05-31T18:00:00Z"},
    ]
    assert hc.check_ci_runs(runs)["severity"] == "ok"


def test_check_ci_runs_flags_latest_failure_only():
    runs = [
        {"name": "Deploy", "conclusion": "failure", "createdAt": "2026-06-01T02:00:00Z"},
        {"name": "Deploy", "conclusion": "success", "createdAt": "2026-05-31T18:00:00Z"},
    ]
    f = hc.check_ci_runs(runs)
    assert f["severity"] == "warn"
    assert f["evidence"]["failing"] == ["Deploy"]


def test_check_ci_runs_ignores_in_progress():
    runs = [
        {"name": "X", "conclusion": None, "createdAt": "2026-06-01T03:00:00Z"},
        {"name": "X", "conclusion": "success", "createdAt": "2026-06-01T02:00:00Z"},
    ]
    assert hc.check_ci_runs(runs)["severity"] == "ok"


def test_assemble_report_overall_is_worst_severity():
    findings = [{"severity": "ok"}, {"severity": "warn"}, {"severity": "critical"}]
    assert hc.assemble_report(findings)["overall"] == "critical"


class _Resp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text


def test_fetch_endpoint_retries_a_cold_start_then_succeeds(monkeypatch):
    import requests
    calls = {"n": 0}

    def fake_get(url, timeout=0):
        calls["n"] += 1
        if calls["n"] < 2:
            raise Exception("cold start / connection timed out")
        return _Resp(200, '{"ok": true}')

    monkeypatch.setattr(requests, "get", fake_get)
    status, _ = hc.fetch_endpoint("http://x", "spy", attempts=3, sleeper=lambda s: None)
    assert status == 200          # a slow first load is NOT flagged as down
    assert calls["n"] == 2        # it retried once, then succeeded


def test_fetch_endpoint_recovers_from_a_cold_5xx(monkeypatch):
    import requests
    seq = [_Resp(503, "cold"), _Resp(200, '{"ok": true}')]

    monkeypatch.setattr(requests, "get", lambda url, timeout=0: seq.pop(0))
    status, _ = hc.fetch_endpoint("http://x", "spy", attempts=3, sleeper=lambda s: None)
    assert status == 200


def test_fetch_endpoint_returns_failure_after_exhausting_attempts(monkeypatch):
    import requests

    def always_down(url, timeout=0):
        raise Exception("genuinely down")

    monkeypatch.setattr(requests, "get", always_down)
    status, _ = hc.fetch_endpoint("http://x", "spy", attempts=2, sleeper=lambda s: None)
    assert status == 0            # a truly dead endpoint is still flagged


def test_fetch_endpoint_retries_transient_degraded_then_healthy(monkeypatch):
    import requests
    seq = [_Resp(200, '{"_meta":{"hasErrors":true}}'),
           _Resp(200, '{"_meta":{"hasErrors":false},"x":1}')]
    monkeypatch.setattr(requests, "get", lambda url, timeout=0: seq.pop(0))
    status, body = hc.fetch_endpoint("http://x", "market-extra", attempts=3, sleeper=lambda s: None)
    assert status == 200
    # A momentary degraded blip is NOT alarmed — the healthy retry wins.
    assert hc.check_endpoint("market-extra", status, body)["severity"] == "ok"


def test_fetch_endpoint_still_flags_persistent_degradation(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda url, timeout=0: _Resp(200, '{"_meta":{"hasErrors":true}}'))
    status, body = hc.fetch_endpoint("http://x", "market-extra", attempts=2, sleeper=lambda s: None)
    assert status == 200
    # Degraded on every attempt → genuinely degraded → still a warn.
    assert hc.check_endpoint("market-extra", status, body)["severity"] == "warn"


def test_format_summary_green_is_a_clean_check():
    report = {"overall": "ok", "generated_at": "2026-06-01T03:00:00Z",
              "findings": [{"id": "a", "severity": "ok"}, {"id": "b", "severity": "ok"}]}
    msg = hc.format_summary(report)
    assert "ALL GREEN" in msg
    assert "2/2" in msg


def test_format_summary_falls_back_to_alert_on_problems():
    report = {"overall": "warn", "generated_at": "2026-06-01T03:00:00Z",
              "findings": [{"id": "x", "severity": "warn", "title": "t", "detail": "d",
                            "remediation": "manual", "evidence": {}}]}
    msg = hc.format_summary(report)
    assert "ALL GREEN" not in msg
    assert "id=x" in msg


def test_format_alert_is_claude_actionable_and_omits_ok():
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
    assert "endpoint_spy" not in msg                 # ok findings are omitted
    assert "Paste" in msg or "paste" in msg or "Claude" in msg  # actionable framing


def test_format_alert_eli10_reassures_when_auto_remediated():
    report = {"overall": "critical", "generated_at": "2026-06-02T18:05:00Z", "findings": [
        {"id": "report_delivered_today", "severity": "critical", "title": "No report delivered today",
         "detail": "no marker", "remediation": "auto:redispatch_daily_report", "evidence": {}}]}
    msg = hc.format_alert(report)
    assert "💬" in msg                          # per-finding plain-English line
    assert "auto-re-sending" in msg             # plain explanation for this finding
    assert "BOTTOM LINE" in msg
    assert "don't need to do anything" in msg   # all-auto → reassuring, not alarming


def test_format_alert_eli10_flags_a_real_secret_leak():
    report = {"overall": "critical", "generated_at": "x", "findings": [
        {"id": "secret_leak", "severity": "critical", "title": "leak", "detail": "",
         "remediation": "manual", "evidence": {}}]}
    msg = hc.format_alert(report)
    assert "needs your attention" in msg        # a genuine issue is NOT reassured away


def test_fred_metrics_for_na_check_maps_horsemen_current_to_value():
    fred = {
        "indicators": {"sahmRule": {"value": 0.13, "unavailable": False}},
        "checklist": {"m2": {"value": 4.1, "unavailable": False}},
        "horsemen": {
            "claims": {"current": 221000, "unavailable": False, "staleDays": 0, "history": [1, 2]},
            "bankruptcies": {"current": None, "unavailable": True, "staleDays": 0, "history": []},
            "not_a_dict": "ignored",
        },
    }
    metrics = hc.fred_metrics_for_na_check(fred)
    assert metrics["sahmRule"]["value"] == 0.13
    assert metrics["m2"]["value"] == 4.1
    assert metrics["horsemen_claims"]["value"] == 221000
    assert metrics["horsemen_bankruptcies"]["value"] is None
    assert "horsemen_not_a_dict" not in metrics
    # An unavailable horseman then warns through the normal N/A sweep.
    finding = hc.check_indicators_na(metrics)
    assert finding["severity"] == "warn"
    assert "horsemen_bankruptcies" in finding["detail"]


def test_fred_metrics_for_na_check_flags_overdue_bankruptcies():
    fred = {"horsemen": {"bankruptcies": {"current": 25960, "unavailable": False, "staleDays": 40}}}
    finding = hc.check_indicators_na(hc.fred_metrics_for_na_check(fred))
    assert finding["severity"] == "warn"
    assert "overdue" in finding["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# check_lambda_path — the HTTP data path's liveness signal (added 2026-08-06).
#
# /api/spy, /api/market-extra, /api/polymarket and /api/spy-daily-move call the
# Lambda FIRST and fall back to direct sources. The fallback build deliberately
# returns hasErrors:false (a full direct build is healthy data), so if the Lambda
# lost its apigateway invoke grant — the EXACT failure that hid for two months on
# the EventBridge grant (AGENTS §2 gotcha #0) — all four routes would silently
# switch to fallbacks and every health signal would stay green.
#
# The source strings below are the REAL ones, captured live on 2026-08-06.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_LAMBDA_PAYLOADS = {
    # Lambda-built: bot/fetchers.py composes these (`<tier> + Finnhub Spot`).
    "spy": {"current": 631.0, "_meta": {"source": "Polygon + Finnhub Spot", "hasErrors": False}},
    "market-extra": {"_meta": {"source": "yfinance/Polygon/Finnhub/FRED/ER-API", "hasErrors": False}},
    # These two carry a TOP-LEVEL `source` and no _meta.source at all.
    "spy-daily-move": {"value": "+0.29%", "source": "Google Sheets"},
    "polymarket": {"bets": [{"name": "x"}], "timestamp": "2026-08-06T00:00:00Z"},
}

LIVE_FALLBACK_PAYLOADS = {
    "spy": {"current": 631.0, "_meta": {"source": "Polygon + Finnhub (fallback)", "hasErrors": False}},
    "market-extra": {"_meta": {"source": "Direct sources (fallback)", "hasErrors": False}},
    "spy-daily-move": {"value": "+0.29%", "source": "Finnhub (fallback)"},
    "polymarket": {"bets": [], "source": "Polymarket Gamma API (fallback)"},
}


def test_check_lambda_path_ok_on_the_real_live_healthy_strings():
    """The Lambda IS answering today; these exact strings must NOT alarm."""
    f = hc.check_lambda_path(LIVE_LAMBDA_PAYLOADS)
    assert f["severity"] == "ok", f
    assert f["id"] == "lambda_primary_path"


def test_check_lambda_path_warns_when_every_route_is_on_a_fallback():
    f = hc.check_lambda_path(LIVE_FALLBACK_PAYLOADS)
    assert f["severity"] == "warn"
    for name in ("spy", "market-extra", "spy-daily-move", "polymarket"):
        assert name in f["detail"], f["detail"]


def test_check_lambda_path_warns_when_only_one_route_fell_back():
    payloads = dict(LIVE_LAMBDA_PAYLOADS)
    payloads["spy"] = LIVE_FALLBACK_PAYLOADS["spy"]
    f = hc.check_lambda_path(payloads)
    assert f["severity"] == "warn"
    assert "spy" in f["detail"]
    assert f["evidence"]["fallback_routes"] == ["spy"]


def test_check_lambda_path_partial_fallback_blames_the_handler_not_the_grant():
    """2026-09-01: market-extra fell back (Lambda took 49-59 s > API Gateway's 30 s cap)
    while spy was Lambda-served. The invoke grant was fine; the alert must not send
    the reader there."""
    payloads = dict(LIVE_LAMBDA_PAYLOADS)
    payloads["market-extra"] = LIVE_FALLBACK_PAYLOADS["market-extra"]
    f = hc.check_lambda_path(payloads)
    assert f["severity"] == "warn"
    assert "30 s" in f["detail"] and "reachable" in f["detail"], f["detail"]
    assert "resource policy" not in f["detail"], f["detail"]
    assert "spy" in f["detail"]  # names what the Lambda DID serve


def test_check_lambda_path_total_fallback_points_at_the_invoke_grant():
    f = hc.check_lambda_path(LIVE_FALLBACK_PAYLOADS)
    assert "resource policy" in f["detail"], f["detail"]
    assert f["evidence"]["lambda_routes"] == []


def test_check_lambda_path_reads_a_top_level_source_when_there_is_no_meta():
    """spy-daily-move / polymarket have no _meta.source — the label is top-level."""
    payloads = dict(LIVE_LAMBDA_PAYLOADS)
    payloads["spy-daily-move"] = {"value": "+0.29%", "source": "Yahoo Finance (fallback)"}
    f = hc.check_lambda_path(payloads)
    assert f["severity"] == "warn"
    assert "spy-daily-move" in f["detail"]


def test_check_lambda_path_is_ok_when_nothing_is_readable():
    """Unknowable is not known-bad (the endpoint_* checks already cover an outage);
    a probe that could not be parsed must never manufacture a Lambda alarm."""
    f = hc.check_lambda_path({"spy": None, "market-extra": None,
                              "spy-daily-move": None, "polymarket": None})
    assert f["severity"] == "ok"


def test_check_lambda_path_ignores_a_cached_lambda_label():
    """serve()'s last-known-good re-labels the source but the payload still came
    from the Lambda originally — endpoint_* already flags it via hasErrors."""
    payloads = dict(LIVE_LAMBDA_PAYLOADS)
    payloads["spy"] = {"_meta": {"source": "Polygon + Finnhub Spot (last-known-good 2026-08-05T09:00:00Z)",
                                 "hasErrors": True}}
    assert hc.check_lambda_path(payloads)["severity"] == "ok"


def test_check_lambda_path_catches_every_lambda_spy_tier_label():
    """bot/fetchers.py can label spy from any waterfall tier; none of them may
    look like a dashboard fallback."""
    for tier in ("yfinance + Finnhub Spot", "Polygon + Finnhub Spot", "Google Sheet + Finnhub Spot",
                 "FRED S&P 500 Index"):
        payloads = dict(LIVE_LAMBDA_PAYLOADS)
        payloads["spy"] = {"_meta": {"source": tier}}
        assert hc.check_lambda_path(payloads)["severity"] == "ok", tier


# ─────────────────────────────────────────────────────────────────────────────
# The N/A sweep must also cover the TOP-LEVEL cards (added 2026-08-06).
#
# yieldCurve / profitMargin / spEps live at the root of /api/fred, not under
# `indicators` or `checklist`, so they sat outside fred_metrics_for_na_check
# entirely. yieldCurve is the 10Y-2Y horseman (AGENTS: it reuses fred.yieldCurve
# rather than duplicating it), so a frozen T10Y2Y that the repair cascade cannot
# replace left the line frozen with failed.length==0, hasErrors:false and
# endpoint_fred green — FOREVER — while its two sibling horsemen would have
# warned through staleDays > 3.
# ─────────────────────────────────────────────────────────────────────────────
def test_fred_metrics_for_na_check_covers_the_top_level_cards():
    fred = {
        "yieldCurve": {"current": 0.44, "asOf": "2026-08-06", "stale": False,
                       "staleDays": 0, "unavailable": False},
        "profitMargin": {"current": 14.9, "asOf": "2026-01-01", "stale": False,
                         "staleDays": 0, "unavailable": False},
        "spEps": {"current": 264.69, "asOf": "2026-03-31", "stale": False,
                  "staleDays": 0, "unavailable": False},
    }
    metrics = hc.fred_metrics_for_na_check(fred)
    assert metrics["yieldCurve"]["value"] == 0.44
    assert metrics["profitMargin"]["value"] == 14.9
    assert metrics["spEps"]["value"] == 264.69
    assert hc.check_indicators_na(metrics)["severity"] == "ok"


def test_a_frozen_yield_curve_now_warns_like_its_sibling_horsemen():
    """The whole point: FRED freezes T10Y2Y, the repair cascade adopts nothing,
    and the 10Y-2Y line goes on showing a stale number. That must alarm."""
    fred = {"yieldCurve": {"current": 0.44, "asOf": "2026-05-01", "stale": True,
                           "staleDays": 90, "unavailable": False}}
    f = hc.check_indicators_na(hc.fred_metrics_for_na_check(fred))
    assert f["severity"] == "warn"
    assert "yieldCurve" in f["detail"]
    assert "overdue" in f["detail"]


def test_an_unavailable_top_level_card_warns():
    fred = {"spEps": {"current": None, "asOf": None, "unavailable": True, "staleDays": 0}}
    f = hc.check_indicators_na(hc.fred_metrics_for_na_check(fred))
    assert f["severity"] == "warn"
    assert "spEps" in f["detail"]


def test_top_level_cards_are_skipped_when_absent_or_malformed():
    """A payload predating these fields (deploy lag) must not manufacture N/A warnings."""
    f = hc.check_indicators_na(hc.fred_metrics_for_na_check({"yieldCurve": "nope"}))
    assert f["severity"] == "ok", f


# --- Rubber Band Radar ---------------------------------------------------------
def test_rubber_band_endpoint_is_swept():
    assert "rubber-band" in hc.ENDPOINTS


def test_check_rubber_band_fresh_is_ok():
    p = {"asOf": "2026-09-01", "dials": {"slow": {"colour": "green"}}, "verdict": {"colour": "green"},
         "_meta": {"stale": False, "ageDays": 1}}
    f = hc.check_rubber_band(p)
    assert f["severity"] == "ok"


def test_check_rubber_band_stale_is_warn_with_age():
    p = {"asOf": "2026-08-25", "dials": {"slow": {"colour": "green"}}, "verdict": {"colour": "green"},
         "_meta": {"stale": True, "ageDays": 8, "messages": ["snapshot is 8 days old"]}}
    f = hc.check_rubber_band(p)
    assert f["severity"] == "warn" and "8" in f["title"] + f["detail"]
    assert "launchctl" in f["detail"] or "Mac mini" in f["detail"]


def test_check_rubber_band_missing_dials_is_critical():
    f = hc.check_rubber_band({"asOf": None, "dials": None, "verdict": None, "_meta": {"source": "Unavailable"}})
    assert f["severity"] == "critical"
    assert hc.check_rubber_band(None)["severity"] == "critical"
