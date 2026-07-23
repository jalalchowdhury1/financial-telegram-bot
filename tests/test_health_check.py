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
