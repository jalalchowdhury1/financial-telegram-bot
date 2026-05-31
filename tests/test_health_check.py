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
    assert "NaN" in f["title"] or "NaN" in f["detail"]


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


def test_assemble_report_overall_is_worst_severity():
    findings = [{"severity": "ok"}, {"severity": "warn"}, {"severity": "critical"}]
    assert hc.assemble_report(findings)["overall"] == "critical"


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
