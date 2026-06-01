import importlib.util
import os
import json

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
