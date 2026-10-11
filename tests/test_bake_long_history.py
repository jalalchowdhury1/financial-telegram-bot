"""dashboard/scripts/bake_long_history.py — the quarterly long-history re-bake.

Three promises: a CNBC answer that CNBC itself cached as an error cannot fail a stat; one
bad stat cannot end the run; and the owner hears about a stat only when its chart is
really at risk. Before 2026-10-11 every one of PR #61's four dry runs died on the first:
CNBC replayed a stored "503 Backend fetch failed" for the exact URL the bake retried.
"""
import importlib.util
import io
import json
import os
from contextlib import redirect_stdout
from datetime import date, timedelta

import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dashboard", "scripts",
                      "bake_long_history.py")

BARS = {"barData": {"priceBars": [{"tradeTime": "20261009000000", "close": "98.7"}]}}
ERROR = {"status": "ERROR", "statusMessage":
         "Error in getting bars for symbol: .DXY and barType: ONE_WEEK 503 Backend fetch failed"}


@pytest.fixture
def bake(monkeypatch):
    spec = importlib.util.spec_from_file_location("bake_long_history", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    return mod


def cnbc_with_cache(failed_fetches):
    """CNBC's chart API as measured 2026-10-11: the first answer for an EXACT URL is cached
    and replayed, an error too (.DXY ALL kept replaying the 23:42 error of the night before
    until at least 09:42); a URL it has not cached reaches the backend, which here fails
    the first `failed_fetches` fetches and then recovers."""
    cache, fetches = {}, []

    def curl(url, ua=None, headers=(), **kw):
        if url not in cache:
            fetches.append(url)
            cache[url] = ERROR if len(fetches) <= failed_fetches else BARS
        return json.dumps(cache[url]).encode()

    curl.fetches = fetches
    return curl


def test_a_cached_cnbc_error_cannot_sink_the_stat(bake, monkeypatch):
    monkeypatch.setattr(bake, "curl", cnbc_with_cache(failed_fetches=2))
    assert bake.cnbc_bars.__wrapped__(".DXY", "10Y") == ((date(2026, 10, 9), 98.7),)


def test_every_cnbc_attempt_asks_a_url_the_cache_has_not_seen(bake, monkeypatch):
    fake = cnbc_with_cache(failed_fetches=3)
    monkeypatch.setattr(bake, "curl", fake)
    bake.cnbc_bars.__wrapped__("@GC.1", "ALL")
    assert len(fake.fetches) == 4                      # three errors, then the good answer
    for url in fake.fetches:                           # still the same symbol and range
        assert url.startswith("https://ts-api.cnbc.com/harmony/app/charts/ALL.json?symbol=%40GC.1&")


def test_cnbc_gives_up_naming_cnbcs_own_status(bake, monkeypatch):
    monkeypatch.setattr(bake, "curl", cnbc_with_cache(failed_fetches=99))
    with pytest.raises(RuntimeError, match="503 Backend fetch"):
        bake.cnbc_bars.__wrapped__(".DXY", "ALL")


def test_curl_tries_once_more_than_it_has_waits(bake, monkeypatch):
    """The sheet is fetched once, before any stat, so a Google blip used to crash the whole
    run; it now gets a patient schedule (SHEET_WAITS) instead of curl's ~7 s default."""
    calls = []

    class Done:
        def __init__(self, ok):
            self.returncode, self.stdout = (0, b"csv") if ok else (22, b"")

    def run(cmd, capture_output):
        calls.append(cmd)
        return Done(len(calls) == 6)

    monkeypatch.setattr(bake.subprocess, "run", run)
    assert bake.curl("https://example.test", waits=(1, 1, 1, 1, 1)) == b"csv"
    assert len(calls) == 6
    calls.clear()
    with pytest.raises(RuntimeError):
        bake.curl("https://example.test", waits=(1, 1))
    assert len(calls) == 3
    assert sum(bake.SHEET_WAITS) >= 120


def test_runway_names_only_charts_that_would_get_a_gap_within_a_year(bake):
    """/api/history serves 730 days of sheet, so a baked file leaves a gap in its chart from
    (its last day + 730). One missed quarter is nobody's problem; a gap inside a year is,
    and so is a stat with no file at all. `entries` is the index after the run."""
    today = date(2026, 10, 11)
    entries = {"fresh": {"to": "2026-10-09"},    # gap from 2028-10-08
               "edge": {"to": "2025-10-11"},     # gap from 2027-10-11: exactly a year away
               "stale": {"to": "2025-10-10"}}    # gap from 2027-10-10: 364 days away
    assert bake.runway_low(entries, ["fresh", "edge", "stale", "missing"], today) == [
        "stale (chart gap from 2027-10-10)", "missing (no baked file)"]


def _down():
    raise RuntimeError("CNBC .DXY 10Y: no bars after 4 tries (503 Backend fetch failed)")


def _daily(last, days=400):
    return [(last - timedelta(days=d), 1.0 + d % 7) for d in range(days, -1, -1)]


def _run_main(bake, monkeypatch, tmp_path, specs, index=None):
    """main() --dry against fake stats, a sheet with no rows (so no self-check runs) and
    an index whose entries say how far each stat's committed file reaches."""
    monkeypatch.setattr(bake, "OUT_DIR", tmp_path / "history")
    monkeypatch.setattr(bake, "INDEX", tmp_path / "index.json")
    monkeypatch.setattr(bake, "SPECS", specs)
    monkeypatch.setattr(bake, "curl", lambda url, **kw: b"Date,a,b\n")
    if index is not None:
        (tmp_path / "index.json").write_text(json.dumps(
            {"keys": {k: {"from": "1980-01-01", "to": to.isoformat(), "n": 1, "source": "test"}
                      for k, to in index.items()}}))
    out = io.StringIO()
    with redirect_stdout(out):
        code = bake.main(["--dry"])
    return code, out.getvalue()


def test_a_stat_that_kept_a_fresh_bake_is_listed_but_raises_no_alarm(bake, monkeypatch, tmp_path):
    today = date.today()
    code, out = _run_main(bake, monkeypatch, tmp_path,
                          {"ok": (1, 0, "test", lambda: _daily(today)), "down": (2, 0, "test", _down)},
                          index={"ok": today - timedelta(days=90), "down": today - timedelta(days=3)})
    assert code == 1                                   # a dry run still reports the miss
    assert "FAILED/REFUSED: ['down']" in out
    assert "RUNWAY LOW" not in out
    assert "NOTHING BAKED" not in out


def test_a_stat_whose_chart_is_running_out_of_runway_is_named(bake, monkeypatch, tmp_path):
    to = date.today() - timedelta(days=400)
    code, out = _run_main(bake, monkeypatch, tmp_path,
                          {"ok": (1, 0, "test", lambda: _daily(date.today())), "down": (2, 0, "test", _down)},
                          index={"down": to})
    assert code == 1
    assert f"RUNWAY LOW: down (chart gap from {to + timedelta(days=730)})" in out


def test_a_source_that_answers_but_stopped_updating_is_named(bake, monkeypatch, tmp_path):
    """It bakes without complaint, yet its file ends where the source froze: the chart gets
    a gap all the same, so the runway check covers every stat, not only the failed ones."""
    last = date.today() - timedelta(days=400)
    code, out = _run_main(bake, monkeypatch, tmp_path, {"stuck": (1, 0, "test", lambda: _daily(last))})
    assert code == 0
    assert f"RUNWAY LOW: stuck (chart gap from {last + timedelta(days=730)})" in out


def test_every_stat_failing_says_nothing_baked(bake, monkeypatch, tmp_path):
    """The workflow's one reason to ping when no file changed: nothing baked at all."""
    code, out = _run_main(bake, monkeypatch, tmp_path,
                          {"a": (1, 0, "test", _down), "b": (2, 0, "test", _down)},
                          index={"a": date.today(), "b": date.today()})
    assert code == 1
    assert "NOTHING BAKED" in out


def test_a_bug_in_one_stats_checks_does_not_sink_the_others(bake, monkeypatch, tmp_path):
    """A source that answers with no points used to raise in the self-check, outside the
    download's try, and end the whole run before the summary."""
    code, out = _run_main(bake, monkeypatch, tmp_path,
                          {"empty": (1, 0, "test", lambda: []), "ok": (2, 0, "test", lambda: _daily(date.today()))},
                          index={"empty": date.today()})
    assert code == 1
    assert "FAILED/REFUSED: ['empty']" in out
    assert "dry run" in out
