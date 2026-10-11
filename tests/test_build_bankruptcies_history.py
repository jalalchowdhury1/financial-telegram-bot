"""scripts/build_bankruptcies_history.py: what it writes, and when it asks for the owner.

The monthly rebake pings the owner whenever this script exits non-zero. Until 11 Oct 2026
a run one quarter short (a uscourts blip on any of ~100 page loads) refused to write and
failed, so a hiccup read as a broken scraper. Now a quarter the run did not get keeps its
baked value, and only a bake that stopped GROWING exits 1. A uscourts outage stops the run
within minutes (it used to run past the step's time cap), and `--strict`, the PR dry run,
still fails a scraper that reads nothing.
"""
import importlib.util
import json
import os
from datetime import date

import pytest
import requests

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "build_bankruptcies_history.py")
TODAY = date(2026, 10, 11)    # newest published quarter: 2026-06-30 (2026-09-30 is out ~14 Nov)


@pytest.fixture
def bk(monkeypatch):
    spec = importlib.util.spec_from_file_location("build_bankruptcies_history", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    return mod


def _bake(bk, end=(2026, 6, 30), skip=(), bump=()):
    """Every quarter through `end`, minus `skip`; a quarter in `bump` has a corrected total."""
    out = []
    for i, (y, m, d) in enumerate(bk.quarters(end=end)):
        day = f"{y}-{m:02d}-{d:02d}"
        if day not in skip:
            out.append({"date": day, "total": 600000 + i + (7 if day in bump else 0), "business": 25000 + i})
    return out


def _file(tmp_path, quarters):
    path = tmp_path / "bankruptciesBaked.json"
    path.write_text(json.dumps(quarters, indent=1))
    return path


def test_a_quarter_the_run_could_not_fetch_keeps_its_baked_value(bk, tmp_path, capsys):
    path = _file(tmp_path, _bake(bk))
    before = path.read_text()
    assert bk.write_bake(_bake(bk, skip={"2010-06-30"}), path=str(path), today=TODAY) == 0
    assert path.read_text() == before
    out = capsys.readouterr().out
    assert "no change (102 quarters)" in out and "kept 1 quarter(s)" in out


def test_uscourts_down_for_the_whole_run_changes_nothing_and_stays_silent(bk, tmp_path):
    path = _file(tmp_path, _bake(bk))
    before = path.read_text()
    assert bk.write_bake([], path=str(path), today=TODAY) == 0
    assert path.read_text() == before


def test_a_new_quarter_is_added(bk, tmp_path, capsys):
    path = _file(tmp_path, _bake(bk, end=(2026, 3, 31)))
    assert bk.write_bake(_bake(bk), path=str(path), today=TODAY) == 0
    assert json.loads(path.read_text()) == _bake(bk)
    assert "wrote 102 quarters" in capsys.readouterr().out


def test_a_corrected_quarter_wins(bk, tmp_path, capsys):
    path = _file(tmp_path, _bake(bk))
    assert bk.write_bake(_bake(bk, bump={"2020-03-31"}), path=str(path), today=TODAY) == 0
    assert json.loads(path.read_text()) == _bake(bk, bump={"2020-03-31"})
    assert "revised: ['2020-03-31']" in capsys.readouterr().out


def test_only_the_newest_quarter_missing_is_silent(bk, tmp_path, capsys):
    """Published late, or caught by a blip: next month's run retries it."""
    path = _file(tmp_path, _bake(bk, end=(2026, 3, 31)))
    assert bk.write_bake(_bake(bk, end=(2026, 3, 31)), path=str(path), today=TODAY) == 0
    assert "2026-06-30 is not in yet" in capsys.readouterr().out


def test_a_bake_that_stopped_growing_exits_1_and_writes_nothing(bk, tmp_path, capsys):
    """2026-03-31 came out in May: every monthly run since has missed it."""
    path = _file(tmp_path, _bake(bk, end=(2025, 12, 31)))
    before = path.read_text()
    assert bk.write_bake(_bake(bk, end=(2026, 6, 30), skip={"2026-03-31"}), path=str(path), today=TODAY) == 1
    assert path.read_text() == before
    assert "NOT GROWING: no 2026-03-31 quarter" in capsys.readouterr().err


def test_an_unreadable_bake_is_never_overwritten(bk, tmp_path, capsys):
    path = tmp_path / "bankruptciesBaked.json"
    path.write_text('[{"date": "2001-03-31", ')
    assert bk.write_bake(_bake(bk), path=str(path), today=TODAY) == 1
    assert path.read_text() == '[{"date": "2001-03-31", '
    assert "REFUSING TO WRITE" in capsys.readouterr().err


class _Session:
    """Answers with the next of `answers`: a status code, or an exception to raise."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def get(self, url, timeout):
        self.calls += 1
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, Exception):
            raise a
        return type("Response", (), {"status_code": a, "text": "", "content": b""})()


@pytest.mark.parametrize("answers, status, calls", [
    ((503, 200), 200, 2),                                   # a blip, then the page
    ((requests.ConnectionError("reset"), 200), 200, 2),
    ((429, 429, 200), 200, 3),
    ((404,), 404, 1),                                       # an answer: not published yet
])
def test_get_retries_a_blip_but_not_an_answer(bk, answers, status, calls):
    sess = _Session(*answers)
    assert bk.get(sess, "https://x", 30).status_code == status
    assert sess.calls == calls


@pytest.mark.parametrize("answer, why", [(503, "HTTP 503"), (requests.ConnectionError("reset"), "ConnectionError: reset")])
def test_get_says_no_answer_when_every_try_fails(bk, answer, why):
    sess = _Session(answer)
    with pytest.raises(bk.NoAnswer, match=why):
        bk.get(sess, "https://x", 30)
    assert sess.calls == 3


class _Clock:
    """Fake time for a whole run: each request takes `per_call` s, sleeps add their s."""

    def __init__(self, bk, monkeypatch, per_call=0.3):
        self.now, self.per_call = 0.0, per_call
        monkeypatch.setattr(bk, "clock", lambda: self.now)
        monkeypatch.setattr(bk.time, "sleep", self.sleep)

    def sleep(self, s):
        self.now += s


def _uscourts(bk, monkeypatch, answer, clock=None, urls=None):
    """Every request gets `answer`: a status code, an exception, or a callable(url) -> either."""
    def get(self, url, timeout):
        if urls is not None:
            urls.append(url)
        if clock:
            clock.now += timeout if isinstance(answer, requests.Timeout) else clock.per_call
        a = answer(url) if callable(answer) else answer
        if isinstance(a, Exception):
            raise a
        return type("Response", (), {"status_code": a, "text": "", "content": b""})()
    monkeypatch.setattr(bk.requests, "Session", lambda: type("S", (), {"headers": {}, "get": get})())


def test_uscourts_down_stops_the_run_in_minutes_and_keeps_the_file(bk, tmp_path, monkeypatch, capsys):
    """main() end to end on today's date. Every failed quarter costs ~27 s of retries, and
    ~100 of them used to run past the step's 45-min cap, which pinged the owner."""
    path = _file(tmp_path, _bake(bk, end=bk.latest_quarter_end()))
    before = path.read_text()
    monkeypatch.setattr(bk, "OUT", str(path))
    clock, urls = _Clock(bk, monkeypatch), []
    _uscourts(bk, monkeypatch, 503, clock, urls)
    assert bk.main([]) == 0
    assert path.read_text() == before
    assert len(urls) == bk.BREAKER * 3 and clock.now < 3 * 60
    assert "STOPPED EARLY: uscourts did not answer 5 quarters in a row (HTTP 503)" in capsys.readouterr().out


def test_uscourts_hanging_also_stops_well_inside_the_cap(bk, tmp_path, monkeypatch):
    """Every request times out after its full 30 s: ~2 min a quarter."""
    monkeypatch.setattr(bk, "OUT", str(_file(tmp_path, _bake(bk, end=bk.latest_quarter_end()))))
    clock = _Clock(bk, monkeypatch)
    _uscourts(bk, monkeypatch, requests.Timeout("read timed out"), clock)
    assert bk.main([]) == 0
    assert clock.now < 15 * 60


def test_an_answer_resets_the_breaker(bk, tmp_path, monkeypatch, capsys):
    """Down for 4 quarters, then a 404 (an answer), then down again: never 5 in a row."""
    monkeypatch.setattr(bk, "OUT", str(_file(tmp_path, _bake(bk, end=bk.latest_quarter_end()))))
    pages = []
    def answer(url):
        if url not in pages:
            pages.append(url)                       # one per quarter, retries included
        return 404 if len(pages) % 5 == 0 else 503
    _uscourts(bk, monkeypatch, answer)
    bk.main([])
    out = capsys.readouterr().out
    assert "STOPPED EARLY" not in out and f"fetched 0 of {len(list(bk.quarters()))}" in out


def test_a_slow_uscourts_stops_at_the_time_budget(bk, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bk, "OUT", str(_file(tmp_path, _bake(bk, end=bk.latest_quarter_end()))))
    clock = _Clock(bk, monkeypatch, per_call=25)       # every answer takes 25 s
    _uscourts(bk, monkeypatch, 404, clock)
    assert bk.main([]) == 0
    assert "STOPPED EARLY: the 30-min time budget ran out" in capsys.readouterr().out
    assert clock.now < bk.BUDGET_S + 6 * 60             # one quarter at most past the budget


def test_the_newest_quarter_is_fetched_first(bk, tmp_path, monkeypatch):
    """So a run cut short still gets the new quarter; the old ones are already baked."""
    monkeypatch.setattr(bk, "OUT", str(_file(tmp_path, _bake(bk, end=bk.latest_quarter_end()))))
    urls = []
    _uscourts(bk, monkeypatch, 404, urls=urls)
    bk.main([])
    y, m, d = bk.latest_quarter_end()
    assert urls[0].endswith(f"/{y}/{m:02d}/{d:02d}/bankruptcy-filings/f-2")
    assert urls[-1].endswith("/2001/03/31/bankruptcy-filings/f-2")


def test_strict_fails_a_scraper_that_reads_nothing(bk, tmp_path, monkeypatch, capsys):
    """The PR dry run's whole point. Every page 404s (a broken URL template): the merge keeps
    the bake, so a plain run passes, and --strict must not."""
    baked = _bake(bk, end=bk.latest_quarter_end())
    monkeypatch.setattr(bk, "OUT", str(_file(tmp_path, baked)))
    _uscourts(bk, monkeypatch, 404)
    assert bk.main([]) == 0
    assert bk.main(["--strict"]) == 1
    assert f"STRICT: {len(baked)} baked quarters not re-read (first: 2001-03-31" in capsys.readouterr().err


def test_strict_fails_a_run_that_stopped_early(bk, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bk, "OUT", str(_file(tmp_path, _bake(bk, end=bk.latest_quarter_end()))))
    _Clock(bk, monkeypatch)
    _uscourts(bk, monkeypatch, 503)
    assert bk.main(["--strict"]) == 1
    assert "STRICT: uscourts did not answer 5 quarters in a row" in capsys.readouterr().err


@pytest.mark.parametrize("missing, code", [((), 0), (("2004-12-31", "2005-03-31"), 0),
                                           (("2004-12-31", "2005-03-31", "2010-06-30"), 1)])
def test_strict_allows_two_quarters_not_re_read(bk, tmp_path, missing, code):
    path = _file(tmp_path, _bake(bk))
    assert bk.check_strict(_bake(bk, skip=set(missing)), None, path=str(path)) == code
