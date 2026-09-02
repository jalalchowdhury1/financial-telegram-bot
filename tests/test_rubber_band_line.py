"""
The daily brief carries one Rubber Band Radar line (five coloured dots + the verdict),
read from the dashboard's /api/rubber-band. It must NEVER fail the brief: a dashboard
outage, a stale snapshot, or a malformed payload degrades to a short marker (or nothing),
and the rest of the report goes out unchanged.
"""
from unittest.mock import patch, MagicMock

from bot.fetchers import fetch_rubber_band_line, fetch_google_sheet_indicators


def _api(payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


SNAP = {
    "asOf": "2026-09-01",
    "verdict": {"colour": "green", "text": "The rubber band is working: the last 30 dips paid +0.63% more than an ordinary day. All legs inside their lines."},
    "dials": {"slow": {"colour": "green", "excess_pct": 0.634}, "fast": {"colour": "green"}, "age": {"colour": "green", "years": 2.8},
              "rip": {"colour": "green", "excess_pct": -0.068}, "machines": {"colour": "amber"}},
    "_meta": {"stale": False},
}


def test_line_has_five_dots_verdict_and_date():
    with patch("bot.fetchers.requests.get", return_value=_api(SNAP)):
        line = fetch_rubber_band_line()
    assert line.startswith("🪢 Rubber band 🟢🟢🟢🟢🟡")
    assert "+0.63%" in line and "2026-09-01" in line
    assert "All legs inside" in line


def test_stale_snapshot_is_marked_not_hidden():
    with patch("bot.fetchers.requests.get", return_value=_api({**SNAP, "_meta": {"stale": True, "ageDays": 6}})):
        line = fetch_rubber_band_line()
    assert "STALE" in line and "🟢🟢🟢🟢🟡" in line


def test_dashboard_down_or_garbage_returns_empty_never_raises():
    with patch("bot.fetchers.requests.get", side_effect=Exception("boom")):
        assert fetch_rubber_band_line() == ""
    with patch("bot.fetchers.requests.get", return_value=_api({"_meta": {"source": "Unavailable"}})):
        assert fetch_rubber_band_line() == ""
    with patch("bot.fetchers.requests.get", return_value=_api({"dials": "nonsense", "verdict": None})):
        assert fetch_rubber_band_line() == ""


def test_brief_includes_the_line_before_the_history_link_and_survives_without_it():
    def fake_get(url, timeout=10, **kw):
        if "rubber-band" in url:
            return _api(SNAP)
        if "/api/sheets" in url:
            return _api({"VIX": {"current": "14.4", "threeMonth": "17.5", "fearGreed": "GREED13"}})
        r = MagicMock(); r.status_code = 200
        r.text = "a,b,c,d,e\nx,y,z,w,v\nq,r,s,t,u\n"
        return r
    with patch("bot.fetchers.requests.get", side_effect=fake_get):
        out = fetch_google_sheet_indicators()
    assert "🪢 Rubber band" in out
    assert out.index("🪢 Rubber band") < out.index("[Financial Dashboard History]")
    assert out.index("🎢 VIX") < out.index("🪢 Rubber band")

    def fake_get_down(url, timeout=10, **kw):
        if "rubber-band" in url:
            raise Exception("dashboard down")
        return fake_get(url, timeout)
    with patch("bot.fetchers.requests.get", side_effect=fake_get_down):
        out = fetch_google_sheet_indicators()
    assert out and "🪢" not in out and "[Financial Dashboard History]" in out
