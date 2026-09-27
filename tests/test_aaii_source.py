"""
The brief's AAII line comes from the dashboard's /api/aaii, not the old Google
Sheet (which was fed by a leaked key that is being retired, 2026-09-27).

Rules under test:
  - fresh  -> the line looks exactly like the old sheet line
  - stale  -> the value is shown but clearly marked STALE with its date
  - 503 / timeout -> "AAII unavailable", after retries, and the rest of the
    brief still renders
"""
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import requests

from bot.fetchers import fetch_aaii, format_aaii_line, fetch_google_sheet_indicators

TODAY = date.today().isoformat()


def _resp(payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    if status >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    else:
        r.raise_for_status.return_value = None
    return r


FRESH = {"bull": 32.7, "neutral": 19.2, "bear": 48.0, "diff": "15.40%",
         "as_of": TODAY, "source": "aaii.com", "stale": False}


def test_fresh_line_is_identical_to_the_old_sheet_line():
    with patch("bot.fetchers.requests.get", return_value=_resp(FRESH)) as g:
        a = fetch_aaii()
    assert g.call_count == 1 and "/api/aaii" in g.call_args[0][0]
    assert a["stale"] is False
    assert format_aaii_line(a) == "🔸 AAII Diff : 15.40% (G | >20% | 6mths out)"


def test_stale_flag_marks_the_line():
    with patch("bot.fetchers.requests.get", return_value=_resp({**FRESH, "stale": True, "as_of": "2026-09-16"})):
        line = format_aaii_line(fetch_aaii())
    assert line == "🔸 AAII Diff : 15.40% ⚠️ STALE (as of 2026-09-16) (G | >20% | 6mths out)"


def test_old_as_of_is_stale_even_if_api_says_fresh():
    old = (date.today() - timedelta(days=30)).isoformat()
    with patch("bot.fetchers.requests.get", return_value=_resp({**FRESH, "as_of": old})):
        assert fetch_aaii()["stale"] is True


def test_missing_stale_flag_is_treated_as_stale():
    payload = {k: v for k, v in FRESH.items() if k != "stale"}
    with patch("bot.fetchers.requests.get", return_value=_resp(payload)):
        assert fetch_aaii()["stale"] is True


@patch("bot.fetchers.time.sleep")
def test_503_retries_twice_then_unavailable(_sleep):
    with patch("bot.fetchers.requests.get", return_value=_resp({"error": "down"}, 503)) as g:
        a = fetch_aaii()
    assert a is None and g.call_count == 3
    assert format_aaii_line(a) == "🔸 AAII Diff : ⚠️ AAII unavailable (G | >20% | 6mths out)"


@patch("bot.fetchers.time.sleep")
def test_timeout_retries_twice_then_unavailable(_sleep):
    with patch("bot.fetchers.requests.get", side_effect=requests.Timeout("slow")) as g:
        assert fetch_aaii() is None
    assert g.call_count == 3
    assert g.call_args.kwargs["timeout"] == 10


@patch("bot.fetchers.time.sleep")
def test_recovers_on_retry(_sleep):
    with patch("bot.fetchers.requests.get",
               side_effect=[requests.Timeout("slow"), _resp(FRESH)]) as g:
        assert fetch_aaii()["diff"] == "15.40%"
    assert g.call_count == 2


def test_malformed_200_is_unavailable_without_retry():
    with patch("bot.fetchers.requests.get", return_value=_resp({"bull": 1})) as g:
        assert fetch_aaii() is None
    assert g.call_count == 1


def _brief_get(aaii_behaviour):
    nsb = MagicMock(status_code=200, text="a,b\nc,d\ne,ON\n")
    fr = MagicMock(status_code=200, text="h\nBIL (T-Bill ETF)1\n")
    dash = _resp({"VIX": {"current": "14.43", "threeMonth": "17.48", "fearGreed": "GREED13"}})

    def side_effect(url, *a, **kw):
        if "/api/aaii" in url:
            return aaii_behaviour()
        if "/api/sheets" in url:
            return dash
        if "rubber-band" in url:
            raise requests.ConnectionError("n/a")
        if "10Y8Jus8" in url:
            return nsb
        if "1zQQ2am1" in url:
            raise AssertionError("the retired AAII sheet must never be read")
        return fr
    return side_effect


def test_brief_renders_fresh_aaii_line():
    with patch("bot.fetchers.requests.get", side_effect=_brief_get(lambda: _resp(FRESH))):
        out = fetch_google_sheet_indicators()
    assert "🔸 AAII Diff : 15.40% (G | >20% | 6mths out)\n\n" in out
    assert "🎢 VIX" in out


@patch("bot.fetchers.time.sleep")
def test_aaii_outage_never_breaks_the_rest_of_the_brief(_sleep):
    def boom():
        raise requests.Timeout("slow")
    with patch("bot.fetchers.requests.get", side_effect=_brief_get(boom)):
        out = fetch_google_sheet_indicators()
    assert "AAII unavailable" in out
    assert "🛡️ NotSoBoring : ON" in out
    assert "🎢 VIX: (Current | 3M) : 14.43 | 17.48 | GREED13" in out
    assert "[Financial Dashboard History]" in out


def test_config_no_longer_has_the_sheet_url():
    from bot.config import URLS
    assert "AAII" not in URLS
    assert not any("1zQQ2am1" in u for u in URLS.values())
