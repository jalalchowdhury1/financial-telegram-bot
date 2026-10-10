"""bot.fetchers.fetch_signal_pills: NotSoBoring + FrontRunner from the dashboard's
/api/sheets (computed from daily prices), sheet CSVs only as the fallback."""
from unittest.mock import MagicMock, patch

import requests

from bot.fetchers import fetch_signal_pills, FR_SHEET_NOTE


def _json(body):
    r = MagicMock(status_code=200)
    r.json.return_value = body
    r.raise_for_status.return_value = None
    return r


def _csv(text):
    return MagicMock(status_code=200, text=text)


NSB_SHEET = _csv("SPY,\nUltimate Frontrunner : ,BIL (T-Bill ETF)1\nNot so Boring : ,OFF\n")
FR_SHEET = _csv("h\nBIL (T-Bill ETF)1\n\n\nThis message was sent automatically with n8n\n")


def _get(dash):
    def side_effect(url, *a, **kw):
        if "/api/sheets" in url:
            if isinstance(dash, Exception):
                raise dash
            return dash
        if "10Y8Jus8" in url:
            return NSB_SHEET
        return FR_SHEET
    return side_effect


def test_dashboard_values_win_and_the_sheets_are_not_read():
    calls = []
    def side_effect(url, *a, **kw):
        calls.append(url)
        return _get(_json({"NotSoBoring": "ON", "FrontRunner": "UPRO", "_meta": {"staleFields": []}}))(url)
    with patch("bot.fetchers.requests.get", side_effect=side_effect):
        assert fetch_signal_pills() == ("ON", "UPRO")
    assert not any("docs.google.com" in u for u in calls)


def test_a_stale_dashboard_value_is_marked():
    dash = _json({"NotSoBoring": "ON", "FrontRunner": "UPRO", "_meta": {"staleFields": ["FrontRunner"]}})
    with patch("bot.fetchers.requests.get", side_effect=_get(dash)):
        assert fetch_signal_pills() == ("ON", "UPRO ⚠️ STALE")


def test_dashboard_down_falls_back_to_the_sheets_and_marks_frontrunner_frozen():
    with patch("bot.fetchers.requests.get", side_effect=_get(requests.ConnectionError("down"))):
        assert fetch_signal_pills() == ("OFF", f"BIL (T-Bill ETF) {FR_SHEET_NOTE}")


def test_dashboard_na_for_one_field_only_falls_back_for_that_field():
    dash = _json({"NotSoBoring": "N/A", "FrontRunner": "SOXL", "_meta": {}})
    with patch("bot.fetchers.requests.get", side_effect=_get(dash)):
        assert fetch_signal_pills() == ("OFF", "SOXL")


def test_everything_down_gives_na_and_never_raises():
    with patch("bot.fetchers.requests.get", side_effect=requests.ConnectionError("all down")):
        assert fetch_signal_pills() == ("N/A", "N/A")
