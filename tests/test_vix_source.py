"""
The VIX row of the daily brief must come from the DASHBOARD, not from the
Google Sheet cell that the (now retired) vix-fear-greed repo used to write.

Cell C2 is written by nothing once that repo is deleted, so reading it would
silently freeze the fear/greed tag at its last value forever — no error, no
alert. The dashboard computes the tag itself (CBOE -> FRED cascade), so the
bot consumes that instead and the formula lives in exactly one place.

The sheet stays wired up as a graceful fallback for a dashboard outage
(AGENTS.md §1: "prefer a graceful fallback over a hard change").
"""
from unittest.mock import patch, MagicMock

from bot.fetchers import fetch_vix_row


def _api(payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


def _sheet_csv(cur="11.11", m3="22.22", tag="SHEET99"):
    r = MagicMock()
    r.status_code = 200
    r.text = f"VIX,VIX 3M,Fear and Greed\n{cur},{m3},{tag}\n"
    return r


DASH_OK = _api({"VIX": {"current": "14.43", "threeMonth": "17.48", "fearGreed": "GREED13"}})


def test_uses_dashboard_when_available():
    with patch("bot.fetchers.requests.get", return_value=DASH_OK) as g:
        assert fetch_vix_row() == ("14.43", "17.48", "GREED13")
        # exactly one call, and it went to the dashboard — not the sheet
        assert g.call_count == 1
        assert "vercel.app" in g.call_args[0][0]


def test_falls_back_to_sheet_when_dashboard_raises():
    def side_effect(url, *a, **kw):
        if "vercel.app" in url:
            raise requests_exc()
        return _sheet_csv()
    with patch("bot.fetchers.requests.get", side_effect=side_effect):
        # levels from the sheet, but never its frozen C2 tag
        assert fetch_vix_row() == ("11.11", "22.22", "N/A")


def test_falls_back_to_sheet_when_dashboard_payload_is_unusable():
    bad = _api({"VIX": {"current": "N/A", "threeMonth": "N/A", "fearGreed": "N/A"}})

    def side_effect(url, *a, **kw):
        return bad if "vercel.app" in url else _sheet_csv()
    with patch("bot.fetchers.requests.get", side_effect=side_effect):
        assert fetch_vix_row() == ("11.11", "22.22", "N/A")


def test_dashboard_levels_without_a_tag_keep_the_levels_and_say_na():
    """Tag N/A on the dashboard must not send us to the sheet's frozen C2."""
    no_tag = _api({"VIX": {"current": "14.84", "threeMonth": "17.77", "fearGreed": "N/A"}})

    def side_effect(url, *a, **kw):
        return no_tag if "vercel.app" in url else _sheet_csv(tag="GREED13")
    with patch("bot.fetchers.requests.get", side_effect=side_effect) as g:
        assert fetch_vix_row() == ("14.84", "17.77", "N/A")
        assert g.call_count == 1


def test_cached_vix_fields_are_marked_cached():
    cached = _api({"VIX": {"current": "14.84", "threeMonth": "17.77", "fearGreed": "GREED04"},
                   "_meta": {"staleFields": ["vixCurrent"]}})
    with patch("bot.fetchers.requests.get", return_value=cached):
        assert fetch_vix_row() == ("14.84", "17.77", "GREED04 (cached)")


def test_returns_dashboard_tag_even_if_sheet_would_disagree():
    """The whole point: the dashboard is authoritative, the sheet is only a net."""
    def side_effect(url, *a, **kw):
        return DASH_OK if "vercel.app" in url else _sheet_csv(tag="STALE00")
    with patch("bot.fetchers.requests.get", side_effect=side_effect):
        assert fetch_vix_row()[2] == "GREED13"


def requests_exc():
    import requests
    return requests.RequestException("dashboard down")


# --- the fear/greed SCORE must survive into the brief ---------------------
# clean_val() strips trailing digits because the FrontRunner cell arrives as
# "BIL (T-Bill ETF)1". Applying it to the fear/greed tag also ate the score:
# "GREED13" rendered as bare "GREED". The score is real signal (13 = VIX sits
# ~13% under its own 50-day mean), not a spreadsheet artifact.
def _row(*cells):
    r = MagicMock()
    r.status_code = 200
    r.text = "h\n" + ",".join(cells) + "\n"
    return r


def test_brief_keeps_the_fear_greed_score():
    nsb = MagicMock(status_code=200, text="a,b\nc,d\ne,ON\n")
    fr = MagicMock(status_code=200, text="h\nBIL (T-Bill ETF)1\n")
    aaii = _api({"diff": "11.50%", "as_of": "2099-01-01", "source": "aaii.com", "stale": False})

    def side_effect(url, *a, **kw):
        if "/api/aaii" in url:
            return aaii
        if "vercel.app" in url:
            return DASH_OK
        if "10Y8Jus8" in url:
            return nsb
        return fr

    with patch("bot.fetchers.requests.get", side_effect=side_effect):
        from bot.fetchers import fetch_google_sheet_indicators
        out = fetch_google_sheet_indicators()

    assert "GREED13" in out, f"score stripped from the tag:\n{out}"
    # the FrontRunner artifact digit must STILL be stripped
    assert "BIL (T-Bill ETF)\n" in out
    assert "BIL (T-Bill ETF)1" not in out
