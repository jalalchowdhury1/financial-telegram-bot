"""The Polymarket "Market Sentiment" board (bot/fetchers.py fetch_polymarket_board).

Every request goes through one URL router (the board fetches in threads, so a side_effect
list would be order-dependent). Fixtures are trimmed copies of the real shapes:
Gamma events/keyset + events?tag_slug, polymarket.com/api/biggest-movers, Gamma markets,
CLOB prices-history.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

import bot.fetchers as f


def _resp(data):
    r = MagicMock()
    r.json.return_value = data
    r.raise_for_status.return_value = None
    return r


def _mkt(label, yes, change=None, change30=None, token=None, closed=False, outcomes='["Yes", "No"]'):
    return {
        "groupItemTitle": label, "outcomes": outcomes,
        "outcomePrices": json.dumps([str(yes), str(round(1 - yes, 4))]),
        "oneDayPriceChange": change, "oneMonthPriceChange": change30,
        "clobTokenIds": json.dumps([token or f"tok-{label}", "no-token"]), "closed": closed,
    }


def _event(title, slug, markets, tags=None, volume=1_000_000):
    return {"title": title, "slug": slug, "markets": markets, "tags": tags or [],
            "volume": volume, "volume24hr": volume / 10, "endDate": "2026-11-03T00:00:00Z"}


FEATURED = {"events": [
    _event("Balance of Power: 2026 Midterms", "balance-of-power-2026-midterms", [
        _mkt("Republicans Sweep", 0.08), _mkt("Democrats Sweep", 0.62, change=0.012),
        _mkt("R Senate, D House", 0.30, change=-0.02), _mkt("Old outcome", 0.5, closed=True),
    ], tags=[{"label": "Politics", "slug": "politics"}], volume=21_400_000),
    _event("Ballon d'Or Winner 2026", "ballon-dor-winner-2026", [_mkt("Harry Kane", 0.66)]),
    _event("Lakers at Celtics", "lakers-celtics", [_mkt("Lakers", 0.4)],
           tags=[{"label": "Sports", "slug": "sports"}]),
    _event("Bitcoin Up or Down - 5 min", "btc-updown-5m-1760000000", [_mkt("Up", 0.5)]),
    _event("US recession by end of 2026?", "us-recession-by-end-of-2026", [_mkt("", 0.065)]),
    _event("Brazil Presidential Election", "brazil-presidential-election", [
        _mkt("Flávio Bolsonaro", 0.88, change=-0.01), _mkt("Luiz Inácio Lula da Silva", 0.12)],
        tags=[{"label": "Macro Election 2", "slug": "macro-election-2"}]),
    _event("Modi out by December 31, 2026?", "modi-out-2026", [_mkt("", 0.07, change=0.02)]),
    _event("Everything closed", "all-closed", [_mkt("A", 0.5, closed=True)]),
]}


def _mover(mid, question, slug, price, live, volume, history=None, closed=False):
    return {"id": mid, "question": question, "slug": f"m-{mid}", "currentPrice": price,
            "livePriceChange": live, "closed": closed, "events": [{"slug": slug, "volume": volume}],
            "history": history if history is not None else
            [{"t": i, "p": round(price - live / 100 * (1 - i / 99), 4)} for i in range(100)]}


MOVERS = {"markets": [
    _mover("1", "Will Putin hold a bilateral meeting with Alexander Lukashenko?", "putin-turkmenistan",
           0.015, -54, 79_000),
    _mover("2", "Will MrBeast Gaming's next video get 17.5-20M views?", "mrbeast-views", 0.665, 35, 66_000),
    _mover("s1", "Some sports market", "sports-thing", 0.5, 40, 500_000),          # in the sports bucket
    _mover("3", "Will Querétaro FC win on 2026-10-10?", "queretaro", 0.6, 30, 200_000),  # "FC" = sports
    _mover("4", "Will Tomáš Vohryzka be the next Mayor?", "usti-mayor", 0.6, 53, 21_000),  # < $50k
    _mover("5", "Small move market?", "small", 0.5, 3, 900_000),                    # < 5 points
    _mover("6", "Will MrBeast Gaming's video get 20-22.5M views?", "mrbeast-views", 0.2, -12, 66_000),  # same event
    _mover("7", "Closed already?", "closed-one", 0.99, 60, 900_000, closed=True),
]}
MOVERS_SPORTS = {"markets": [{"id": "s1"}]}


def _gmkt(question, slug, yes, change, volume=120_000, **extra):
    return {"question": question, "events": [{"slug": slug}], "outcomePrices": json.dumps([str(yes), "0"]),
            "oneDayPriceChange": change, "volumeNum": volume, "clobTokenIds": json.dumps([f"tok-{slug}", "x"]),
            "endDate": "2099-01-01T00:00:00Z", **extra}


GAMMA_UP = [
    _gmkt("Will SpaceXAI rename itself by October 15?", "spacexai-rename", 0.865, 0.52),
    _gmkt("CF América vs. CF Monterrey: O/U 1.5", "liga-mx", 0.5, 0.45, sportsMarketType="totals"),
    _gmkt("Will team X win on 2026-10-10?", "team-x", 0.9, 0.40, gameStartTime="2026-10-10 20:00:00+00"),
    _gmkt("Saudi action against Yemen on October 6?", "saudi-yemen-oct-6", 0.76, 0.28,
          endDate="2020-10-06T00:00:00Z"),                                          # ended: old news
]
GAMMA_DOWN = [_gmkt("Will KMT win the most local elections?", "kmt", 0.53, -0.345)]

MACRO = {
    "macro-graph": [_event("US recession by end of 2026?", "us-recession-by-end-of-2026",
                           [_mkt("", 0.065, change30=-0.03, token="tok-rec26")])],
    "macro-single": [
        _event("US recession by end of 2026?", "us-recession-by-end-of-2026", [_mkt("", 0.065, token="tok-rec26")]),
        _event("Another Fed rate hike in 2026?", "fed-hike-2026", [_mkt("", 0.775, change30=-0.02, token="tok-fed")]),
    ],
    "macro-fed": [_event("Fed decision in October?", "fed-october", [
        _mkt("No change", 0.84, change30=0.05, token="tok-hold"), _mkt("25 bps increase", 0.16)])],
    "macro-geopolitics": [_event("Grand Prix winner?", "grand-prix", [_mkt("Driver", 0.5)])],   # sports
}
CLOB = {  # token -> prices over time
    "tok-rec26": [0.085, 0.08, 0.07, 0.065],
    "tok-fed": [0.79, 0.78, 0.775],
    "tok-hold": [],                       # no history: keeps the 30-day change Gamma reported
}


def _router(overrides=None):
    """A fake requests.get. `overrides` maps a URL prefix to data, or to an Exception to raise."""
    overrides = overrides or {}

    def get(url, params=None, timeout=None):
        params = params or {}
        for prefix, val in overrides.items():
            if url.startswith(prefix) and (prefix != f"{f.PM_SITE}/api/biggest-movers" or "category" not in params):
                if isinstance(val, Exception):
                    raise val
                return _resp(val)
        if url == f"{f.PM_GAMMA}/events/keyset":
            assert params.get("featured_order") == "true"
            return _resp(FEATURED)
        if url == f"{f.PM_SITE}/api/biggest-movers":
            return _resp(MOVERS_SPORTS if params.get("category") == "sports" else MOVERS)
        if url == f"{f.PM_GAMMA}/markets":
            return _resp(GAMMA_UP if params.get("ascending") == "false" else GAMMA_DOWN)
        if url == f"{f.PM_GAMMA}/events":
            return _resp(MACRO.get(params.get("tag_slug"), []))
        if url == f"{f.PM_CLOB}/prices-history":
            prices = CLOB.get(params.get("market"))
            if prices is None:   # gamma backup tokens: a short 24h history
                prices = [0.3, 0.35, 0.4]
            return _resp({"history": [{"t": i, "p": p} for i, p in enumerate(prices)]})
        raise AssertionError(f"unexpected URL {url} {params}")
    return get


@pytest.fixture
def router():
    with patch("bot.fetchers.requests.get") as mock_get:
        mock_get.side_effect = _router()
        yield mock_get


def test_trending_is_the_featured_list_minus_sports_and_coin_flips(router):
    rows = f.fetch_polymarket_trending()
    assert [r["title"] for r in rows] == [
        "Balance of Power: 2026 Midterms", "US recession by end of 2026?",
        "Brazil Presidential Election", "Modi out by December 31, 2026?"]
    bop = rows[0]
    # open outcomes only, likeliest first, with the 24h change
    assert [(o["label"], o["odds"]) for o in bop["outcomes"]] == [
        ("Democrats Sweep", 0.62), ("R Senate, D House", 0.3), ("Republicans Sweep", 0.08)]
    assert bop["outcomes"][0]["change"] == 0.012
    assert bop["nOutcomes"] == 3 and bop["volume"] == 21_400_000
    assert (bop["topic"], bop["topicEmoji"]) == ("Politics", "🏛️")
    # Polymarket tags elections "Macro Election": still Politics, not Economy
    assert rows[2]["topic"] == "Politics"
    # a single-market event reads as a plain Yes chance
    assert rows[3]["outcomes"] == [{"label": "Yes", "odds": 0.07, "change": 0.02}]


def test_breaking_comes_from_the_sites_movers_feed(router):
    rows, source = f._pm_breaking_rows(10)
    assert source == "biggest-movers"
    # sports bucket, "FC", < $50k, < 5 points and closed are gone; one row per event
    assert [r["question"] for r in rows] == [
        "Will Putin hold a bilateral meeting with Alexander Lukashenko?",
        "Will MrBeast Gaming's next video get 17.5-20M views?"]
    putin = rows[0]
    assert putin["change"] == -0.54 and putin["odds"] == 0.015 and putin["volume"] == 79_000
    assert putin["slug"] == "putin-turkmenistan"          # the EVENT slug: /event/<slug> links work
    assert 2 <= len(putin["spark"]) <= 48 and putin["spark"][-1] == pytest.approx(0.015, abs=1e-3)
    assert putin["topic"] == "Geopolitics"


def test_breaking_falls_back_to_gamma_with_its_own_sparklines(router):
    router.side_effect = _router({f"{f.PM_SITE}/api/biggest-movers": requests.HTTPError("403")})
    rows, source = f._pm_breaking_rows(10)
    assert source == "gamma"
    # sportsMarketType / gameStartTime / an end date already passed are dropped
    assert [(r["question"], r["change"]) for r in rows] == [
        ("Will SpaceXAI rename itself by October 15?", 0.52), ("Will KMT win the most local elections?", -0.345)]
    assert rows[0]["spark"] == [0.3, 0.35, 0.4]
    assert rows[0]["slug"] == "spacexai-rename"


def test_breaking_falls_back_when_the_feed_changes_shape(router):
    # 1 market back, none usable: more likely a renamed field than a quiet day
    router.side_effect = _router({f"{f.PM_SITE}/api/biggest-movers": {"markets": [{"id": "9", "title": "?"}]}})
    rows, source = f._pm_breaking_rows(10)
    assert source == "gamma" and rows


def test_quiet_day_is_empty_but_still_has_a_source(router):
    router.side_effect = _router({f"{f.PM_SITE}/api/biggest-movers": requests.Timeout("slow"),
                                  f"{f.PM_GAMMA}/markets": []})
    assert f._pm_safe(f._pm_breaking_rows, 10) == ([], "gamma")


def test_macro_tiles_follow_the_tag_order_with_30_day_sparklines(router):
    tiles = f.fetch_polymarket_macro()
    assert [t["title"] for t in tiles] == [
        "US recession by end of 2026?", "Another Fed rate hike in 2026?", "Fed decision in October?"]
    rec = tiles[0]
    assert rec["spark"] == [0.085, 0.08, 0.07, 0.065]
    assert rec["change"] == pytest.approx(-0.02)           # what the sparkline draws, not Gamma's -0.03
    assert rec["label"] == "Yes" and rec["slug"] == "us-recession-by-end-of-2026"
    fed = tiles[2]
    assert fed["label"] == "No change" and fed["odds"] == 0.84
    assert fed["spark"] == [] and fed["change"] == 0.05    # no history: Gamma's 30-day change stands
    assert all("_token" not in t for t in tiles)


def test_board_drops_trending_rows_the_macro_strip_already_shows(router):
    board = f.fetch_polymarket_board()
    assert board["sources"] == {"trending": "featured", "breaking": "biggest-movers", "macro": "macro-tags"}
    assert "US recession by end of 2026?" not in [t["title"] for t in board["trending"]]
    assert "US recession by end of 2026?" in [m["title"] for m in board["macro"]]
    assert len(board["breaking"]) == 2


def test_everything_down_gives_empty_lists_and_no_sources_without_raising():
    with patch("bot.fetchers.requests.get", side_effect=requests.ConnectionError("down")):
        board = f.fetch_polymarket_board()
        assert board == {"trending": [], "breaking": [], "macro": [],
                         "sources": {"trending": None, "breaking": None, "macro": None}}
        assert f.fetch_polymarket_trending() == []
        assert f.fetch_polymarket_breaking() == []
        assert f.fetch_polymarket_macro() == []


@pytest.mark.parametrize("text,sports", [
    ("Fed interest rates in December?", False),                 # "inter" is not Inter Milan
    ("Will the U.S. invade a Latin American country?", False),  # "america" is not Club América
    ("Ballon d'Or Winner 2026", True),
    ("Lakers vs. Celtics", True),
    ("Will Querétaro FC win on 2026-10-10?", True),
    ("Spread: Chiefs (-3.5)", True),
])
def test_sports_filter_is_word_bounded(text, sports):
    assert f._pm_is_sports(text) is sports


def test_lambda_route_returns_the_board(monkeypatch):
    import lambda_handler
    board = {"trending": [{"title": "x", "slug": "x"}], "breaking": [], "macro": [],
             "sources": {"trending": "featured", "breaking": "gamma", "macro": None}}
    monkeypatch.setattr(lambda_handler, "fetch_polymarket_board", lambda: board)
    res = lambda_handler.handle_http_api({"rawPath": "/api/polymarket"}, {})
    body = json.loads(res["body"])
    assert res["statusCode"] == 200
    assert body["trending"] == board["trending"] and body["sources"] == board["sources"]
    assert body["source"] == "Polymarket API" and body["error"] is None   # no "(fallback)": Lambda-served
