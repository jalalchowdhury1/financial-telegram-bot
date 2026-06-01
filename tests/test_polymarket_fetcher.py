from unittest.mock import patch, MagicMock
from bot.fetchers import fetch_polymarket_trending


def _market(q, yes, vol, ticker, tags=None, outcomes='["Yes","No"]', change=None):
    m = {
        "question": q,
        "outcomes": outcomes,
        "outcomePrices": f'["{yes}", "{1 - yes:.4f}"]',
        "volume": str(vol),
        "events": [{"ticker": ticker, "slug": ticker, "title": ticker.replace("-", " ")}],
        "tags": tags or [],
    }
    if change is not None:
        m["oneMonthPriceChange"] = change
    return m


# A realistic mixed pool: meaningful binary markets across topics, plus a sports market,
# an extreme longshot, and a duplicate-event market that must be de-duped.
POOL = [
    _market("Will the US strike Iran by 2027?", 0.35, 2_500_000, "us-iran", change=-0.07),
    _market("Will Bitcoin hit $200k by 2026?", 0.62, 1_800_000, "btc-200k", change=0.05),
    _market("Will JD Vance win the 2028 nomination?", 0.31, 1_300_000, "vance-2028"),
    _market("Will a US recession be called in 2026?", 0.22, 900_000, "recession-2026"),
    _market("Will the Lakers win the 2026 Championship?", 0.45, 5_000_000, "nba",
            tags=[{"label": "Sports"}]),                                   # sports → filtered
    _market("Will fringe candidate win the 2028 race?", 0.01, 3_000_000, "fringe"),  # extreme → filtered
    _market("Will Bitcoin hit $200k by Dec 2026?", 0.60, 500_000, "btc-200k"),  # dup event → de-duped
    _market("Tiny obscure market?", 0.50, 5_000, "tiny"),                  # below volume floor → filtered
]


def _resp(data):
    r = MagicMock()
    r.json.return_value = data
    r.raise_for_status.return_value = None
    return r


def _paged(first):
    """First page returns the pool; subsequent pages empty (stops pagination)."""
    return [_resp(first)] + [_resp([]) for _ in range(6)]


@patch('bot.fetchers.requests.get')
def test_returns_curated_meaningful_markets(mock_get):
    mock_get.side_effect = _paged(POOL)
    bets = fetch_polymarket_trending()
    names = [b["name"] for b in bets]
    assert 0 < len(bets) <= 8
    # meaningful binary markets are included
    assert any("Iran" in n for n in names)
    assert any("Bitcoin" in n for n in names)
    # sports, extremes, and tiny-volume are excluded
    assert not any("Lakers" in n for n in names)
    assert not any("fringe" in n for n in names)
    assert not any("obscure" in n for n in names)


@patch('bot.fetchers.requests.get')
def test_each_bet_has_richer_fields(mock_get):
    mock_get.side_effect = _paged(POOL)
    bets = fetch_polymarket_trending()
    b = bets[0]
    for key in ("name", "odds", "volume", "change", "topic", "topicEmoji", "endDate", "eventSlug"):
        assert key in b, f"missing {key}"
    assert 0.08 <= b["odds"] <= 0.92


@patch('bot.fetchers.requests.get')
def test_dedupes_by_event(mock_get):
    mock_get.side_effect = _paged(POOL)
    bets = fetch_polymarket_trending()
    # Two "btc-200k" markets in the pool → only one survives.
    btc = [b for b in bets if "Bitcoin" in b["name"]]
    assert len(btc) == 1


@patch('bot.fetchers.requests.get')
def test_topic_tagging(mock_get):
    mock_get.side_effect = _paged(POOL)
    bets = fetch_polymarket_trending()
    by_name = {b["name"]: b for b in bets}
    iran = next(b for n, b in by_name.items() if "Iran" in n)
    assert iran["topic"] == "Geopolitics"
    btc = next(b for n, b in by_name.items() if "Bitcoin" in n)
    assert btc["topic"] == "Crypto"


@patch('bot.fetchers.requests.get')
def test_filters_extremes(mock_get):
    # A pool of only extreme markets → nothing meaningful → empty result (graceful).
    extremes = [_market("Near-certain thing?", 0.97, 2_000_000, "a"),
                _market("Near-impossible thing?", 0.02, 2_000_000, "b")]
    mock_get.side_effect = _paged(extremes)
    assert fetch_polymarket_trending() == []


@patch('bot.fetchers.requests.get')
def test_handles_api_failure_gracefully(mock_get):
    mock_get.side_effect = Exception("API timeout")
    assert fetch_polymarket_trending() == []


def _candidate(event_title, ticker, candidate, yes, vol):
    return {
        "question": f"Will {candidate} win {event_title}?",
        "outcomes": '["Yes","No"]',
        "outcomePrices": f'["{yes}", "{1 - yes:.4f}"]',
        "volume": str(vol),
        "groupItemTitle": candidate,
        "events": [{"ticker": ticker, "slug": ticker, "title": event_title}],
        "tags": [],
    }


@patch('bot.fetchers.requests.get')
def test_multi_candidate_event_collapses_to_favorite(mock_get):
    nominee = [
        _candidate("the 2028 Democratic Presidential Nominee", "dem-nom", "Gavin Newsom", 0.22, 2_000_000),
        _candidate("the 2028 Democratic Presidential Nominee", "dem-nom", "Kamala Harris", 0.15, 1_500_000),
        _candidate("the 2028 Democratic Presidential Nominee", "dem-nom", "AOC", 0.10, 1_000_000),
    ]
    mock_get.side_effect = _paged(nominee)
    bets = fetch_polymarket_trending()
    # Three candidate markets → ONE event row naming the favorite (highest odds), even
    # though 22% is below the 8% standalone floor (event favorites get their own band).
    assert len(bets) == 1
    b = bets[0]
    assert "Newsom" in b["name"]
    assert b["odds"] == 0.22
    assert b["volume"] == 4_500_000   # event total volume
    assert b["topic"] == "Politics"
