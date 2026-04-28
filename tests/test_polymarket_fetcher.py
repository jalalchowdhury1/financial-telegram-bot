import pytest
from unittest.mock import patch, MagicMock
from bot.fetchers import fetch_polymarket_trending

# Mock response from Polymarket API
MOCK_POLYMARKET_RESPONSE = [
    {
        "title": "Will Donald Trump be re-elected in 2024?",
        "category": "politics",
        "orderbook": {
            "bids": [{"price": 0.65, "size": 1000000}],
            "asks": [{"price": 0.66, "size": 1000000}]
        },
        "volume": 2500000,
        "active": True,
        "resolved": False,
        "resolved_price": None
    },
    {
        "title": "Will Bitcoin reach $100k by 2025?",
        "category": "crypto",
        "orderbook": {
            "bids": [{"price": 0.72, "size": 800000}],
            "asks": [{"price": 0.73, "size": 800000}]
        },
        "volume": 1800000,
        "active": True,
        "resolved": False,
        "resolved_price": None
    },
    {
        "title": "Will the Lakers win the 2024 Championship?",
        "category": "sports",
        "orderbook": {
            "bids": [{"price": 0.45, "size": 500000}],
            "asks": [{"price": 0.46, "size": 500000}]
        },
        "volume": 1000000,
        "active": True,
        "resolved": False,
        "resolved_price": None
    },
]

@patch('bot.fetchers.poly_client')
def test_fetch_polymarket_trending_returns_top_10_non_sports(mock_client):
    """Test that function returns top 10 non-sports, non-resolved markets."""
    mock_client.get_markets.return_value = MOCK_POLYMARKET_RESPONSE

    result = fetch_polymarket_trending()

    assert len(result) <= 10
    assert all(bet['name'] for bet in result)
    assert all('odds' in bet for bet in result)
    assert all('volume' in bet for bet in result)
    # Verify sports are filtered out
    assert not any('Lakers' in bet['name'] for bet in result)
    # Verify sorted by volume descending
    assert result[0]['volume'] >= result[1]['volume']

@patch('bot.fetchers.poly_client')
def test_fetch_polymarket_trending_handles_api_failure(mock_client):
    """Test graceful handling when API is unavailable."""
    mock_client.get_markets.side_effect = Exception("API timeout")

    result = fetch_polymarket_trending()

    assert result == []

@patch('bot.fetchers.poly_client')
def test_fetch_polymarket_trending_filters_resolved_markets(mock_client):
    """Test that resolved markets are excluded."""
    resolved_market = {
        "title": "Did the Fed raise rates?",
        "category": "economics",
        "resolved": True,
        "resolved_price": 1.0,
        "volume": 5000000
    }
    mock_client.get_markets.return_value = [resolved_market]

    result = fetch_polymarket_trending()

    assert len(result) == 0

@patch('bot.fetchers.poly_client')
def test_fetch_polymarket_trending_calculates_odds_from_orderbook(mock_client):
    """Test that odds are calculated from orderbook midpoint."""
    market = MOCK_POLYMARKET_RESPONSE[0]
    mock_client.get_markets.return_value = [market]

    result = fetch_polymarket_trending()

    assert len(result) == 1
    assert 0 <= result[0]['odds'] <= 1
