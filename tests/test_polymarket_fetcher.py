import pytest
from unittest.mock import patch, MagicMock
from bot.fetchers import fetch_polymarket_trending

# Mock response from Polymarket Gamma REST API
MOCK_POLYMARKET_RESPONSE = [
    {
        "question": "Will Donald Trump be re-elected in 2024?",
        "tags": [{"label": "Politics"}],
        "outcomePrices": "[\"0.65\", \"0.35\"]",
        "volume": "2500000",
        "active": True,
        "closed": False,
        "id": "123"
    },
    {
        "question": "Will Bitcoin reach $100k by 2025?",
        "tags": [{"label": "Crypto"}],
        "outcomePrices": "[\"0.72\", \"0.28\"]",
        "volume": "1800000",
        "active": True,
        "closed": False,
        "id": "124"
    },
    {
        "question": "Will the Lakers win the 2024 Championship?",
        "tags": [{"label": "Sports"}],
        "outcomePrices": "[\"0.45\", \"0.55\"]",
        "volume": "1000000",
        "active": True,
        "closed": False,
        "id": "125"
    },
]

@patch('bot.fetchers.requests.get')
def test_fetch_polymarket_trending_returns_top_10_non_sports(mock_get):
    """Test that function returns top 10 non-sports, non-resolved markets."""
    mock_get.return_value.json.return_value = MOCK_POLYMARKET_RESPONSE
    mock_get.return_value.raise_for_status.return_value = None

    result = fetch_polymarket_trending()

    assert len(result) <= 10
    assert all(bet['name'] for bet in result)
    assert all('odds' in bet for bet in result)
    assert all('volume' in bet for bet in result)
    # Verify sports are filtered out
    assert not any('Lakers' in bet['name'] for bet in result)
    # Verify sorted by volume descending
    assert result[0]['volume'] >= result[1]['volume']

@patch('bot.fetchers.requests.get')
def test_fetch_polymarket_trending_handles_api_failure(mock_get):
    """Test graceful handling when API is unavailable."""
    mock_get.side_effect = Exception("API timeout")

    result = fetch_polymarket_trending()

    assert result == []

@patch('bot.fetchers.requests.get')
def test_fetch_polymarket_trending_filters_sports_markets(mock_get):
    """Test that sports markets are excluded."""
    sports_only = [
        {
            "question": "Will the Lakers win the 2024 Championship?",
            "tags": [{"label": "Sports"}],
            "outcomePrices": "[\"0.45\", \"0.55\"]",
            "volume": "5000000",
            "active": True,
            "closed": False,
            "id": "126"
        }
    ]
    mock_get.return_value.json.return_value = sports_only
    mock_get.return_value.raise_for_status.return_value = None

    result = fetch_polymarket_trending()

    assert len(result) == 0

@patch('bot.fetchers.requests.get')
def test_fetch_polymarket_trending_parses_odds_from_outcome_prices(mock_get):
    """Test that odds are extracted from outcomePrices."""
    market = MOCK_POLYMARKET_RESPONSE[0]
    mock_get.return_value.json.return_value = [market]
    mock_get.return_value.raise_for_status.return_value = None

    result = fetch_polymarket_trending()

    assert len(result) == 1
    assert 0 <= result[0]['odds'] <= 1
