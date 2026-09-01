"""fetch_market_extra must answer inside API Gateway's 30 s integration cap.

WHY THIS EXISTS. On 2026-09-01 the health check flagged `lambda_primary_path`: the
dashboard's /api/market-extra had silently fallen back to direct sources because the
Lambda answered with HTTP 503. The Lambda was healthy -- /api/spy was served fine --
but fetch_market_extra ran ~25 upstream HTTP calls strictly one after another (9 FRED
+ ER-API + 5 yfinance + 4 Polygon + 5 Finnhub + 2 Stooq), each with a 10-20 s timeout
and no overall budget. Over 14 days: median 10.6 s, p90 21 s, 6% of calls > 30 s
(max 59 s). API Gateway HTTP APIs hard-cap integrations at 30 s and return 503, so
any slow upstream day flips the dashboard onto fallbacks with every check green.

Two properties guard against that here: the independent provider chains run
concurrently, and the whole fetch is bounded by a deadline that is honoured even
when one provider hangs.
"""
import time
from unittest.mock import patch

import pytest

from bot import fetchers
from bot.fetchers import fetch_market_extra

_real_sleep = time.sleep  # captured before the quiet_sleep fixture patches time.sleep

HELPERS = ['_fetch_fred_series', '_fetch_exchange_rates', '_fetch_yfinance',
           '_fetch_polygon_aggs', '_fetch_finnhub_quote', '_fetch_stooq']


def _slow(delay, value):
    def fn(*a, **k):
        _real_sleep(delay)
        return value
    return fn


def _metric(v):
    return {'current': v, 'dailyChange': {'value': 0, 'pct': 0}, 'history': [], 'lastDate': None}


@pytest.fixture
def quiet_sleep():
    """The function's own rate-limit pauses aren't what's under test."""
    with patch('time.sleep'):
        yield


def test_provider_chains_run_concurrently(quiet_sleep):
    # 25 calls x 0.2 s: sequential is ~5 s; concurrent is bounded by the longest
    # chain (FRED, 9 x 0.2 s = 1.8 s).
    with patch.object(fetchers, '_fetch_fred_series', _slow(0.2, [])), \
         patch.object(fetchers, '_fetch_exchange_rates', _slow(0.2, None)), \
         patch.object(fetchers, '_fetch_yfinance', _slow(0.2, None)), \
         patch.object(fetchers, '_fetch_polygon_aggs', _slow(0.2, None)), \
         patch.object(fetchers, '_fetch_finnhub_quote', _slow(0.2, None)), \
         patch.object(fetchers, '_fetch_stooq', _slow(0.2, None)):
        t0 = time.monotonic()
        fetch_market_extra('key', polygon_api_key='p', finnhub_api_key='f')
        elapsed = time.monotonic() - t0
    assert elapsed < 3.5, f'took {elapsed:.1f}s -- provider chains are still sequential'


def test_deadline_bounds_total_time_and_falls_through(quiet_sleep):
    # yfinance hangs far past the budget; the fetch must still return on time and
    # resolve USD/CAD from the next source in the waterfall (Polygon here).
    with patch.object(fetchers, '_fetch_fred_series', _slow(0, [])), \
         patch.object(fetchers, '_fetch_exchange_rates', _slow(0, {'CAD': 1.37, 'INR': 88.0, 'BDT': 122.0})), \
         patch.object(fetchers, '_fetch_yfinance', _slow(5, _metric(1.0))), \
         patch.object(fetchers, '_fetch_polygon_aggs', _slow(0, _metric(1.36))), \
         patch.object(fetchers, '_fetch_finnhub_quote', _slow(0, None)), \
         patch.object(fetchers, '_fetch_stooq', _slow(0, None)):
        t0 = time.monotonic()
        out = fetch_market_extra('key', polygon_api_key='p', finnhub_api_key='f',
                                 deadline_seconds=1.0)
        elapsed = time.monotonic() - t0
    assert elapsed < 2.5, f'took {elapsed:.1f}s -- deadline not honoured'
    assert out['fx']['usdcad']['current'] == 1.36
    assert out['_meta']['sourceLog']['usdcad'] == 'Polygon'
    assert any('yfinance' in m and 'deadline' in m for m in out['_meta']['messages']), out['_meta']


def test_default_deadline_fits_under_api_gateway_cap():
    assert fetchers.MARKET_EXTRA_DEADLINE_SECONDS < 30


def test_waterfall_priority_unchanged(quiet_sleep):
    # yfinance > Polygon > Finnhub for USD/CAD; the concurrency refactor must not
    # reorder the resolution.
    with patch.object(fetchers, '_fetch_fred_series', _slow(0, [])), \
         patch.object(fetchers, '_fetch_exchange_rates', _slow(0, None)), \
         patch.object(fetchers, '_fetch_yfinance', _slow(0, _metric(1.30))), \
         patch.object(fetchers, '_fetch_polygon_aggs', _slow(0, _metric(1.31))), \
         patch.object(fetchers, '_fetch_finnhub_quote', _slow(0, _metric(1.32))), \
         patch.object(fetchers, '_fetch_stooq', _slow(0, None)):
        out = fetch_market_extra('key', polygon_api_key='p', finnhub_api_key='f')
    assert out['fx']['usdcad']['current'] == 1.30
    assert out['_meta']['sourceLog']['usdcad'] == 'yfinance'
    assert out['commodities']['btc']['current'] == 1.30
