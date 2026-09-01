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
         patch.object(fetchers, '_fetch_gold_api', _slow(0.2, None)), \
         patch.object(fetchers, '_fetch_coinbase_spot', _slow(0.2, None)):
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
         patch.object(fetchers, '_fetch_gold_api', _slow(0, None)), \
         patch.object(fetchers, '_fetch_coinbase_spot', _slow(0, None)):
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
         patch.object(fetchers, '_fetch_gold_api', _slow(0, None)), \
         patch.object(fetchers, '_fetch_coinbase_spot', _slow(0, None)):
        out = fetch_market_extra('key', polygon_api_key='p', finnhub_api_key='f')
    assert out['fx']['usdcad']['current'] == 1.30
    assert out['_meta']['sourceLog']['usdcad'] == 'yfinance'
    assert out['commodities']['btc']['current'] == 1.30


def _fresh_fred_rows(v_today, v_prev):
    from datetime import date, timedelta
    today = date.today()
    return [{'date': today.isoformat(), 'value': v_today},
            {'date': (today - timedelta(days=1)).isoformat(), 'value': v_prev}]


def _all_dead():
    """Every tier dead except what a test re-patches. Returns the patch contexts."""
    return [patch.object(fetchers, '_fetch_fred_series', _slow(0, [])),
            patch.object(fetchers, '_fetch_exchange_rates', _slow(0, None)),
            patch.object(fetchers, '_fetch_yfinance', _slow(0, None)),
            patch.object(fetchers, '_fetch_polygon_aggs', _slow(0, None)),
            patch.object(fetchers, '_fetch_finnhub_quote', _slow(0, None)),
            patch.object(fetchers, '_fetch_gold_api', _slow(0, None)),
            patch.object(fetchers, '_fetch_coinbase_spot', _slow(0, None))]


def test_finnhub_is_only_asked_for_the_symbol_its_free_tier_serves(quiet_sleep):
    """OANDA:* (gold, oil, FX) returned 403 on every Lambda call for weeks -- forex and
    commodities need a paid Finnhub plan. Only BINANCE:BTCUSDT works; don't waste the
    other four calls (or the log noise)."""
    asked = []

    def fh(symbol, key):
        asked.append(symbol)
        return None

    with patch.object(fetchers, '_fetch_finnhub_quote', fh):
        ctxs = [c for c in _all_dead() if 'finnhub' not in str(c.attribute)]
        for c in ctxs:
            c.start()
        try:
            fetch_market_extra('key', polygon_api_key='p', finnhub_api_key='f')
        finally:
            for c in ctxs:
                c.stop()
    assert asked == ['BINANCE:BTCUSDT']


def test_gold_falls_to_fred_london_fix_then_gold_api(quiet_sleep):
    """Stooq (the old last resort) is behind a JS challenge now. Gold: yfinance ->
    Polygon -> FRED GOLDPMGBD228NLBM (history, stale-checked) -> gold-api.com spot."""
    def fred(series_id, key, limit):
        return _fresh_fred_rows(4370.0, 4350.0) if series_id == 'GOLDPMGBD228NLBM' else []

    ctxs = _all_dead()
    for c in ctxs:
        c.start()
    try:
        with patch.object(fetchers, '_fetch_fred_series', fred):
            out = fetch_market_extra('key')
        assert out['commodities']['gc']['current'] == 4370.0
        assert out['_meta']['sourceLog']['gold'] == 'FRED'
        assert len(out['commodities']['gc']['history']) == 2

        with patch.object(fetchers, '_fetch_gold_api', _slow(0, _metric(4374.0))):
            out = fetch_market_extra('key')
        assert out['commodities']['gc']['current'] == 4374.0
        assert out['_meta']['sourceLog']['gold'] == 'gold-api'
    finally:
        for c in ctxs:
            c.stop()


def test_btc_last_resort_is_coinbase(quiet_sleep):
    ctxs = _all_dead()
    for c in ctxs:
        c.start()
    try:
        with patch.object(fetchers, '_fetch_coinbase_spot', _slow(0, _metric(110000.0))):
            out = fetch_market_extra('key')
        assert out['commodities']['btc']['current'] == 110000.0
        assert out['_meta']['sourceLog']['btc'] == 'Coinbase'
    finally:
        for c in ctxs:
            c.stop()


def test_usd_rates_chain_falls_through_erapi_frankfurter_fawaz():
    """Finnhub's FX spot is gone; the keyless USD-rates chain now mirrors the
    dashboard's (ER-API -> Frankfurter -> Fawaz). Fawaz is the only one with BDT."""
    def boom(*a, **k):
        raise RuntimeError('down')

    with patch.object(fetchers, '_fetch_erapi_rates', boom), \
         patch.object(fetchers, '_fetch_frankfurter_rates', _slow(0, {'CAD': 1.39, 'INR': 95.0})):
        assert fetchers._fetch_exchange_rates() == {'CAD': 1.39, 'INR': 95.0}

    with patch.object(fetchers, '_fetch_erapi_rates', boom), \
         patch.object(fetchers, '_fetch_frankfurter_rates', boom), \
         patch.object(fetchers, '_fetch_fawaz_rates', _slow(0, {'CAD': 1.39, 'INR': 95.0, 'BDT': 123.7})):
        assert fetchers._fetch_exchange_rates()['BDT'] == 123.7

    with patch.object(fetchers, '_fetch_erapi_rates', boom), \
         patch.object(fetchers, '_fetch_frankfurter_rates', boom), \
         patch.object(fetchers, '_fetch_fawaz_rates', boom):
        assert fetchers._fetch_exchange_rates() is None


def test_rate_helpers_uppercase_and_shape():
    """Fawaz keys are lowercase and nested under 'usd'; Frankfurter nests under 'rates'."""
    with patch.object(fetchers.requests, 'get') as get:
        get.return_value.json.return_value = {'date': '2026-09-01', 'usd': {'cad': 1.389, 'bdt': 123.68}}
        get.return_value.raise_for_status.return_value = None
        assert fetchers._fetch_fawaz_rates() == {'CAD': 1.389, 'BDT': 123.68}
        get.return_value.json.return_value = {'base': 'USD', 'rates': {'CAD': 1.3888, 'INR': 94.95}}
        assert fetchers._fetch_frankfurter_rates() == {'CAD': 1.3888, 'INR': 94.95}
        get.return_value.json.return_value = {'symbol': 'XAU', 'price': 4374.0, 'updatedAt': '2026-09-01T16:02:20Z'}
        g = fetchers._fetch_gold_api()
        assert g['current'] == 4374.0 and g['history'] == []
        get.return_value.json.return_value = {'data': {'base': 'BTC', 'currency': 'USD', 'amount': '110000.12'}}
        assert fetchers._fetch_coinbase_spot()['current'] == 110000.12
