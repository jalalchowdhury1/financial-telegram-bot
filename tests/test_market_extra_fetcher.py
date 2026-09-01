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
from unittest.mock import MagicMock, patch

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
         patch.object(fetchers, '_fetch_coinbase_spot', _slow(0.2, None)), \
         patch.object(fetchers, '_fetch_cnbc_metrics', _slow(0.2, {})), \
         patch.object(fetchers, '_fetch_frankfurter_series', _slow(0.2, {})), \
         patch.object(fetchers, '_fetch_coinbase_candles', _slow(0.2, None)):
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
         patch.object(fetchers, '_fetch_coinbase_spot', _slow(0, None)), \
         patch.object(fetchers, '_fetch_cnbc_metrics', _slow(0, {})), \
         patch.object(fetchers, '_fetch_frankfurter_series', _slow(0, {})), \
         patch.object(fetchers, '_fetch_coinbase_candles', _slow(0, None)):
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
         patch.object(fetchers, '_fetch_coinbase_spot', _slow(0, None)), \
         patch.object(fetchers, '_fetch_cnbc_metrics', _slow(0, {})), \
         patch.object(fetchers, '_fetch_frankfurter_series', _slow(0, {})), \
         patch.object(fetchers, '_fetch_coinbase_candles', _slow(0, None)):
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
            patch.object(fetchers, '_fetch_coinbase_spot', _slow(0, None)),
            patch.object(fetchers, '_fetch_cnbc_metrics', _slow(0, {})),
            patch.object(fetchers, '_fetch_frankfurter_series', _slow(0, {})),
            patch.object(fetchers, '_fetch_coinbase_candles', _slow(0, None))]


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


def test_gold_last_resort_is_gold_api_spot(quiet_sleep):
    """Stooq (the old last resort) is behind a JS challenge, and FRED's LBMA gold fix
    GOLDPMGBD228NLBM was discontinued (404 on both FRED hosts, 2026-09-01 -- the tier
    added in #44 never worked). Gold: yfinance -> Polygon -> CNBC -> gold-api.com spot."""
    asked = []

    def fred(series_id, key, limit):
        asked.append(series_id)
        return []

    ctxs = _all_dead()
    for c in ctxs:
        c.start()
    try:
        with patch.object(fetchers, '_fetch_fred_series', fred), \
             patch.object(fetchers, '_fetch_gold_api', _slow(0, _metric(4374.0))):
            out = fetch_market_extra('key')
        assert out['commodities']['gc']['current'] == 4374.0
        assert out['_meta']['sourceLog']['gold'] == 'gold-api'
        assert 'GOLDPMGBD228NLBM' not in asked
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


# ─── history-bearing middle tiers (2026-09-01) ──────────────────────────────
# yfinance is Yahoo-rate-limited for hours at a time and Polygon 429s on the free
# tier, and every remaining fallback was spot-only (sparklines vanished). CNBC,
# Frankfurter's time-series and Coinbase candles all carry daily history and need
# no key; the dashboard already trusts CNBC for gold/copper/VIX.

def _hist_metric(v_now, v_prev, n=30):
    from datetime import date, timedelta
    today = date.today()
    rows = [{'date': (today - timedelta(days=n - i)).isoformat(), 'price': v_prev} for i in range(n)]
    rows.append({'date': today.isoformat(), 'price': v_now})
    return {'current': v_now, 'dailyChange': {'value': v_now - v_prev, 'pct': 0.0},
            'history': rows, 'lastDate': today.isoformat()}


def _run_with(overrides):
    ctxs = _all_dead()
    ctxs = [c for c in ctxs if c.attribute not in overrides]
    for c in ctxs:
        c.start()
    try:
        with patch.multiple(fetchers, **overrides):
            return fetch_market_extra('key', polygon_api_key='p', finnhub_api_key='f')
    finally:
        for c in ctxs:
            c.stop()


def test_cnbc_serves_gold_oil_fx_btc_with_history_when_yfinance_and_polygon_are_out(quiet_sleep):
    cnbc = {'@GC.1': _hist_metric(4404.3, 4481.5), '@CL.1': _hist_metric(89.59, 85.76),
            'CAD=': _hist_metric(1.39, 1.3854), 'INR=': _hist_metric(94.95, 95.16),
            'BTC.CB=': _hist_metric(77485.75, 78919.36)}
    asked = []

    def cnbc_fn(symbols):
        asked.extend(symbols)
        return cnbc

    out = _run_with({'_fetch_cnbc_metrics': cnbc_fn})
    log = out['_meta']['sourceLog']
    assert out['commodities']['gc']['current'] == 4404.3 and log['gold'] == 'CNBC'
    assert out['commodities']['cl']['current'] == 89.59 and log['cl'] == 'CNBC'
    assert out['fx']['usdcad']['current'] == 1.39 and log['usdcad'] == 'CNBC'
    assert out['fx']['usdinr']['current'] == 94.95 and log['usdinr'] == 'CNBC'
    assert out['commodities']['btc']['current'] == 77485.75 and log['btc'] == 'CNBC'
    assert len(out['commodities']['gc']['history']) == 31  # sparkline survives
    assert set(asked) == {'@GC.1', '@CL.1', 'CAD=', 'INR=', 'BTC.CB='}


def test_cnbc_sits_below_polygon_but_above_spot(quiet_sleep):
    out = _run_with({'_fetch_polygon_aggs': _slow(0, _metric(4400.0)),
                     '_fetch_cnbc_metrics': _slow(0, {'@GC.1': _hist_metric(4404.3, 4481.5)})})
    assert out['_meta']['sourceLog']['gold'] == 'Polygon'
    out = _run_with({'_fetch_cnbc_metrics': _slow(0, {'@GC.1': _hist_metric(4404.3, 4481.5)}),
                     '_fetch_gold_api': _slow(0, _metric(4374.0))})
    assert out['_meta']['sourceLog']['gold'] == 'CNBC'


def test_frankfurter_series_keeps_fx_history_when_fred_is_stale(quiet_sleep):
    out = _run_with({'_fetch_frankfurter_series': _slow(0, {'CAD': _hist_metric(1.3888, 1.3874),
                                                             'INR': _hist_metric(94.95, 87.5)}),
                     '_fetch_exchange_rates': _slow(0, {'CAD': 1.3863, 'INR': 95.2, 'BDT': 123.1})})
    assert out['fx']['usdcad']['current'] == 1.3888
    assert out['_meta']['sourceLog']['usdcad'] == 'Frankfurter'
    assert len(out['fx']['usdcad']['history']) == 31
    # BDT is only on the spot chain (Frankfurter has no BDT)
    assert out['fx']['usdbdt']['current'] == 123.1


def test_coinbase_candles_beat_spot_only_tiers_for_btc(quiet_sleep):
    out = _run_with({'_fetch_coinbase_candles': _slow(0, _hist_metric(77592.93, 78919.36)),
                     '_fetch_finnhub_quote': _slow(0, _metric(77800.0)),
                     '_fetch_coinbase_spot': _slow(0, _metric(77900.0))})
    assert out['commodities']['btc']['current'] == 77592.93
    assert out['_meta']['sourceLog']['btc'] == 'Coinbase'
    assert len(out['commodities']['btc']['history']) == 31


def test_cnbc_helper_merges_live_quote_onto_daily_bars():
    """Quote gives the live price + day change; bars give history. When the quote's
    date is newer than the last bar, the live print is appended as today's point."""
    quote_payload = {'QuickQuoteResult': {'QuickQuote': [
        {'symbol': '@GC.1', 'last': '4404.30', 'change': '-77.20', 'change_pct': '-1.72',
         'last_time': '2026-09-01T12:22:43.000-0400'},
        {'symbol': 'CAD=', 'last': '1.39', 'change': '0.0046', 'change_pct': '0.33', 'last_time': '2026-09-01'},
    ]}}
    bars = {'@GC.1': [{'tradeTime': '20260831000000', 'close': '4481.50'}, {'tradeTime': '20260828000000', 'close': '4470.00'}],
            'CAD=': [{'tradeTime': '20260901000000', 'close': '1.3854'}, {'tradeTime': '20260831000000', 'close': '1.3850'}]}

    def get(url, **kw):
        r = MagicMock(); r.raise_for_status.return_value = None
        if 'quote.cnbc.com' in url:
            r.json.return_value = quote_payload
        else:
            sym = url.split('symbol=')[1]
            from urllib.parse import unquote
            r.json.return_value = {'barData': {'priceBars': bars[unquote(sym)]}}
        return r

    with patch.object(fetchers.requests, 'get', get):
        out = fetchers._fetch_cnbc_metrics(['@GC.1', 'CAD='])
    g = out['@GC.1']
    assert g['current'] == 4404.3 and g['dailyChange'] == {'value': -77.2, 'pct': -1.72}
    assert [h['date'] for h in g['history']] == ['2026-08-28', '2026-08-31', '2026-09-01']  # sorted + live appended
    assert g['history'][-1]['price'] == 4404.3 and g['lastDate'] == '2026-09-01'
    c = out['CAD=']
    assert c['current'] == 1.39 and len(c['history']) == 2  # same-day bar: replaced, not duplicated
    assert c['history'][-1] == {'date': '2026-09-01', 'price': 1.39}


def test_cnbc_helper_survives_a_missing_history_or_quote():
    quote_payload = {'QuickQuoteResult': {'QuickQuote': {'symbol': '@CL.1', 'last': '89.59', 'change': '3.83',
                                                          'change_pct': '4.47', 'last_time': '2026-09-01'}}}

    def get(url, **kw):
        r = MagicMock(); r.raise_for_status.return_value = None
        if 'quote.cnbc.com' in url:
            r.json.return_value = quote_payload
        else:
            raise RuntimeError('history down')
        return r

    with patch.object(fetchers.requests, 'get', get):
        out = fetchers._fetch_cnbc_metrics(['@CL.1', 'INR='])
    assert out['@CL.1']['current'] == 89.59 and out['@CL.1']['history'] == []
    assert 'INR=' not in out


def test_frankfurter_series_and_coinbase_candles_parse_live_shapes():
    with patch.object(fetchers.requests, 'get') as get:
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {'base': 'USD', 'start_date': '2026-08-28', 'end_date': '2026-09-01',
                                              'rates': {'2026-09-01': {'CAD': 1.3888, 'INR': 94.95},
                                                        '2026-08-28': {'CAD': 1.3874, 'INR': 87.5}}}
        out = fetchers._fetch_frankfurter_series(['CAD', 'INR'])
        assert out['CAD']['current'] == 1.3888 and out['CAD']['history'][0]['date'] == '2026-08-28'
        assert out['INR']['dailyChange']['value'] == pytest.approx(7.45)
        # Coinbase: newest first, [time, low, high, open, close, volume]
        get.return_value.json.return_value = [[1788220800, 76000, 78000, 77000, 77592.93, 1.0],
                                              [1788134400, 77000, 79500, 79000, 78919.36, 1.0]]
        b = fetchers._fetch_coinbase_candles()
        assert b['current'] == 77592.93 and b['history'][0]['price'] == 78919.36
        assert b['history'][-1]['date'] == '2026-09-01' and b['lastDate'] == '2026-09-01'


# ─── Polygon: free tier is 5 requests/min and the key is shared ─────────────
# One dashboard load = spy (1) + market-extra (4) = 5 calls, so ANY second
# invocation inside the same minute (health-check retry, Vercel warm-up, a
# refresh) 429s. CloudWatch 2026-09-01: 429 count tracked invocation bursts 1:1.

def _poly_payload(closes):
    return {'results': [{'t': 1756684800000 + i * 86400000, 'c': c} for i, c in enumerate(closes)]}


@pytest.fixture
def poly_cache_clear():
    fetchers._POLYGON_CACHE.clear()
    yield
    fetchers._POLYGON_CACHE.clear()


def test_polygon_result_is_reused_within_the_warm_container(poly_cache_clear):
    with patch.object(fetchers.requests, 'get') as get:
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = _poly_payload([1.0, 1.1])
        a = fetchers._fetch_polygon_aggs('C:USDCAD', 'k', days=400)
        b = fetchers._fetch_polygon_aggs('C:USDCAD', 'k', days=400)
        fetchers._fetch_polygon_aggs('C:USDINR', 'k', days=400)  # different symbol -> its own call
    assert a == b and a['current'] == 1.1
    assert get.call_count == 2


def test_polygon_cache_expires(poly_cache_clear):
    with patch.object(fetchers.requests, 'get') as get, patch('time.monotonic') as mono:
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = _poly_payload([1.0, 1.1])
        mono.return_value = 1000.0
        fetchers._fetch_polygon_aggs('X:BTCUSD', 'k')
        mono.return_value = 1000.0 + fetchers.POLYGON_CACHE_TTL_SECONDS + 1
        fetchers._fetch_polygon_aggs('X:BTCUSD', 'k')
    assert get.call_count == 2


def test_polygon_429_is_a_distinct_error_and_is_not_cached(poly_cache_clear):
    import requests as rq
    with patch.object(fetchers.requests, 'get') as get:
        resp = MagicMock(); resp.status_code = 429
        get.return_value.raise_for_status.side_effect = rq.HTTPError('429 Too Many Requests', response=resp)
        get.return_value.status_code = 429
        with pytest.raises(fetchers.PolygonRateLimited):
            fetchers._fetch_polygon_aggs('C:XAUUSD', 'k')
    assert not fetchers._POLYGON_CACHE


def test_polygon_chain_staggers_calls_and_stops_at_the_first_429():
    asked, sleeps = [], []

    def poly(sym, key, days=400):
        asked.append(sym)
        if len(asked) == 2:
            raise fetchers.PolygonRateLimited('429')
        return _metric(1.0)

    with patch('time.sleep', lambda s: sleeps.append(s)):
        ctxs = [c for c in _all_dead() if c.attribute != '_fetch_polygon_aggs']
        for c in ctxs:
            c.start()
        try:
            with patch.object(fetchers, '_fetch_polygon_aggs', poly):
                out = fetch_market_extra('key', polygon_api_key='p', finnhub_api_key='f')
        finally:
            for c in ctxs:
                c.stop()
    assert asked == ['C:USDCAD', 'C:USDINR']  # bailed out; BTC + gold never requested
    assert fetchers.POLYGON_STAGGER_SECONDS in sleeps
    assert out['_meta']['sourceLog']['usdcad'] == 'Polygon'
    assert any('Polygon' in m and '429' in m for m in out['_meta']['messages']), out['_meta']['messages']


# ─── FRED: slow API hours must not take the rates tier down ──────────────────
# 2026-09-01 17:00 UTC: api.stlouisfed.org answered in 1-7 s and read-timed-out at
# 15 s while fred.stlouisfed.org/graph/fredgraph.csv (keyless) answered in 0.3 s.
# Nine serial calls x 15 s blew the 22 s deadline and 10Y/2Y/mortgage -- FRED-only
# metrics -- went unavailable.

def test_fred_series_falls_back_to_the_keyless_fredgraph_csv_host():
    import requests as rq
    calls = []

    def get(url, **kw):
        calls.append(url)
        if 'api.stlouisfed.org' in url:
            raise rq.exceptions.ReadTimeout('read timeout')
        r = MagicMock(); r.raise_for_status.return_value = None
        r.text = 'observation_date,DGS10\n2026-08-26,4.66\n2026-08-27,.\n2026-08-28,4.73\n'
        return r

    with patch.object(fetchers.requests, 'get', get):
        obs = fetchers._fetch_fred_series('DGS10', 'key', limit=2)
    assert [u for u in calls if 'fredgraph.csv?id=DGS10' in u]
    assert obs == [{'date': '2026-08-28', 'value': 4.73}, {'date': '2026-08-26', 'value': 4.66}]  # desc, '.' skipped, limited


def test_fred_series_uses_the_api_when_it_works():
    with patch.object(fetchers.requests, 'get') as get:
        get.return_value.raise_for_status.return_value = None
        get.return_value.json.return_value = {'observations': [{'date': '2026-08-28', 'value': '4.73'},
                                                               {'date': '2026-08-27', 'value': '.'}]}
        obs = fetchers._fetch_fred_series('DGS10', 'key')
    assert obs == [{'date': '2026-08-28', 'value': 4.73}]
    assert get.call_count == 1
    assert get.call_args.kwargs['timeout'] <= 8


def test_fred_chain_fetches_series_concurrently(quiet_sleep):
    # 9 series x 0.3 s: serial ~2.7 s; 3 workers ~0.9 s.
    ctxs = [c for c in _all_dead() if c.attribute != '_fetch_fred_series']
    for c in ctxs:
        c.start()
    try:
        with patch.object(fetchers, '_fetch_fred_series', _slow(0.3, [])):
            t0 = time.monotonic()
            fetch_market_extra('key')
            elapsed = time.monotonic() - t0
    finally:
        for c in ctxs:
            c.stop()
    assert elapsed < 2.0, f'took {elapsed:.1f}s -- FRED series are still fetched one at a time'
