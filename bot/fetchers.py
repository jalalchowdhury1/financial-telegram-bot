"""
Data fetching logic for the financial-telegram-bot.
Handles FRED API, the SPY/market waterfalls, and Google Sheets integration.
"""

import csv
import logging
import requests
from io import StringIO
from typing import Dict, Any, List, Optional
from bot.config import URLS, RSI_PERIOD

# Optional heavy dependencies (for fetchers that need them)
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

def _vix_from_sheet() -> tuple:
    """The old path: read the VIX tab's A2/B2/C2 straight off the Google Sheet."""
    r = requests.get(URLS['VIX'], timeout=10)
    rows = list(csv.reader(StringIO(r.text)))
    return rows[1][0].strip(), rows[1][1].strip(), rows[1][2].strip()


def fetch_vix_row() -> tuple:
    """
    Return (current, 3-month, fear/greed tag) for the brief's VIX line.

    PRIMARY is the dashboard's /api/sheets. The dashboard computes the
    fear/greed tag itself (CBOE same-day -> FRED cascade; see
    dashboard/lib/vixFearGreed.js), so the formula lives in exactly ONE place
    and the site and this brief can never disagree.

    This deliberately does NOT read the sheet's cell C2 first. C2 was written
    once a day by the separate vix-fear-greed repo; the moment that repo is
    deleted nothing writes it, and the cell keeps serving its last value
    forever — a frozen tag with no error and no alert. A fallback that
    silently satisfies the caller is a false negative (AGENTS.md §7).

    FALLBACK is the sheet, exactly as before, for a dashboard outage. That is
    still C2 for the tag, so it can be stale post-deletion — but it only fires
    when the dashboard is down, which fleet-health already alarms on.
    """
    def _usable(v):
        return bool(v) and str(v).strip().upper() not in ('', 'N/A', 'NONE')

    try:
        r = requests.get(URLS['DASHBOARD_SHEETS'], timeout=15)
        r.raise_for_status()
        vix = (r.json() or {}).get('VIX') or {}
        current, three_m, tag = vix.get('current'), vix.get('threeMonth'), vix.get('fearGreed')
        if _usable(current) and _usable(three_m) and _usable(tag):
            return str(current).strip(), str(three_m).strip(), str(tag).strip()
        logging.warning("VIX: dashboard returned unusable values %r — falling back to the sheet", vix)
    except Exception as e:
        logging.warning("VIX: dashboard fetch failed (%s) — falling back to the sheet", e)

    return _vix_from_sheet()


def fetch_google_sheet_indicators() -> str:
    """
    Fetch custom indicator values from assigned Google Sheets via CSV export.
    Returns a rigidly formatted string to be prepended to the Telegram report.
    """
    print("Fetching Google Sheet custom indicators...")
    try:
        # 1. NotSoBoring
        r_nsb = requests.get(URLS['NOT_SO_BORING'], timeout=10)
        reader_nsb = list(csv.reader(StringIO(r_nsb.text)))
        not_so_boring_val = reader_nsb[2][1].strip()

        # 2. FrontRunner
        r_fr = requests.get(URLS['FRONT_RUNNER'], timeout=10)
        reader_fr = list(csv.reader(StringIO(r_fr.text)))
        front_runner_val = reader_fr[1][0].strip().split('\n')[0].strip()

        # 3. AAII Diff
        r_aaii = requests.get(URLS['AAII'], timeout=10)
        reader_aaii = list(csv.reader(StringIO(r_aaii.text)))
        aaii_val = reader_aaii[1][4].strip()

        # 4. VIX — from the dashboard (single source of truth for the
        #    fear/greed tag), sheet as graceful fallback. See fetch_vix_row.
        vix_current, vix_3m, fear_greed_status = fetch_vix_row()

        # Helper to strip trailing non-text/non-special characters (like the weird numbers in the screenshot)
        import re
        def clean_val(v):
            if not v: return v
            v_stripped = v.strip()
            # If the value is a pure number or decimal (e.g. "17.5", "20"), don't strip digits
            try:
                float(v_stripped)
                return v_stripped
            except ValueError:
                # If it's a string with trailing digits (e.g. "BIL (T-Bill ETF)1"), strip them
                return re.sub(r'\d+$', '', v_stripped)

        output = (
            f"🛡️ NotSoBoring : {clean_val(not_so_boring_val)}\n\n"
            f"🔑 FrontRunner : {clean_val(front_runner_val)}\n\n"
            f"🔸 AAII Diff : {clean_val(aaii_val)} (G | >20% | 6mths out)\n\n"
            # fear_greed_status is NOT passed through clean_val: the tag's
            # trailing digits are its score (GREED13 = VIX ~13% under its
            # 50-day mean), not the spreadsheet artifact clean_val exists to
            # strip off FrontRunner's "BIL (T-Bill ETF)1". It arrives already
            # clean from /api/sheets (and from C2 on the fallback path).
            f"🎢 VIX: (Current | 3M) : {clean_val(vix_current)} | {clean_val(vix_3m)} | {fear_greed_status}\n"
            f"\n[Financial Dashboard History](https://docs.google.com/spreadsheets/d/1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc/edit?gid=0#gid=0)"
        )
        print("✓ Successfully fetched and parsed Google Sheet indicators")
        return output

    except Exception as e:
        print(f"WARNING: Failed to fetch Google Sheet indicators: {e}")
        return ""

def calculate_rsi(prices: Any, period: int = RSI_PERIOD) -> float:
    """
    Calculate Relative Strength Index (RSI) using Wilder's Smoothing.

    Implements the traditional two-phase approach:
    1. Seed: SMA of first `period` bars
    2. Wilder's smoothing: exponential smoothing for subsequent bars

    Formula: RSI = 100 - (100 / (1 + RS)) where RS = avg_up / avg_down

    Must have at least `period + 1` prices to compute one diff and seed.
    """
    prices_array = prices.values.astype(float)

    # Step 1: Compute price differences
    diffs = np.diff(prices_array)
    ups = np.where(diffs > 0, diffs, 0.0)
    downs = np.where(diffs < 0, np.abs(diffs), 0.0)

    # Step 2: Seed with SMA of first `period` periods
    avg_up = np.mean(ups[:period])
    avg_down = np.mean(downs[:period])

    # Step 3: Apply Wilder's smoothing for all subsequent bars
    for i in range(period, len(diffs)):
        avg_up = (avg_up * (period - 1) + ups[i]) / period
        avg_down = (avg_down * (period - 1) + downs[i]) / period

    # Step 4: Calculate RS and RSI
    # Edge case: if avg_down == 0, RSI = 100
    if avg_down == 0:
        return 100.0

    rs = avg_up / avg_down
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Lambda fetchers
# These replicate the Next.js API route logic in Python so that AWS Lambda
# can serve the dashboard directly, replacing the Vercel/Next.js routes.
# ─────────────────────────────────────────────────────────────────────────────

_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; financial-bot/1.0)'}


def _calc_pct(current: float, base: float) -> float:
    if not base or base == 0:
        return 0.0
    return ((current - base) / base) * 100


def _fetch_fred_series(series_id: str, api_key: str, limit: int = 35) -> list:
    url = (f'https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={series_id}&api_key={api_key}'
           f'&file_type=json&sort_order=desc&limit={limit}')
    r = requests.get(url, timeout=15, headers=_HEADERS)
    r.raise_for_status()
    data = r.json()
    return [{'date': o['date'], 'value': float(o['value'])}
            for o in data.get('observations', []) if o['value'] != '.']


def _standardize_fred(data: list, multiplier: float = 1.0) -> Optional[Dict[str, Any]]:
    if not data or len(data) < 2:
        return None
    history = [{'date': d['date'], 'price': float(d['value']) * multiplier}
               for d in reversed(data)]
    current: float = history[-1]['price']
    prev: float = history[-2]['price']
    return {
        'current': current,
        'dailyChange': {'value': current - prev, 'pct': _calc_pct(current, prev)},
        'history': history,
        'lastDate': history[-1]['date'],
    }


def _is_stale(data: Optional[Dict[str, Any]]) -> bool:
    """True if the last data point is more than 1 calendar day old."""
    from datetime import datetime, timezone, timedelta
    if not data:
        return True
    last_date_str = data.get('lastDate') or (
        data['history'][-1]['date'] if data.get('history') else None)
    if not last_date_str:
        return True
    last_date = datetime.strptime(last_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last_date).days > 1


def _cross_rate_fred(usd_base: list, usd_target: list) -> Optional[Dict[str, Any]]:
    """Build a cross-rate series: target/base (e.g. INR per CAD = DEXINUS/DEXCAUS)."""
    if not usd_base or not usd_target:
        return None
    base_map = {d['date']: d['value'] for d in usd_base}
    history = []
    for t in usd_target:
        b = base_map.get(t['date'])
        if b and b != 0:
            history.append({'date': t['date'], 'price': round(t['value'] / b, 4)})
    if len(history) < 2:
        return None
    history.reverse()
    current = history[-1]['price']
    prev = history[-2]['price']
    return {
        'current': current,
        'dailyChange': {'value': round(current - prev, 4), 'pct': _calc_pct(current, prev)},
        'history': history,
        'lastDate': history[-1]['date'],
    }


def _spot_only(value: Optional[float]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return {'current': value, 'dailyChange': {'value': 0, 'pct': 0}, 'history': []}


def _fetch_erapi_rates() -> Optional[dict]:
    r = requests.get('https://open.er-api.com/v6/latest/USD', timeout=10, headers=_HEADERS)
    r.raise_for_status()
    data = r.json()
    return data.get('rates') if data.get('result') == 'success' else None


def _fetch_frankfurter_rates() -> Optional[dict]:
    """ECB reference rates (no BDT). Host moved from api.frankfurter.app (now a 301)."""
    r = requests.get('https://api.frankfurter.dev/v1/latest?base=USD', timeout=10, headers=_HEADERS)
    r.raise_for_status()
    return r.json().get('rates') or None


def _fetch_fawaz_rates() -> Optional[dict]:
    """fawazahmed0/currency-api on jsDelivr: keyless, ~200 currencies incl. BDT."""
    r = requests.get('https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.min.json',
                     timeout=10, headers=_HEADERS)
    r.raise_for_status()
    usd = r.json().get('usd') or {}
    return {k.upper(): float(v) for k, v in usd.items()} or None


def _fetch_exchange_rates() -> Optional[dict]:
    """USD-base spot rates: ER-API -> Frankfurter -> Fawaz (same chain the dashboard's
    /api/market-extra uses). Replaces Finnhub's OANDA:* FX quotes, which 403 on the
    free tier."""
    for name, fn in (('ER-API', _fetch_erapi_rates), ('Frankfurter', _fetch_frankfurter_rates),
                     ('Fawaz', _fetch_fawaz_rates)):
        try:
            rates = fn()
            if rates and (rates.get('CAD') or rates.get('INR')):
                return rates
        except Exception as e:
            print(f'[ExchangeRate] {name} failed: {e}')
    return None


def _compute_dxy(rates: dict) -> Optional[float]:
    try:
        dxy = (50.14348112
               * rates['EUR'] ** 0.576
               * rates['JPY'] ** 0.136
               * rates['GBP'] ** 0.119
               * rates['CAD'] ** 0.091
               * rates['SEK'] ** 0.042
               * rates['CHF'] ** 0.036)
        return round(dxy * 100) / 100
    except (KeyError, TypeError):
        return None


def _fetch_yfinance(symbol: str, invert: bool = False, days: int = 1500) -> Optional[Dict[str, Any]]:
    """Fetch daily history from Yahoo Finance via yfinance. Top of every waterfall."""
    try:
        import yfinance as yf
        import os
        
        # In AWS Lambda, only /tmp is writable. yfinance needs to store tz and cookie cache
        # or else Yahoo Finance triggers instant rate-limiting for repeated cookie-less scrapes.
        cache_dir = '/tmp/yfinance'
        os.makedirs(cache_dir, exist_ok=True)
        yf.set_tz_cache_location(cache_dir)
        
        from datetime import datetime, timedelta
        start = (datetime.now() - timedelta(days=days + 60)).strftime('%Y-%m-%d')
        # We omit 'end' to ensure we get up to the latest available live/closed price (including today)
        hist = yf.Ticker(symbol).history(start=start, auto_adjust=True)
        if hist.empty or len(hist) < 2:
            return None
        rows = []
        for dt_idx, row in hist.iterrows():
            price = float(row['Close'])
            if invert and price != 0:
                price = round(1.0 / price, 6)
            rows.append({'date': str(dt_idx.date()), 'price': price})
        if len(rows) < 2:
            return None
        current = rows[-1]['price']
        prev = rows[-2]['price']
        return {
            'current': current,
            'dailyChange': {'value': round(current - prev, 6), 'pct': _calc_pct(current, prev)},
            'history': rows,
            'lastDate': rows[-1]['date'],
        }
    except Exception as e:
        print(f'[yfinance] {symbol}: {e}')
        return None


# Polygon's free tier allows 5 requests/min and the key is shared with the dashboard.
# One dashboard load = /api/spy (1) + /api/market-extra (4) = 5, so any second
# invocation inside the same minute (health-check retry, Vercel warm-up, a refresh)
# 429s -- CloudWatch 2026-09-01 showed 429 counts tracking invocation bursts 1:1.
# Three mitigations: results are cached in the warm container so a burst re-uses
# data instead of re-fetching; calls are staggered; the chain stops at the first
# 429 rather than burning the remaining calls (the next tier fills the gaps).
POLYGON_CACHE_TTL_SECONDS = 600
POLYGON_STAGGER_SECONDS = 1.0
_POLYGON_CACHE: Dict[str, Any] = {}   # (symbol, days) -> (monotonic_stamp, metric)


class PolygonRateLimited(Exception):
    """Polygon answered 429 -- the per-minute quota is spent for the whole key."""


def _fetch_polygon_aggs(symbol: str, api_key: str, days: int = 1500) -> Optional[Dict[str, Any]]:
    """Fetch daily aggregates from Polygon.io — returns standard data shape with history.
    Cached per (symbol, days) for POLYGON_CACHE_TTL_SECONDS while the container is warm."""
    if not api_key:
        return None
    import time
    cache_key = f'{symbol}|{days}'
    hit = _POLYGON_CACHE.get(cache_key)
    if hit and time.monotonic() - hit[0] < POLYGON_CACHE_TTL_SECONDS:
        return hit[1]
    from datetime import datetime, timedelta
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    url = (f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}'
           f'?apiKey={api_key}&limit=1000&sort=asc&adjusted=true')
    r = requests.get(url, timeout=20, headers=_HEADERS)
    if r.status_code == 429:
        raise PolygonRateLimited(f'Polygon 429 for {symbol}')
    r.raise_for_status()
    data = r.json()
    results = data.get('results', [])
    if len(results) < 2:
        return None
    from datetime import datetime as _dt, timezone as _tz
    rows = [{'date': _dt.fromtimestamp(res['t'] / 1000, _tz.utc).strftime('%Y-%m-%d'), 'price': res['c']}
            for res in results]
    current = rows[-1]['price']
    prev = rows[-2]['price']
    metric = {
        'current': current,
        'dailyChange': {'value': round(current - prev, 4), 'pct': _calc_pct(current, prev)},
        'history': rows,
        'lastDate': rows[-1]['date'],
    }
    _POLYGON_CACHE[cache_key] = (time.monotonic(), metric)
    return metric


def _fetch_finnhub_quote(symbol: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch current quote from Finnhub — spot-only, no history."""
    if not api_key:
        return None
    url = f'https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}'
    r = requests.get(url, timeout=10, headers=_HEADERS)
    r.raise_for_status()
    data = r.json()
    current = data.get('c')
    prev = data.get('pc')
    if not current or current == 0:
        return None
    current = float(current)
    prev = float(prev) if prev else current
    return {
        'current': current,
        'dailyChange': {'value': round(current - prev, 4), 'pct': _calc_pct(current, prev)},
        'history': [],
        'lastDate': None,
    }


def _fetch_gold_api() -> Optional[Dict[str, Any]]:
    """Live XAU/USD spot from gold-api.com (keyless; the dashboard's /api/fred uses it
    too). Spot-only, no history."""
    r = requests.get('https://api.gold-api.com/price/XAU', timeout=10, headers=_HEADERS)
    r.raise_for_status()
    price = r.json().get('price')
    return _spot_only(float(price)) if price else None


def _fetch_coinbase_spot() -> Optional[Dict[str, Any]]:
    """BTC-USD spot from Coinbase (keyless). Spot-only, no history."""
    r = requests.get('https://api.coinbase.com/v2/prices/BTC-USD/spot', timeout=10, headers=_HEADERS)
    r.raise_for_status()
    amount = (r.json().get('data') or {}).get('amount')
    return _spot_only(float(amount)) if amount else None


def _rows_to_metric(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """[{date, price}] oldest->newest -> the standard metric shape (needs >= 2 rows)."""
    if len(rows) < 2:
        return None
    current, prev = rows[-1]['price'], rows[-2]['price']
    return {
        'current': current,
        'dailyChange': {'value': round(current - prev, 6), 'pct': _calc_pct(current, prev)},
        'history': rows,
        'lastDate': rows[-1]['date'],
    }


# ── CNBC (keyless; live quote + 1Y daily bars; datacenter-reachable) ─────────
# The dashboard already trusts these endpoints for gold/copper/VIX (lib/sources.js
# cnbcQuotes / cnbcHistory). Symbols: '@GC.1' gold (COMEX front month), '@CL.1' WTI,
# 'CAD=' / 'INR=' USD FX spot, 'BTC.CB=' Bitcoin (Coinbase).
HISTORY_ROWS = 400  # what yfinance/Polygon return; keeps the Lambda payload the same size
CNBC_QUOTE_URL = 'https://quote.cnbc.com/quote-html-webservice/quote.htm?symbols={}&output=json'
CNBC_HISTORY_URL = 'https://ts-api.cnbc.com/harmony/app/charts/1Y.json?symbol={}'


def _fetch_cnbc_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """One batched call -> {symbol: {price, change, change_pct, as_of}}."""
    from urllib.parse import quote
    url = CNBC_QUOTE_URL.format('%7C'.join(quote(sym, safe='') for sym in symbols))
    r = requests.get(url, timeout=10, headers=_HEADERS)
    r.raise_for_status()
    raw = (r.json().get('QuickQuoteResult') or {}).get('QuickQuote')
    items = raw if isinstance(raw, list) else ([raw] if raw else [])
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        try:
            price = float(it.get('last'))
        except (TypeError, ValueError):
            continue
        def _f(v: Any) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
        last_time = it.get('last_time') or ''
        out[it.get('symbol')] = {'price': price, 'change': _f(it.get('change')),
                                 'change_pct': _f(it.get('change_pct')),
                                 'as_of': last_time[:10] if len(last_time) >= 10 else None}
    return out


def _fetch_cnbc_history(symbol: str) -> List[Dict[str, Any]]:
    """1Y daily closes, oldest -> newest."""
    from urllib.parse import quote
    r = requests.get(CNBC_HISTORY_URL.format(quote(symbol, safe='')), timeout=10, headers=_HEADERS)
    r.raise_for_status()
    bars = (r.json().get('barData') or {}).get('priceBars') or []
    rows = []
    for b in bars:
        tt = str(b.get('tradeTime') or '')
        try:
            price = float(b.get('close'))
        except (TypeError, ValueError):
            continue
        if len(tt) < 8:
            continue
        rows.append({'date': f'{tt[:4]}-{tt[4:6]}-{tt[6:8]}', 'price': price})
    # CNBC's 1Y feed carries several bars per day; keep the last close per date.
    by_date = {r['date']: r for r in rows}
    return [by_date[d] for d in sorted(by_date)][-HISTORY_ROWS:]


def _fetch_cnbc_metrics(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """{symbol: metric} — live quote merged onto the daily bars. A symbol with a
    quote but no bars is served spot-only; one with neither is omitted."""
    try:
        quotes = _fetch_cnbc_quotes(symbols)
    except Exception as e:
        print(f'[CNBC] quotes failed: {e}')
        quotes = {}
    out: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        try:
            hist = _fetch_cnbc_history(sym)
        except Exception as e:
            print(f'[CNBC] history {sym} failed: {e}')
            hist = []
        q = quotes.get(sym)
        if q:
            if q['as_of'] and hist:  # merge the live print onto the bars (never a 1-point history)
                if hist[-1]['date'] == q['as_of']:
                    hist[-1] = {'date': q['as_of'], 'price': q['price']}
                elif hist[-1]['date'] < q['as_of']:
                    hist.append({'date': q['as_of'], 'price': q['price']})
            out[sym] = {'current': q['price'],
                        'dailyChange': {'value': q['change'], 'pct': q['change_pct']},
                        'history': hist, 'lastDate': hist[-1]['date'] if hist else q['as_of']}
        else:
            m = _rows_to_metric(hist)
            if m:
                out[sym] = m
    return out


def _fetch_frankfurter_series(symbols: List[str], days: int = 400) -> Dict[str, Dict[str, Any]]:
    """ECB daily USD-base rates with history (keyless; no BDT) -> {sym: metric}."""
    from datetime import datetime, timedelta
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    url = f'https://api.frankfurter.dev/v1/{start}..?base=USD&symbols={",".join(symbols)}'
    r = requests.get(url, timeout=15, headers=_HEADERS)
    r.raise_for_status()
    by_date = r.json().get('rates') or {}
    out: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        rows = [{'date': d, 'price': float(v[sym])} for d, v in sorted(by_date.items()) if v.get(sym) is not None]
        m = _rows_to_metric(rows[-HISTORY_ROWS:])
        if m:
            out[sym] = m
    return out


def _fetch_coinbase_candles() -> Optional[Dict[str, Any]]:
    """BTC-USD daily closes from Coinbase Exchange (keyless, ~300 days, newest first)."""
    from datetime import datetime, timezone
    r = requests.get('https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400',
                     timeout=10, headers=_HEADERS)
    r.raise_for_status()
    rows = []
    for c in r.json() or []:  # [time, low, high, open, close, volume]
        try:
            rows.append({'date': datetime.fromtimestamp(int(c[0]), timezone.utc).strftime('%Y-%m-%d'),
                         'price': float(c[4])})
        except (TypeError, ValueError, IndexError):
            continue
    rows.sort(key=lambda x: x['date'])
    return _rows_to_metric(rows[-HISTORY_ROWS:])


def fetch_spy_with_fallback(fred_api_key: Optional[str] = None,
                            polygon_api_key: Optional[str] = None,
                            finnhub_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch SPY stats with waterfall fallback.
    Returns JSON matching the Next.js /api/spy response shape.

    Layer 0: Polygon (full history → compute all indicators)
    Layer 1: Google Sheets pre-calculated indicators
    Layer 2: FRED SP500 series
    """
    if pd is None:
        logging.warning("pandas not available, returning placeholder SPY data")
        return {'current': 0, 'error': 'pandas dependency not available'}

    indicators = None
    rows: List[Dict] = []
    data_source = 'unknown'

    # Layer 0: yfinance — full history, highest priority
    try:
        yf_spy = _fetch_yfinance('SPY', days=1500)
        if yf_spy and len(yf_spy['history']) >= 200:
            rows = [{'date': h['date'], 'close': h['price']} for h in yf_spy['history']]
            data_source = 'yfinance'
            print(f'[SPY] Layer 0 (yfinance) loaded {len(rows)} rows')
    except Exception as e:
        print(f'[SPY] Layer 0 (yfinance) failed: {e}')

    # Layer 1: Polygon — full history
    if not rows and polygon_api_key:
        try:
            poly = _fetch_polygon_aggs('SPY', polygon_api_key, days=1500)
            if poly and len(poly['history']) >= 200:
                rows = [{'date': h['date'], 'close': h['price']} for h in poly['history']]
                data_source = 'Polygon'
                print(f'[SPY] Layer 1 (Polygon) loaded {len(rows)} rows')
        except Exception as e:
            print(f'[SPY] Layer 1 (Polygon) failed: {e}')

    # Layer 2: Google Sheets (skip if yfinance or Polygon already gave us rows)
    if not rows:
        try:
            r = requests.get(URLS['SPY_INDICATORS'], timeout=15, headers=_HEADERS)
            text = r.text
            if text and len(text.strip()) >= 50:
                parsed: Dict[str, float] = {}
                for line in text.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            parsed[parts[0].strip()] = float(parts[1].strip())
                        except ValueError:
                            pass
                required = ['200d MA SPY', '9d RSI SPY', 'SPY 52 week high', 'Current SPY']
                if all(k in parsed for k in required):
                    return3y_val = parsed.get('Three-Year Return')
                    if return3y_val is None:
                        try:
                            r2 = requests.get(URLS['SPY_DAILY_MOVE'], timeout=10, headers=_HEADERS)
                            daily_rows = list(csv.reader(StringIO(r2.text)))
                            raw = daily_rows[10][1].strip() if len(daily_rows) > 10 and len(daily_rows[10]) > 1 else None
                            if raw:
                                return3y_val = float(raw.replace('%', '').strip())
                        except Exception:
                            pass
                    indicators = {
                        'ma200': parsed['200d MA SPY'],
                        'rsi': parsed['9d RSI SPY'],
                        'week52High': parsed['SPY 52 week high'],
                        'current': parsed['Current SPY'],
                        'return3y': return3y_val,
                    }
                    data_source = 'Google Sheet'
                    print(f'[SPY] Layer 1 loaded: current={indicators["current"]}, MA200={indicators["ma200"]}, 3yr={return3y_val}')
        except Exception as e:
            print(f'[SPY] Layer 1 (Google Sheet) failed: {e}')

    # Layer 2: FRED SP500
    if not indicators and not rows and fred_api_key:
        try:
            url = (f'https://api.stlouisfed.org/fred/series/observations'
                   f'?series_id=SP500&api_key={fred_api_key}'
                   f'&file_type=json&observation_start=2010-01-01&limit=5000&sort_order=asc')
            r = requests.get(url, timeout=20, headers=_HEADERS)
            fred_rows = [{'date': o['date'], 'close': float(o['value'])}
                         for o in r.json().get('observations', []) if o['value'] != '.']
            if len(fred_rows) >= 10:
                rows = fred_rows
                data_source = 'FRED S&P 500 Index'
                print(f'[SPY] Layer 2 (FRED) loaded {len(rows)} rows')
        except Exception as e:
            print(f'[SPY] Layer 2 (FRED SP500) failed: {e}')

    if not indicators and len(rows) < 10:
        raise ValueError('Insufficient SPY data — all sources failed')

    chart_history = []

    if indicators:
        current: float = indicators['current'] or 0.0
        ma200: float = indicators['ma200'] or 0.0
        week52High: float = indicators['week52High'] or 0.0
        rsi: float = indicators['rsi'] or 0.0
        return3y = indicators.get('return3y')
        ma200_pct = _calc_pct(current, ma200)
        high52w_pct = _calc_pct(current, week52High)
        daily_change: Dict[str, float] = {'value': 0, 'pct': 0}

        # Fetch chart history from FRED for the SPY chart component
        if fred_api_key:
            try:
                url = (f'https://api.stlouisfed.org/fred/series/observations'
                       f'?series_id=SP500&api_key={fred_api_key}'
                       f'&file_type=json&observation_start=2010-01-01&limit=5000&sort_order=asc')
                r = requests.get(url, timeout=30, headers=_HEADERS)
                fred_rows = [{'date': o['date'], 'close': float(o['value'])}
                             for o in r.json().get('observations', []) if o['value'] != '.']
                if len(fred_rows) >= 200:
                    closes = pd.Series([row['close'] for row in fred_rows])
                    ma200_arr = closes.rolling(200).mean()
                    ma50_arr = closes.rolling(50).mean()
                    for i in range(199, len(fred_rows)):
                        chart_history.append({
                            'date': fred_rows[i]['date'],
                            'price': fred_rows[i]['close'],
                            'ma50': round(float(ma50_arr.iloc[i]) * 100) / 100,
                            'ma200': round(float(ma200_arr.iloc[i]) * 100) / 100,
                        })
            except Exception as e:
                print(f'[SPY] Chart history fetch failed: {e}')
    else:
        current = rows[-1]['close']
        prev_close = rows[-2]['close']
        
        # Override with Finnhub spot price to guarantee live data if Polygon metrics are stale
        # Finnhub 'c' (current) / 'pc' (previous close)
        if finnhub_api_key:
            try:
                fh = _fetch_finnhub_quote('SPY', finnhub_api_key)
                if fh and fh.get('current') and fh['current'] > 0:
                    current = fh['current']
                    # Ensure prev_close correctly targets the actual previous close depending on date staleness
                    # Finnhub handles "previous close" identically in `fh['dailyChange']['value']` via pc reference.
                    # so we calculate backwards from the dailyChange.value!
                    prev_close = current - fh['dailyChange']['value']
                    data_source += " + Finnhub Spot"
            except Exception:
                pass

        daily_change = {'value': current - prev_close, 'pct': _calc_pct(current, prev_close)}

        closes = pd.Series([row['close'] for row in rows])
        ma200 = float(closes.tail(200).mean()) if len(rows) >= 200 else float(closes.mean())
        ma200_pct = _calc_pct(current, ma200)

        week52High = float(closes.tail(252).max())
        high52w_pct = _calc_pct(current, week52High)

        rsi = float(calculate_rsi(closes, period=9))

        days3y = min(756, len(rows))
        return3y = _calc_pct(current, rows[-days3y]['close'])

        ma200_arr = closes.rolling(200).mean()
        ma50_arr = closes.rolling(50).mean()
        for i in range(199, len(rows)):
            chart_history.append({
                'date': rows[i]['date'],
                'price': rows[i]['close'],
                'ma50': round(float(ma50_arr.iloc[i]) * 100) / 100,
                'ma200': round(float(ma200_arr.iloc[i]) * 100) / 100,
            })

    return {
        'current': current,
        'dailyChange': daily_change,
        'ma200': {'value': ma200, 'pct': ma200_pct},
        'week52High': {'value': week52High, 'pct': high52w_pct},
        'rsi': rsi,
        'return3y': return3y,
        'chartHistory': chart_history,
        '_meta': {'source': data_source, 'hasErrors': False, 'messages': [f'Loaded from {data_source}']},
    }


def fetch_spy_daily_move() -> Dict[str, Any]:
    """
    Fetch the SPY daily move percentage from Google Sheets cell B12.
    Returns JSON matching the Next.js /api/spy-daily-move response shape.
    """
    try:
        r = requests.get(URLS['SPY_DAILY_MOVE'], timeout=10, headers=_HEADERS)
        rows = list(csv.reader(StringIO(r.text)))
        value = rows[11][1].strip() if len(rows) > 11 and len(rows[11]) > 1 else None
        print(f'[spy-daily-move] B12 value: {value}')
        return {'value': value, 'source': 'Google Sheets'}
    except Exception as e:
        print(f'[spy-daily-move] Error: {e}')
        return {'value': None, 'source': 'Failed', 'error': str(e)}


# API Gateway HTTP APIs hard-cap every integration at 30 s and answer 503 past it;
# the dashboard then silently falls back to direct sources. Leave headroom for the
# gateway hop and JSON serialisation.
MARKET_EXTRA_DEADLINE_SECONDS = 22.0


def fetch_market_extra(fred_api_key: str,
                       polygon_api_key: Optional[str] = None,
                       finnhub_api_key: Optional[str] = None,
                       deadline_seconds: float = MARKET_EXTRA_DEADLINE_SECONDS) -> Dict[str, Any]:
    """
    Fetch FX rates, commodities, rates, and real-estate data.
    Returns JSON matching the Next.js /api/market-extra response shape.

    Layer 1: yfinance (everything it covers, with history)
    Layer 2: Polygon (FX with history, BTC, Gold)
    Layer 3: CNBC (keyless; live quote + 1Y daily bars for gold, WTI, USD/CAD, USD/INR,
             BTC). Added 2026-09-01 because yfinance is Yahoo-rate-limited for hours at a
             time and Polygon 429s on the free tier, leaving only spot-only fallbacks.
    Layer 4: FRED (FX, Oil, Gold London PM fix, rates, real-estate, all with history)
             + Frankfurter time-series (FX with history) + Coinbase candles (BTC history)
    Layer 5: spot last resorts: Finnhub BINANCE:BTCUSDT (its OANDA:* forex/commodity
             symbols are paid-tier, 403 forever, not requested), USD spot chain
             (ER-API -> Frankfurter -> Fawaz; BDT, DXY), gold-api.com (Gold), Coinbase (BTC).
             (Stooq, the old last resort, now sits behind a JS proof-of-work challenge.)

    The six provider chains are independent, so they run CONCURRENTLY and the whole
    fetch is bounded by `deadline_seconds`; a chain still running at the deadline is
    skipped (its metrics resolve from the next source in the waterfall) and noted in
    `_meta.messages`. WHY: run strictly one after another, the ~25 upstream calls
    (each with a 10-20 s timeout) averaged 10 s and exceeded 30 s on ~6% of Lambda
    invocations (max 59 s). API Gateway HTTP APIs cut the integration at 30 s with a
    503, so on any slow-upstream day the dashboard quietly served from its fallbacks
    while every check stayed green (health check `lambda_primary_path`, 2026-09-01).
    Calls WITHIN a chain stay sequential: those pauses are per-provider rate-limit
    etiquette (FRED 429s, Polygon's 5/min free tier).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    if not fred_api_key:
        raise ValueError('FRED_API_KEY not configured')

    late_messages: List[str] = []  # surfaced in _meta.messages (deadline skips, Polygon 429)

    # 1. FRED -- 3 per batch with 300 ms gap to avoid 429
    def fred_chain() -> List[Any]:
        fred_spec = [
            ('MORTGAGE30US', 30),    # 0 - 30Y mortgage rate
            ('CUUR0000SEHA', 30),    # 1 - shelter/rent CPI
            ('MSPUS', 5),            # 2 - median home price
            ('DCOILWTICO', 35),      # 3 - WTI crude oil
            ('DGS10', 35),           # 4 - 10Y treasury yield
            ('DGS2', 35),            # 5 - 2Y treasury yield
            ('DEXCAUS', 35),         # 6 - USD/CAD
            ('DEXINUS', 35),         # 7 - USD/INR
            ('ATNHPIUS39300Q', 35),  # 8 - All-Trans House Price Index
            ('GOLDPMGBD228NLBM', 35),  # 9 - Gold, London PM fix (USD/oz)
        ]
        raw: List[Any] = []
        for i in range(0, len(fred_spec), 3):
            for j in range(i, min(i + 3, len(fred_spec))):
                series_id, limit = fred_spec[j]
                try:
                    raw.append(_fetch_fred_series(series_id, fred_api_key, limit))
                except Exception as e:
                    print(f'[FRED] {series_id} failed: {e}')
                    raw.append([])
            if i + 3 < len(fred_spec):
                time.sleep(0.3)
        return raw

    # 2. ExchangeRate-API (free, no key) for BDT pairs and DXY
    def er_chain() -> Optional[dict]:
        try:
            return _fetch_exchange_rates()
        except Exception as e:
            print(f'[ExchangeRate] failed: {e}')
            return None

    # 3. yfinance -- highest priority for everything it covers
    def yf_chain() -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for sym, var_name, invert in [
            ('USDCAD=X', 'usdcad', False),
            ('USDINR=X', 'usdinr', False),
            ('BTC-USD',  'btc',    False),
            ('GC=F',     'gold',   False),
            ('CL=F',     'oil',    False),
        ]:
            try:
                result = _fetch_yfinance(sym, invert=invert, days=400)
                if result and result.get('current'):
                    out[var_name] = result
                    print(f'[yfinance] {sym}: {result["current"]}')
            except Exception as e:
                print(f'[yfinance] {sym} failed: {e}')
            time.sleep(0.1)
        return out

    # 4. Polygon -- FX with history, BTC, Gold (staggered; cached; stops at the first 429)
    def poly_chain() -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        if not polygon_api_key:
            return out
        symbols = [('C:USDCAD', 'usdcad'), ('C:USDINR', 'usdinr'), ('X:BTCUSD', 'btc'), ('C:XAUUSD', 'gold')]
        for i, (sym, var_name) in enumerate(symbols):
            if i:
                time.sleep(POLYGON_STAGGER_SECONDS)
            try:
                result = _fetch_polygon_aggs(sym, polygon_api_key, days=400)
                if result and result.get('current'):
                    out[var_name] = result
                    print(f'[Polygon] {sym}: {result["current"]}')
            except PolygonRateLimited as e:
                skipped = [s_ for s_, _ in symbols[i:]]
                print(f'[Polygon] {e} -- quota spent, skipping {", ".join(skipped)}')
                late_messages.append(f'Polygon 429: skipped {", ".join(skipped)}')
                break
            except Exception as e:
                print(f'[Polygon] {sym} failed: {e}')
        return out

    # 5. Finnhub -- BTC spot only (see docstring: OANDA:* symbols are paid-tier)
    def fh_chain() -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        if not finnhub_api_key:
            return out
        try:
            result = _fetch_finnhub_quote('BINANCE:BTCUSDT', finnhub_api_key)
            if result and result.get('current'):
                out['btc'] = result
                print(f'[Finnhub] BINANCE:BTCUSDT: {result["current"]}')
        except Exception as e:
            print(f'[Finnhub] BINANCE:BTCUSDT failed: {e}')
        return out

    # 6. CNBC -- live quote + daily history, one batched quote call + one history call each
    CNBC_SYMBOLS = {'gold': '@GC.1', 'oil': '@CL.1', 'usdcad': 'CAD=', 'usdinr': 'INR=', 'btc': 'BTC.CB='}

    def cnbc_chain() -> Dict[str, Dict]:
        try:
            got = _fetch_cnbc_metrics(list(CNBC_SYMBOLS.values()))
        except Exception as e:
            print(f'[CNBC] failed: {e}')
            return {}
        out = {name: got[sym] for name, sym in CNBC_SYMBOLS.items() if got.get(sym, {}).get('current')}
        for name in out:
            print(f'[CNBC] {CNBC_SYMBOLS[name]}: {out[name]["current"]}')
        return out

    # 7. Frankfurter time-series -- FX history when yfinance/Polygon/CNBC/FRED all miss
    def frankfurter_chain() -> Dict[str, Dict]:
        try:
            return _fetch_frankfurter_series(['CAD', 'INR'])
        except Exception as e:
            print(f'[Frankfurter] series failed: {e}')
            return {}

    # 8. Keyless last resorts
    def gold_api_chain() -> Optional[Dict]:
        try:
            return _fetch_gold_api()
        except Exception as e:
            print(f'[gold-api] failed: {e}')
            return None

    def coinbase_chain() -> Dict[str, Optional[Dict]]:
        out: Dict[str, Optional[Dict]] = {'candles': None, 'spot': None}
        try:
            out['candles'] = _fetch_coinbase_candles()
        except Exception as e:
            print(f'[Coinbase] candles failed: {e}')
        if not out['candles']:
            try:
                out['spot'] = _fetch_coinbase_spot()
            except Exception as e:
                print(f'[Coinbase] spot failed: {e}')
        return out

    chains = {'FRED': fred_chain, 'ER-API': er_chain, 'yfinance': yf_chain,
              'Polygon': poly_chain, 'Finnhub': fh_chain, 'CNBC': cnbc_chain,
              'Frankfurter': frankfurter_chain, 'gold-api': gold_api_chain, 'Coinbase': coinbase_chain}
    results: Dict[str, Any] = {'FRED': [], 'ER-API': None, 'yfinance': {}, 'Polygon': {}, 'Finnhub': {},
                               'CNBC': {}, 'Frankfurter': {}, 'gold-api': None, 'Coinbase': {}}
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=len(chains), thread_name_prefix='market-extra')
    futures = {name: pool.submit(fn) for name, fn in chains.items()}
    for name, fut in futures.items():
        remaining = max(0.0, deadline_seconds - (time.monotonic() - started))
        try:
            results[name] = fut.result(timeout=remaining)
        except FutureTimeout:
            print(f'[market-extra] {name} still running at the {deadline_seconds:.0f}s '
                  f'deadline -- skipped, resolving from the next source')
            late_messages.append(f'{name} skipped: deadline {deadline_seconds:.0f}s')
        except Exception as e:
            print(f'[market-extra] {name} chain failed: {e}')
    # Don't block on stragglers; they finish (or hit their own HTTP timeout) in the
    # background and are simply discarded.
    pool.shutdown(wait=False, cancel_futures=True)

    fred_raw: List[Any] = list(results['FRED'] or [])
    while len(fred_raw) < 10:
        fred_raw.append([])
    mortgage_data = fred_raw[0]
    rent_data     = fred_raw[1]
    home_data     = fred_raw[2]
    oil_data      = fred_raw[3]
    tnx_data      = fred_raw[4]
    t2y_data      = fred_raw[5]
    cad_data      = fred_raw[6]
    inr_data      = fred_raw[7]
    atnhpi_data   = fred_raw[8]
    gold_fix_data = fred_raw[9]

    er_rates: Optional[dict] = results['ER-API']

    _yf = results['yfinance'] or {}
    yf_usdcad: Optional[Dict] = _yf.get('usdcad')
    yf_usdinr: Optional[Dict] = _yf.get('usdinr')
    yf_btc: Optional[Dict] = _yf.get('btc')
    yf_gold: Optional[Dict] = _yf.get('gold')
    yf_oil: Optional[Dict] = _yf.get('oil')

    _poly = results['Polygon'] or {}
    poly_usdcad: Optional[Dict] = _poly.get('usdcad')
    poly_usdinr: Optional[Dict] = _poly.get('usdinr')
    poly_btc: Optional[Dict] = _poly.get('btc')
    poly_gold: Optional[Dict] = _poly.get('gold')

    fh_btc: Optional[Dict] = (results['Finnhub'] or {}).get('btc')

    _cnbc = results['CNBC'] or {}
    cnbc_usdcad: Optional[Dict] = _cnbc.get('usdcad')
    cnbc_usdinr: Optional[Dict] = _cnbc.get('usdinr')
    cnbc_gold: Optional[Dict] = _cnbc.get('gold')
    cnbc_oil: Optional[Dict] = _cnbc.get('oil')
    cnbc_btc: Optional[Dict] = _cnbc.get('btc')

    _fx_series = results['Frankfurter'] or {}
    fr_usdcad: Optional[Dict] = _fx_series.get('CAD')
    fr_usdinr: Optional[Dict] = _fx_series.get('INR')

    gold_spot: Optional[Dict] = results['gold-api']
    _cb = results['Coinbase'] or {}
    btc_candles: Optional[Dict] = _cb.get('candles')
    btc_coinbase: Optional[Dict] = _cb.get('spot')

    # 7. Compute spot-only values from ER-API
    bdt_rate = er_rates.get('BDT') if er_rates else None
    inr_rate = er_rates.get('INR') if er_rates else None
    cad_rate = er_rates.get('CAD') if er_rates else None
    dxy_value = _compute_dxy(er_rates) if er_rates else None

    usdbdt_primary = _spot_only(bdt_rate)
    inrbdt_primary = _spot_only(bdt_rate / inr_rate) if (bdt_rate and inr_rate) else None
    cadbdt_primary = _spot_only(bdt_rate / cad_rate) if (bdt_rate and cad_rate) else None
    dxy_primary = _spot_only(dxy_value)

    # 8. Standardize FRED series
    usdcad_fred = _standardize_fred(cad_data)
    usdinr_fred = _standardize_fred(inr_data)
    cadinr_fred = _cross_rate_fred(cad_data, inr_data)
    cl_fred = _standardize_fred(oil_data)
    tnx_fred = _standardize_fred(tnx_data)
    t2y_fred = _standardize_fred(t2y_data)
    mort_std = _standardize_fred(mortgage_data)
    rent_std = _standardize_fred(rent_data, multiplier=4.41)
    atnhpi_std = _standardize_fred(atnhpi_data)
    gold_fred = _standardize_fred(gold_fix_data)

    # 9. Build final values: yfinance → Polygon → Finnhub → FRED/ER-API → keyless spot
    source_log: Dict[str, str] = {}

    def _resolve(key: str, *candidates: Optional[Dict]) -> Optional[Dict]:
        """Pick first non-None candidate, log the source."""
        labels = {
            id(yf_usdcad): 'yfinance', id(yf_usdinr): 'yfinance',
            id(yf_btc): 'yfinance', id(yf_gold): 'yfinance', id(yf_oil): 'yfinance',
            id(poly_usdcad): 'Polygon', id(poly_usdinr): 'Polygon',
            id(poly_btc): 'Polygon', id(poly_gold): 'Polygon',
            id(fh_btc): 'Finnhub',
            id(cnbc_usdcad): 'CNBC', id(cnbc_usdinr): 'CNBC', id(cnbc_gold): 'CNBC',
            id(cnbc_oil): 'CNBC', id(cnbc_btc): 'CNBC',
            id(usdcad_fred): 'FRED', id(usdinr_fred): 'FRED', id(gold_fred): 'FRED',
            id(cl_fred): 'FRED', id(tnx_fred): 'FRED', id(t2y_fred): 'FRED',
            id(fr_usdcad): 'Frankfurter', id(fr_usdinr): 'Frankfurter',
            id(gold_spot): 'gold-api', id(btc_candles): 'Coinbase', id(btc_coinbase): 'Coinbase',
        }
        for c in candidates:
            if c and c.get('current') is not None:
                source_log[key] = labels.get(id(c), 'ER-API')
                return c
        source_log[key] = 'null'
        return None

    # USD/CAD: yfinance → Polygon → CNBC → FRED (stale-check) → Frankfurter series → spot chain
    usdcad = _resolve('usdcad', yf_usdcad, poly_usdcad, cnbc_usdcad)
    if usdcad is None:
        if usdcad_fred and not _is_stale(usdcad_fred):
            usdcad = usdcad_fred
            source_log['usdcad'] = 'FRED'
        elif fr_usdcad and fr_usdcad.get('current'):
            usdcad = fr_usdcad
            source_log['usdcad'] = 'Frankfurter'
        elif cad_rate:
            usdcad = _spot_only(cad_rate)
            source_log['usdcad'] = 'ER-API'

    # USD/INR: yfinance → Polygon → CNBC → FRED (stale-check) → Frankfurter series → spot chain
    usdinr = _resolve('usdinr', yf_usdinr, poly_usdinr, cnbc_usdinr)
    if usdinr is None:
        if usdinr_fred and not _is_stale(usdinr_fred):
            usdinr = usdinr_fred
            source_log['usdinr'] = 'FRED'
        elif fr_usdinr and fr_usdinr.get('current'):
            usdinr = fr_usdinr
            source_log['usdinr'] = 'Frankfurter'
        elif inr_rate:
            usdinr = _spot_only(inr_rate)
            source_log['usdinr'] = 'ER-API'

    # BDT pairs — only ER-API has BDT
    if usdbdt_primary and usdbdt_primary.get('current') is not None:
        usdbdt = usdbdt_primary
        source_log['usdbdt'] = 'ER-API'
    else:
        usdbdt = None
        source_log['usdbdt'] = 'null'

    if inrbdt_primary and inrbdt_primary.get('current') is not None:
        inrbdt = inrbdt_primary
        source_log['inrbdt'] = 'ER-API'
    else:
        inrbdt = None
        source_log['inrbdt'] = 'null'

    if cadbdt_primary and cadbdt_primary.get('current') is not None:
        cadbdt = cadbdt_primary
        source_log['cadbdt'] = 'ER-API'
    else:
        cadbdt = None
        source_log['cadbdt'] = 'null'

    if dxy_primary and dxy_primary.get('current') is not None:
        dxy = dxy_primary
        source_log['dxy'] = 'ER-API'
    else:
        dxy = None
        source_log['dxy'] = 'null'

    # CAD/INR: FRED cross-rate → computed from live USD pairs
    cadinr: Optional[Dict] = None
    if cadinr_fred and cadinr_fred.get('current') is not None and not _is_stale(cadinr_fred):
        cadinr = cadinr_fred
        source_log['cadinr'] = 'FRED'
    elif usdcad and usdinr and usdcad.get('current') and usdinr.get('current'):
        cadinr = _spot_only(usdinr['current'] / usdcad['current'])
        source_log['cadinr'] = 'computed'
    else:
        source_log['cadinr'] = 'null'

    # Oil: yfinance (WTI CL=F) → CNBC @CL.1 (live, with history) → FRED DCOILWTICO
    cl = _resolve('cl', yf_oil, cnbc_oil, cl_fred)

    # BTC: yfinance → Polygon → CNBC → Coinbase candles → Finnhub spot → Coinbase spot
    btc = _resolve('btc', yf_btc, poly_btc, cnbc_btc, btc_candles, fh_btc, btc_coinbase)

    # Gold: yfinance → Polygon → CNBC @GC.1 → FRED London PM fix (stale-check) → gold-api spot
    gold = _resolve('gold', yf_gold, poly_gold, cnbc_gold,
                    gold_fred if (gold_fred and not _is_stale(gold_fred)) else None,
                    gold_spot)

    tnx = _resolve('tnx', tnx_fred)
    t2y = _resolve('t2y', t2y_fred)

    # 7. Compute mortgage payment (principal × 80% LTV, 30-year fixed)
    mort_payment: Optional[Dict] = None
    if home_data and mortgage_data:
        try:
            principal = home_data[0]['value'] * 0.80
            hist = []
            for m in reversed(mortgage_data):
                r_monthly = (m['value'] / 100) / 12
                pmt = ((principal * r_monthly * (1 + r_monthly) ** 360)
                       / ((1 + r_monthly) ** 360 - 1)) if r_monthly > 0 else principal / 360
                hist.append({'date': m['date'], 'price': round(pmt, 2)})
            if len(hist) >= 2:
                cur = hist[-1]['price']
                prv = hist[-2]['price']
                mort_payment = {
                    'current': cur,
                    'dailyChange': {'value': round(cur - prv, 2), 'pct': _calc_pct(cur, prv)},
                    'history': hist,
                }
        except Exception as e:
            print(f'[MORT] payment computation failed: {e}')

    # 10. Build _meta summary
    null_count = sum(1 for v in source_log.values() if 'null' in v)
    msgs = []
    for src in ['yfinance', 'Polygon', 'CNBC', 'Finnhub', 'FRED', 'Frankfurter', 'ER-API', 'gold-api', 'Coinbase']:
        n = sum(1 for v in source_log.values() if v == src)
        if n:
            msgs.append(f'{src}: {n}')
    if null_count:
        msgs.append(f'unavailable: {null_count} metrics')
    msgs.extend(late_messages)

    return {
        'fx': {'usdcad': usdcad, 'usdinr': usdinr, 'usdbdt': usdbdt,
               'inrbdt': inrbdt, 'cadinr': cadinr, 'cadbdt': cadbdt, 'dxy': dxy},
        'commodities': {'cl': cl, 'gc': gold, 'btc': btc},
        'rates': {'tnx': tnx, 't2y': t2y, 'mortgageRate': mort_std},
        'realEstate': {'rentIndex': rent_std, 'mortgagePayment': mort_payment, 'atnhpi': atnhpi_std},
        '_meta': {
            'source': 'yfinance/Polygon/Finnhub/FRED/ER-API',
            'hasErrors': null_count > 0,
            'sourceLog': source_log,
            'messages': msgs,
        },
    }


def fetch_polymarket_trending(limit: int = 8) -> List[Dict[str, Any]]:
    """
    Curated "market sentiment" board: meaningful-probability (8-92%), non-sports, binary
    Yes/No Polymarket markets, de-duped by event and topic-diverse, ranked by volume.
    Uses the public Gamma REST API directly — no API key required.

    Returns: [{name, odds, volume, change, topic, topicEmoji, endDate, eventSlug}, ...]
             ([] on any failure — graceful degradation, never raises).
    """
    # Sports keywords to filter out (checked against question, slug, and event titles)
    SPORTS_KEYWORDS = {
        # Betting notation patterns
        'spread:', 'o/u', 'over/under', 'exact score', 'moneyline', 'parlay',
        # US Sports
        'nfl', 'nba', 'nhl', 'mlb', 'nba draft', 'nfl draft', 'mlb draft', 'ncaa', 'march madness',
        'super bowl', 'world series', 'stanley cup', 'nba finals', 'nfl playoffs',
        # European Football/Soccer
        'fifa', 'world cup', 'champions league', 'premier league', 'laliga', 'bundesliga',
        'serie a', 'ligue 1', 'europa league', 'conference league', 'super league',
        'arsenal', 'manchester', 'chelsea', 'tottenham', 'liverpool', 'real madrid',
        'barcelona', 'juventus', 'ac milan', 'psg', 'bayern', 'borussia', 'dortmund',
        'atletico', 'napoli', 'ajax', 'celtic', 'rangers', 'galatasaray', 'roma',
        'lazio', 'fiorentina', 'inter', 'marseille', 'monaco', 'rennes', 'lyon',
        # South American Football/Soccer
        'palmeiras', 'flamengo', 'santos', 'vasco', 'corinthians', 'sao paulo', 'gremio',
        'internacional', 'cruzeiro', 'atletico mineiro', 'river plate', 'boca juniors',
        'independiente', 'racing', 'velez', 'deportivo cali', 'america', 'chivas',
        # Asian Football/Soccer
        'fc', 'fc.', ' vs ', ' vs.', 'kagoshima', 'kyoto', 'grampus', 'nagoya', 'lanús',
        'lecce', 'pisa', 'ready', 'villarreal', 'betis', 'getafe', 'girona',
        # Other Sports
        'tennis', 'wimbledon', 'us open', 'french open', 'australian open', 'atp', 'wta',
        'golf', 'masters', 'us pga', 'the open', 'ryder cup', 'pga tour',
        'cricket', 'ipl', 'bpl', 'rugby', 'super rugby', 'six nations',
        'rugby world cup', 'nrl', 'afl', 'australian football',
        'formula 1', 'f1', 'formula e', 'motogp', 'moto2', 'moto3',
        'mma', 'ufc', 'boxing', 'wwe', 'wrestling', 'esports', 'dota', 'valorant', 'lol',
        'basketball', 'soccer', 'football', 'baseball', 'hockey', 'ice hockey',
        'pfa player', 'golden ball', 'player of the year', 'manager of the year',
        'ballon dor', 'coach of the year', 'rookie of the year',
        # Gaming / Esports
        'counter-strike', 'cs:go', 'cs2', 'call of duty', 'cod', 'overwatch',
        'starcraft', 'sc2', 'pubg', 'fortnite', 'minecraft', 'twitch', 'gaming',
        'streamer', 'esports tournament', 'esports league', 'fps', 'moba',
        'map 1', 'map 2', 'map 3', 'odd/even total kills', 'total rounds',
        'eternal premium', 'bushido', 'immortals', 'fnatic', 'heroic', 'astralis'
    }

    # Topic tagging (first keyword match wins; order matters — specific before generic).
    TOPIC_KEYWORDS = (
        ("Crypto", "🪙", ("bitcoin", "btc", "ethereum", " eth ", "crypto", "microstrategy",
                          "solana", "coinbase", "stablecoin", "dogecoin", "xrp", "binance")),
        ("Geopolitics", "🌍", ("iran", "israel", "gaza", "ukraine", "russia", "china", "taiwan",
                              "ceasefire", "nuclear", "nato", "hormuz", "north korea", "hostage",
                              "peace deal", " war ", "missile", "sanction", "venezuela", "houthi")),
        ("Politics", "🏛️", ("trump", "biden", "election", "president", "senate", "congress",
                            "governor", "democrat", "republican", "nominee", "primary",
                            "supreme court", "impeach", "cabinet", "vance", "mayor", "parliament")),
        ("Tech", "🤖", ("openai", "anthropic", "gpt", "nvidia", "spacex", "tesla", "apple",
                       "google", "microsoft", "ipo", "chatgpt", "claude", "llm", "agi",
                       "artificial intelligence", "starship", "robotaxi", "quantum")),
        ("Economy", "📉", ("recession", "fed ", "rate cut", "inflation", "gdp", "unemployment",
                          "s&p", "interest rate", "market cap", "valuation", "jobs report")),
    )

    def topic_of(text):
        for tname, emoji, kws in TOPIC_KEYWORDS:
            if any(kw in text for kw in kws):
                return tname, emoji
        return "World", "🌐"

    try:
        url = "https://gamma-api.polymarket.com/markets"
        # Broad pool by WEEKLY volume (recent interest, less churny than 24h). The Gamma API
        # caps `limit` at 100, so paginate to widen the pool, then filter hard below.
        markets = []
        for offset in range(0, 500, 100):
            resp = requests.get(url, params={
                "active": "true", "closed": "false", "order": "volume1wk",
                "ascending": "false", "limit": 100, "offset": offset,
            }, timeout=15)
            resp.raise_for_status()
            page = resp.json()
            if not isinstance(page, list) or not page:
                break
            markets.extend(page)

        from datetime import datetime, timezone, timedelta
        from collections import defaultdict
        import ast
        now = datetime.now(timezone.utc)
        min_horizon = now + timedelta(days=1)   # drop intraday churn (e.g. "BTC up/down 5m")

        _MONTHS = ("january", "february", "march", "april", "may", "june", "july",
                   "august", "september", "october", "november", "december")

        def clean_title(t):
            for a, b in (("Democratic", "Dem"), ("Republican", "GOP"),
                         ("Presidential Election Winner", "US President"),
                         ("Presidential Nominee", "Nominee"),
                         ("Presidential Election", "Election")):
                t = t.replace(a, b)
            return " ".join(t.split()).strip()

        def is_candidate_name(git):
            # A real entity (person/company) — not a date/price/level tranche.
            g = (git or "").strip().lower()
            if not g or g[0] in "<>↑↓$0123456789":
                return False
            return not any(g.startswith(mo) for mo in _MONTHS)

        # Group the pool by event so multi-candidate races (elections, etc.) collapse to a
        # single "Event: favorite" row instead of N separate candidate rows.
        by_event = defaultdict(list)
        for m in markets:
            ev = (m.get("events") or [{}])[0]
            if not isinstance(ev, dict):
                ev = {}
            key = ev.get("ticker") or ev.get("slug") or m.get("slug") or m.get("question") or id(m)
            by_event[key].append((m, ev))

        candidates = []
        for key, members in by_event.items():
            try:
                first_ev = members[0][1]
                ev_title = (first_ev.get("title") or "").strip()

                parsed = []
                for m, _ev in members:
                    end_iso = m.get("endDate")
                    if end_iso:
                        try:
                            if datetime.fromisoformat(end_iso.replace("Z", "+00:00")) < min_horizon:
                                continue
                        except Exception:
                            end_iso = None
                    try:
                        outs = [str(o).lower() for o in ast.literal_eval(m.get("outcomes", "[]"))]
                    except Exception:
                        outs = []
                    if outs != ["yes", "no"]:
                        continue
                    try:
                        odds = float(ast.literal_eval(m.get("outcomePrices", "[]"))[0])
                    except Exception:
                        continue
                    try:
                        chg = float(m["oneMonthPriceChange"]) if m.get("oneMonthPriceChange") is not None else None
                    except Exception:
                        chg = None
                    parsed.append({
                        "odds": odds, "vol": float(m.get("volumeNum") or m.get("volume") or 0),
                        "change": chg, "end": end_iso,
                        "git": (m.get("groupItemTitle") or "").strip(),
                        "q": m.get("question") or "",
                    })
                if not parsed:
                    continue

                # Sports filter (event title + member questions + tags from any member).
                text = (ev_title + " " + " ".join(p["q"] for p in parsed) + " "
                        + (first_ev.get("slug") or "")).lower()
                tag_labels = {(t.get("label") or "").lower()
                              for m, _ev in members for t in (m.get("tags") or []) if isinstance(t, dict)}
                if any("sport" in lbl for lbl in tag_labels) or any(kw in text for kw in SPORTS_KEYWORDS):
                    continue

                topic, emoji = topic_of(text)
                is_multi = len(members) >= 2 and any(p["git"] for p in parsed)

                if is_multi:
                    # Event favorite = highest-odds candidate (the current field leader).
                    fav = max(parsed, key=lambda p: p["odds"])
                    if not (0.05 <= fav["odds"] <= 0.85):
                        continue
                    ev_vol = sum(p["vol"] for p in parsed)
                    if ev_vol < 25000:
                        continue
                    if ev_title and is_candidate_name(fav["git"]):
                        name = f"{clean_title(ev_title)}: {fav['git']}"   # candidate race
                    else:
                        name = fav["q"] or ev_title or "Unknown"          # date/price tranche → self-descriptive
                    if "__" in name:                                      # malformed placeholder question
                        continue
                    chosen, vol, is_event = fav, ev_vol, True
                else:
                    # Standalone binary market: genuine-uncertainty band.
                    p = parsed[0]
                    if not (0.08 <= p["odds"] <= 0.92) or p["vol"] < 25000:
                        continue
                    name, chosen, vol, is_event = (p["q"] or "Unknown"), p, p["vol"], False

                candidates.append({
                    "name": name,
                    "odds": round(chosen["odds"], 2),
                    "volume": vol,
                    "change": round(chosen["change"], 4) if chosen["change"] is not None else None,
                    "topic": topic,
                    "topicEmoji": emoji,
                    "endDate": chosen["end"],
                    "eventSlug": first_ev.get("slug"),
                    "_event": key,
                    "_is_event": is_event,
                })
            except Exception as e:
                logging.warning(f"Polymarket event parse error on {key}: {e}")
                continue

        # Select: rank by volume, de-dupe by event (highest-volume wins), cap 2 per topic,
        # AND cap "longshots" (<30%) so the board is a real SPREAD of sentiment, not a wall
        # of unlikely-but-heavily-traded bets. Pass 1 enforces the longshot cap; pass 2 fills
        # any leftover slots without it (so we still reach `limit` on a quiet day).
        candidates.sort(key=lambda b: b["volume"], reverse=True)
        LONGSHOT, max_longshots = 0.30, max(1, limit // 2)
        seen_events, per_topic, bets, n_longshots = set(), {}, [], 0
        for allow_longshot in (False, True):
            for c in candidates:
                if len(bets) >= limit:
                    break
                if c["_event"] in seen_events or per_topic.get(c["topic"], 0) >= 2:
                    continue
                # Event favorites are exempt from the longshot cap — a field leader is
                # interesting at any %; the cap only tames standalone unlikely-but-traded bets.
                is_longshot = c["odds"] < LONGSHOT and not c["_is_event"]
                if is_longshot and not allow_longshot and n_longshots >= max_longshots:
                    continue
                seen_events.add(c["_event"])
                per_topic[c["topic"]] = per_topic.get(c["topic"], 0) + 1
                n_longshots += 1 if is_longshot else 0
                bets.append({k: v for k, v in c.items() if not k.startswith("_")})
            if len(bets) >= limit:
                break

        return bets

    except Exception as e:
        logging.error(f"Polymarket API error: {e}", exc_info=True)
        return []
