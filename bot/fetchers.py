"""
Data fetching logic for the financial-telegram-bot.
Handles FRED API, Stooq (SPY), and Google Sheets integration.
"""

import csv
import requests
import pandas as pd
import pandas_datareader.data as web
from io import StringIO
from typing import Dict, Any, List, Optional
from bot.config import URLS, RSI_PERIOD

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

        # 4. VIX
        r_vix = requests.get(URLS['VIX'], timeout=10)
        reader_vix = list(csv.reader(StringIO(r_vix.text)))
        vix_current = reader_vix[1][0].strip()
        vix_3m = reader_vix[1][1].strip()
        fear_greed_status = reader_vix[1][2].strip()

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
            f"🎢 VIX: (Current | 3M) : {clean_val(vix_current)} | {clean_val(vix_3m)} | {clean_val(fear_greed_status)}\n"
            f"\n[Financial Dashboard History](https://docs.google.com/spreadsheets/d/1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc/edit?gid=0#gid=0)"
        )
        print("✓ Successfully fetched and parsed Google Sheet indicators")
        return output

    except Exception as e:
        print(f"WARNING: Failed to fetch Google Sheet indicators: {e}")
        return ""

def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> float:
    """Calculate Relative Strength Index (RSI) using Wilder's Smoothing"""
    delta = prices.diff()
    
    # Isolate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate Wilder's smoothed moving average using Exponential Weighted Math (EWM)
    # alpha=1/period is the exact mathematical equivalent to Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi.iloc[-1])

def fetch_spy_stats() -> Dict[str, float]:
    """
    Fetch SPY statistics natively from the Stooq financial database.
    """
    print("Fetching SPY data from Stooq (pandas-datareader)...")

    try:
        df = web.DataReader('SPY.US', 'stooq')
        
        if df.empty:
            raise ValueError("Stooq returned empty data for SPY.US")
            
        df = df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        days_available = min(756, len(df))
        current_price = float(df['Close'].iloc[-1])

        if pd.isna(current_price) or current_price <= 0:
            raise ValueError("Invalid current price data format.")

        ma_200 = float(df['Close'].tail(200).mean()) if len(df) >= 200 else float(df['Close'].mean())
        ma_200_pct = ((current_price - ma_200) / ma_200) * 100

        days_52w = min(252, len(df))
        week_52_high = float(df['Close'].tail(days_52w).max())
        high_52w_pct = ((current_price - week_52_high) / week_52_high) * 100

        rsi_9d = calculate_rsi(df['Close'], period=RSI_PERIOD)

        price_past = float(df['Close'].iloc[-days_available])
        return_3y_pct = ((current_price - price_past) / price_past) * 100

        stats = {
            'current': current_price,
            'ma_200': ma_200,
            'ma_200_pct': ma_200_pct,
            'week_52_high': week_52_high,
            'high_52w_pct': high_52w_pct,
            'rsi_9d': rsi_9d,
            'return_3y_pct': return_3y_pct
        }

        print(f"✓ SPY data fetched successfully")
        return stats

    except Exception as e:
        print(f"ERROR: Failed to fetch SPY data from Stooq: {str(e)}")
        raise


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


def _fetch_exchange_rates() -> Optional[dict]:
    r = requests.get('https://open.er-api.com/v6/latest/USD', timeout=10, headers=_HEADERS)
    r.raise_for_status()
    data = r.json()
    return data.get('rates') if data.get('result') == 'success' else None


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
        from datetime import datetime, timedelta
        start = (datetime.now() - timedelta(days=days + 60)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')
        hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
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


def _fetch_polygon_aggs(symbol: str, api_key: str, days: int = 1500) -> Optional[Dict[str, Any]]:
    """Fetch daily aggregates from Polygon.io — returns standard data shape with history."""
    if not api_key:
        return None
    from datetime import datetime, timedelta
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    url = (f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}'
           f'?apiKey={api_key}&limit=1000&sort=asc&adjusted=true')
    r = requests.get(url, timeout=20, headers=_HEADERS)
    r.raise_for_status()
    data = r.json()
    results = data.get('results', [])
    if len(results) < 2:
        return None
    from datetime import datetime as _dt
    rows = [{'date': _dt.utcfromtimestamp(res['t'] / 1000).strftime('%Y-%m-%d'), 'price': res['c']}
            for res in results]
    current = rows[-1]['price']
    prev = rows[-2]['price']
    return {
        'current': current,
        'dailyChange': {'value': round(current - prev, 4), 'pct': _calc_pct(current, prev)},
        'history': rows,
        'lastDate': rows[-1]['date'],
    }


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


def _fetch_stooq(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        url = f'https://stooq.com/q/d/l/?s={symbol}&i=d'
        r = requests.get(url, timeout=15, headers=_HEADERS)
        lines = r.text.strip().split('\n')
        if len(lines) < 2 or 'No data' in lines[0]:
            return None
        rows: List[Dict[str, Any]] = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    rows.append({'date': parts[0].strip(), 'price': float(parts[4].strip())})
                except ValueError:
                    pass
        if len(rows) < 2:
            return None
        rows = rows[-30:]  # type: ignore[assignment]
        current = rows[-1]['price']
        prev = rows[-2]['price']
        return {
            'current': current,
            'dailyChange': {'value': round(current - prev, 4), 'pct': _calc_pct(current, prev)},
            'history': rows,
        }
    except Exception as e:
        print(f'[Stooq] {symbol}: {e}')
        return None


def fetch_spy_with_fallback(fred_api_key: Optional[str] = None,
                            polygon_api_key: Optional[str] = None,
                            finnhub_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch SPY stats with waterfall fallback.
    Returns JSON matching the Next.js /api/spy response shape.

    Layer 0: Polygon (full history → compute all indicators)
    Layer 1: Google Sheets pre-calculated indicators
    Layer 2: Stooq CSV (raw prices)
    Layer 3: FRED SP500 series
    """
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

    # Layer 2: Stooq CSV
    if not indicators and not rows:
        try:
            r = requests.get(URLS['STOOQ_SPY'], timeout=15, headers=_HEADERS)
            parsed_rows = []
            for line in r.text.strip().split('\n')[1:]:
                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        close = float(parts[4].strip())
                        if close > 0:
                            parsed_rows.append({'date': parts[0].strip(), 'close': close})
                    except ValueError:
                        pass
            if len(parsed_rows) >= 10:
                rows = parsed_rows
                data_source = 'Stooq'
                print(f'[SPY] Layer 2 (Stooq) loaded {len(rows)} rows')
        except Exception as e:
            print(f'[SPY] Layer 2 (Stooq) failed: {e}')

    # Layer 3: FRED SP500
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
                print(f'[SPY] Layer 3 (FRED) loaded {len(rows)} rows')
        except Exception as e:
            print(f'[SPY] Layer 3 (FRED SP500) failed: {e}')

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


def fetch_market_extra(fred_api_key: str,
                       polygon_api_key: Optional[str] = None,
                       finnhub_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch FX rates, commodities, rates, and real-estate data.
    Returns JSON matching the Next.js /api/market-extra response shape.

    Layer 1: Polygon (FX with history, BTC, Gold)
    Layer 2: Finnhub (BTC, Gold, Oil, FX spot)
    Layer 3: FRED (FX with history, Oil, rates, real-estate) + ExchangeRate-API (BDT, DXY)
    Layer 4: Stooq (BTC, Gold last resort)
    """
    import time

    if not fred_api_key:
        raise ValueError('FRED_API_KEY not configured')

    # 1. Batch FRED requests — 3 per batch with 300 ms gap to avoid 429
    fred_spec = [
        ('MORTGAGE30US', 30),    # 0 – 30Y mortgage rate
        ('CUUR0000SEHA', 30),    # 1 – shelter/rent CPI
        ('MSPUS', 5),            # 2 – median home price
        ('DCOILWTICO', 35),      # 3 – WTI crude oil
        ('DGS10', 35),           # 4 – 10Y treasury yield
        ('DGS2', 35),            # 5 – 2Y treasury yield
        ('DEXCAUS', 35),         # 6 – USD/CAD
        ('DEXINUS', 35),         # 7 – USD/INR
        ('ATNHPIUS39300Q', 35),  # 8 – All-Trans House Price Index
    ]
    fred_raw: List[Any] = []
    _fred_spec: List[Any] = list(fred_spec)
    for i in range(0, len(_fred_spec), 3):
        for j in range(i, min(i + 3, len(_fred_spec))):
            series_id, limit = _fred_spec[j]  # type: ignore[misc]
            try:
                fred_raw.append(_fetch_fred_series(series_id, fred_api_key, limit))
            except Exception as e:
                print(f'[FRED] {series_id} failed: {e}')
                fred_raw.append([])
        if i + 3 < len(fred_spec):
            time.sleep(0.3)

    while len(fred_raw) < 9:
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

    # 2. ExchangeRate-API (free, no key) for BDT pairs and DXY
    er_rates: Optional[dict] = None
    try:
        er_rates = _fetch_exchange_rates()
    except Exception as e:
        print(f'[ExchangeRate] failed: {e}')

    # 3. yfinance — highest priority for everything it covers
    yf_usdcad: Optional[Dict] = None
    yf_usdinr: Optional[Dict] = None
    yf_btc: Optional[Dict] = None
    yf_gold: Optional[Dict] = None
    yf_oil: Optional[Dict] = None
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
                if var_name == 'usdcad': yf_usdcad = result
                elif var_name == 'usdinr': yf_usdinr = result
                elif var_name == 'btc': yf_btc = result
                elif var_name == 'gold': yf_gold = result
                elif var_name == 'oil': yf_oil = result
                print(f'[yfinance] {sym}: {result["current"]}')
        except Exception as e:
            print(f'[yfinance] {sym} failed: {e}')
        time.sleep(0.1)

    # 4. Polygon — FX with history, BTC, Gold (fallback to yfinance)
    poly_usdcad: Optional[Dict] = None
    poly_usdinr: Optional[Dict] = None
    poly_btc: Optional[Dict] = None
    poly_gold: Optional[Dict] = None
    if polygon_api_key:
        for sym, var_name in [('C:USDCAD', 'usdcad'), ('C:USDINR', 'usdinr'),
                               ('X:BTCUSD', 'btc'), ('C:XAUUSD', 'gold')]:
            try:
                result = _fetch_polygon_aggs(sym, polygon_api_key, days=400)
                if result and result.get('current'):
                    if var_name == 'usdcad':
                        poly_usdcad = result
                    elif var_name == 'usdinr':
                        poly_usdinr = result
                    elif var_name == 'btc':
                        poly_btc = result
                    elif var_name == 'gold':
                        poly_gold = result
                    print(f'[Polygon] {sym}: {result["current"]}')
            except Exception as e:
                print(f'[Polygon] {sym} failed: {e}')
            time.sleep(0.1)

    # 5. Finnhub — BTC, Gold, Oil, FX spot fallbacks
    fh_btc: Optional[Dict] = None
    fh_gold: Optional[Dict] = None
    fh_oil: Optional[Dict] = None
    fh_usdcad: Optional[Dict] = None
    fh_usdinr: Optional[Dict] = None
    if finnhub_api_key:
        for sym, var_name in [('BINANCE:BTCUSDT', 'btc'), ('OANDA:XAU_USD', 'gold'),
                               ('OANDA:BCO_USD', 'oil'), ('OANDA:USD_CAD', 'usdcad'),
                               ('OANDA:USD_INR', 'usdinr')]:
            try:
                result = _fetch_finnhub_quote(sym, finnhub_api_key)
                if result and result.get('current'):
                    if var_name == 'btc':
                        fh_btc = result
                    elif var_name == 'gold':
                        fh_gold = result
                    elif var_name == 'oil':
                        fh_oil = result
                    elif var_name == 'usdcad':
                        fh_usdcad = result
                    elif var_name == 'usdinr':
                        fh_usdinr = result
                    print(f'[Finnhub] {sym}: {result["current"]}')
            except Exception as e:
                print(f'[Finnhub] {sym} failed: {e}')
            time.sleep(0.1)

    # 6. Stooq for BTC and Gold (last resort)
    time.sleep(0.15)
    btc_stooq = _fetch_stooq('btc.v')
    time.sleep(0.15)
    gold_stooq = _fetch_stooq('xauusd')

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

    # 9. Build final values: yfinance → Polygon → Finnhub → FRED/ER-API → Stooq
    source_log: Dict[str, str] = {}

    def _resolve(key: str, *candidates: Optional[Dict]) -> Optional[Dict]:
        """Pick first non-None candidate, log the source."""
        labels = {
            id(yf_usdcad): 'yfinance', id(yf_usdinr): 'yfinance',
            id(yf_btc): 'yfinance', id(yf_gold): 'yfinance', id(yf_oil): 'yfinance',
            id(poly_usdcad): 'Polygon', id(poly_usdinr): 'Polygon',
            id(poly_btc): 'Polygon', id(poly_gold): 'Polygon',
            id(fh_btc): 'Finnhub', id(fh_gold): 'Finnhub',
            id(fh_oil): 'Finnhub', id(fh_usdcad): 'Finnhub', id(fh_usdinr): 'Finnhub',
            id(usdcad_fred): 'FRED', id(usdinr_fred): 'FRED',
            id(cl_fred): 'FRED', id(tnx_fred): 'FRED', id(t2y_fred): 'FRED',
            id(btc_stooq): 'Stooq', id(gold_stooq): 'Stooq',
        }
        for c in candidates:
            if c and c.get('current') is not None:
                source_log[key] = labels.get(id(c), 'ER-API')
                return c
        source_log[key] = 'null'
        return None

    # USD/CAD: yfinance → Polygon → Finnhub → FRED (stale-check) → ER-API
    usdcad = _resolve('usdcad', yf_usdcad, poly_usdcad, fh_usdcad)
    if usdcad is None:
        if usdcad_fred and not _is_stale(usdcad_fred):
            usdcad = usdcad_fred
            source_log['usdcad'] = 'FRED'
        elif cad_rate:
            usdcad = _spot_only(cad_rate)
            source_log['usdcad'] = 'ER-API'

    # USD/INR: yfinance → Polygon → Finnhub → FRED (stale-check) → ER-API
    usdinr = _resolve('usdinr', yf_usdinr, poly_usdinr, fh_usdinr)
    if usdinr is None:
        if usdinr_fred and not _is_stale(usdinr_fred):
            usdinr = usdinr_fred
            source_log['usdinr'] = 'FRED'
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

    # Oil: yfinance (WTI CL=F) → Finnhub → FRED
    cl = _resolve('cl', yf_oil, fh_oil, cl_fred)

    # BTC: yfinance → Polygon → Finnhub → Stooq
    btc = _resolve('btc', yf_btc, poly_btc, fh_btc, btc_stooq)

    # Gold: yfinance → Polygon → Finnhub → Stooq
    gold = _resolve('gold', yf_gold, poly_gold, fh_gold, gold_stooq)

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
    for src in ['yfinance', 'Polygon', 'Finnhub', 'FRED', 'ER-API', 'Stooq']:
        n = sum(1 for v in source_log.values() if v == src)
        if n:
            msgs.append(f'{src}: {n}')
    if null_count:
        msgs.append(f'unavailable: {null_count} metrics')

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
