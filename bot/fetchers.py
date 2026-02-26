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
        )
        print("✓ Successfully fetched and parsed Google Sheet indicators")
        return output

    except Exception as e:
        print(f"WARNING: Failed to fetch Google Sheet indicators: {e}")
        return ""

def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> float:
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
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
