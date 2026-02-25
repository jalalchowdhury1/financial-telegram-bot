"""
Data fetching logic for the financial-telegram-bot.
Handles FRED API, Stooq (SPY), and Google Sheets integration.
"""

import csv
import requests
import pandas as pd
import pandas_datareader.data as web
from io import StringIO

def fetch_google_sheet_indicators():
    """
    Fetch custom indicator values from assigned Google Sheets via CSV export.
    Returns a rigidly formatted string to be prepended to the Telegram report.
    """
    print("Fetching Google Sheet custom indicators...")
    try:
        # 1. NotSoBoring (gid=0)
        url_nsb = "https://docs.google.com/spreadsheets/d/10Y8Jus8_fMwH9H69vWh7thSzl2hH34Ri3BRbDw_GEgw/export?format=csv&gid=0"
        r_nsb = requests.get(url_nsb, timeout=10)
        reader_nsb = list(csv.reader(StringIO(r_nsb.text)))
        not_so_boring_val = reader_nsb[2][1].strip()  # Row 3 (index 2), Col B (index 1)

        # 2. FrontRunner (gid=1668420064)
        url_fr = "https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=1668420064"
        r_fr = requests.get(url_fr, timeout=10)
        reader_fr = list(csv.reader(StringIO(r_fr.text)))
        front_runner_val = reader_fr[1][0].strip().split('\n')[0].strip()  # Row 2 (index 1), Col A (index 0)

        # 3. AAII Diff (gid=0)
        url_aaii = "https://docs.google.com/spreadsheets/d/1zQQ2am1yhzTwY7nx8xPak4Q0WoNMwxWj7Ekr-fDEIF4/export?format=csv&gid=0"
        r_aaii = requests.get(url_aaii, timeout=10)
        reader_aaii = list(csv.reader(StringIO(r_aaii.text)))
        aaii_val = reader_aaii[1][4].strip()  # Row 2 (index 1), Col E (index 4)

        # 4. VIX (gid=790638481)
        url_vix = "https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=790638481"
        r_vix = requests.get(url_vix, timeout=10)
        reader_vix = list(csv.reader(StringIO(r_vix.text)))
        vix_current = reader_vix[1][0].strip()
        vix_3m = reader_vix[1][1].strip()
        fear_greed_status = reader_vix[1][2].strip()

        # Build exactly mimicking format
        output = (
            f"🛡️ NotSoBoring : {not_so_boring_val}\n\n"
            f"🔑 FrontRunner :  {front_runner_val}\n\n"
            f"🔸 AAII Diff :  {aaii_val} (G | >20% | 6mths out)%\n\n"
            f"🎢 VIX: (Current | 3M) : {vix_current}  | {vix_3m} | {fear_greed_status}\n\n"
        )
        print("✓ Successfully fetched and parsed Google Sheet indicators")
        return output

    except Exception as e:
        print(f"WARNING: Failed to fetch Google Sheet indicators: {e}")
        return ""

def calculate_rsi(prices, period=9):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]

def fetch_spy_stats():
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

        if len(df) < 756:
            days_available = len(df)
        else:
            days_available = 756

        current_price = df['Close'].iloc[-1]

        if pd.isna(current_price) or current_price <= 0:
            raise ValueError("Invalid current price data format.")

        days_for_200d = 200
        if len(df) >= days_for_200d:
            ma_200 = df['Close'].tail(days_for_200d).mean()
        else:
            ma_200 = df['Close'].mean()
        ma_200_pct = ((current_price - ma_200) / ma_200) * 100

        days_52w = min(252, len(df))
        week_52_high = df['Close'].tail(days_52w).max()
        high_52w_pct = ((current_price - week_52_high) / week_52_high) * 100

        rsi_9d = calculate_rsi(df['Close'], period=9)

        if pd.isna(rsi_9d):
            raise ValueError("Invalid RSI calculation generation.")

        price_past = df['Close'].iloc[-days_available]
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
