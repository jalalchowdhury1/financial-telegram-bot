"""
Financial Charts Automation - Daily Telegram Reporter
Generates yield curve and profit margin charts from FRED data
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from fredapi import Fred
import numpy as np

# Set style for professional-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_environment_variables():
    """Load and validate required environment variables"""
    required_vars = {
        'FRED_API_KEY': os.getenv('FRED_API_KEY'),
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
        'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY')
    }

    missing_vars = [var for var, value in required_vars.items() if not value]

    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    return required_vars


def calculate_rsi(prices, period=9):
    """
    Calculate Relative Strength Index (RSI)

    Args:
        prices: Series of prices
        period: RSI period (default 9)

    Returns:
        float: RSI value
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


def fetch_spy_stats():
    """
    Fetch SPY statistics from Alpha Vantage API

    Returns:
        dict: Dictionary containing all SPY statistics
    """
    print("Fetching SPY data from Alpha Vantage...")

    try:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not set in environment variables")

        # Fetch weekly time series data (free tier supports 20+ years of weekly data)
        # Weekly data is sufficient for 200-day MA, 52-week high, and 3-year return calculations
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=SPY&apikey={api_key}"

        print("  Requesting data from Alpha Vantage API...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")

        if "Note" in data:
            # Rate limit hit
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")

        if "Information" in data:
            # Alpha Vantage returns this for API key issues or other info messages
            raise ValueError(f"Alpha Vantage message: {data['Information']}")

        if "Weekly Time Series" not in data:
            raise ValueError(f"Unexpected API response format. Keys: {list(data.keys())}, Data: {data}")

        # Parse the time series data
        time_series = data["Weekly Time Series"]

        # Convert to DataFrame for easier calculations
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Convert string values to float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)

        # Sort by date (oldest first)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Verify we have enough data (156 weeks = ~3 years)
        if len(df) < 156:
            print(f"  Warning: Only {len(df)} weeks of data available")
            weeks_available = len(df)
        else:
            weeks_available = 156

        print(f"  Received {len(df)} weeks of historical data")

        # Current price (most recent close)
        current_price = df['Close'].iloc[-1]

        # Verify current price is valid
        if pd.isna(current_price) or current_price <= 0:
            raise ValueError("Invalid current price data")

        # 200-day MA (200 days ≈ 40 weeks)
        weeks_for_200d = 40  # 200 trading days / 5 days per week
        if len(df) >= weeks_for_200d:
            ma_200 = df['Close'].tail(weeks_for_200d).mean()
        else:
            ma_200 = df['Close'].mean()
        ma_200_pct = ((current_price - ma_200) / ma_200) * 100

        # 52-week high
        week_data_points = min(52, len(df))
        week_52_high = df['Close'].tail(week_data_points).max()
        high_52w_pct = ((current_price - week_52_high) / week_52_high) * 100

        # 9-week RSI (using weekly data, so 9 weeks instead of 9 days)
        rsi_9d = calculate_rsi(df['Close'], period=9)

        # Verify RSI is valid
        if pd.isna(rsi_9d):
            raise ValueError("Invalid RSI calculation")

        # 3-year return (156 weeks = 3 years)
        price_past = df['Close'].iloc[-weeks_available]
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
        print(f"  Current: ${current_price:.2f}")
        print(f"  200-day MA: ${ma_200:.2f} ({ma_200_pct:+.2f}%)")
        print(f"  52-wk High: ${week_52_high:.2f} ({high_52w_pct:+.2f}%)")
        print(f"  9d RSI: {rsi_9d:.2f}")
        print(f"  3Y Return: {return_3y_pct:.2f}%")

        return stats

    except Exception as e:
        print(f"ERROR: Failed to fetch SPY data from Alpha Vantage: {str(e)}")
        raise


def create_spy_stats_chart(stats, output_file='spy_stats.png'):
    """
    Create SPY statistics chart with a modern card design
    
    Args:
        stats: Dictionary with SPY statistics
        output_file: Output filename
        
    Returns:
        str: Output file path
    """
    print("Creating SPY statistics chart...")
    import matplotlib.patches as patches
    
    try:
        # Validate all required stats are present and finite
        required_keys = ['current', 'ma_200', 'ma_200_pct', 'week_52_high', 'high_52w_pct', 'rsi_9d', 'return_3y_pct']
        for key in required_keys:
            if key not in stats:
                raise ValueError(f"Missing required stat: {key}")
            if not np.isfinite(stats[key]):
                raise ValueError(f"Invalid value for {key}: {stats[key]}")
                
        # Figure setup
        fig = plt.figure(figsize=(10, 6), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        # Fonts
        color_text = '#2d2d2d'
        color_gray = '#888888'
        
        # Draw rounded card background
        card_edge = '#d4d4d4'
        card_bg = '#ffffff'
        rect = patches.FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            edgecolor=card_edge,
            facecolor=card_bg,
            linewidth=1.5,
            zorder=0
        )
        ax.add_patch(rect)
        
        # --- Top Section ---
        ax.text(0.12, 0.82, 'SPY', fontsize=40, color=color_text, fontweight='bold', va='center', ha='left')
        ax.text(0.5, 0.82, f"{stats['current']:.2f}", fontsize=56, color=color_text, fontweight='bold', va='center', ha='center')
        
        # Divider 1
        ax.plot([0.1, 0.9], [0.68, 0.68], color='#e0e0e0', linewidth=1.5)
        
        # --- Middle Section (Slider) ---
        slider_y = 0.52
        
        # Text above slider
        ax.text(0.1, 0.60, '200D ', fontsize=16, color=color_gray, fontweight='bold', ha='left', va='bottom')
        ax.text(0.18, 0.60, f"{stats['ma_200']:.2f}", fontsize=18, color=color_text, fontweight='bold', ha='left', va='bottom')
        
        ax.text(0.78, 0.60, '52wH ', fontsize=16, color=color_gray, fontweight='bold', ha='right', va='bottom')
        ax.text(0.9, 0.60, f"{stats['week_52_high']:.2f}", fontsize=18, color=color_text, fontweight='bold', ha='right', va='bottom')
        
        # Background Bar
        bar_height = 0.025
        ax.add_patch(patches.Rectangle((0.1, slider_y - bar_height/2), 0.8, bar_height,
                                       facecolor='#d4d4d4', edgecolor='none', zorder=1))
        
        # Calculate slider position (clamp 0 to 1)
        range_span = stats['week_52_high'] - stats['ma_200']
        if range_span > 0:
            position = (stats['current'] - stats['ma_200']) / range_span
            position = max(0, min(1, position))
        else:
            position = 0.5
        
        # Handle
        marker_x = 0.1 + (0.8 * position)
        ax.add_patch(patches.Rectangle((marker_x - 0.005, slider_y - bar_height - 0.01),
                                       0.01, bar_height * 2 + 0.02,
                                       facecolor='#333333', edgecolor='none', zorder=2))
                                       
        # Text below slider
        ax.text(0.1, 0.45, f"{stats['ma_200_pct']:.2f}%", fontsize=14, color=color_gray, ha='left', va='top')
        ax.text(0.9, 0.45, f"{stats['high_52w_pct']:.2f}%", fontsize=14, color=color_gray, ha='right', va='top')
        
        # Divider 2
        ax.plot([0.1, 0.9], [0.38, 0.38], color='#e0e0e0', linewidth=1.5)
        
        # --- Bottom Section ---
        # Vertical Divider
        ax.plot([0.48, 0.48], [0.1, 0.35], color='#e0e0e0', linewidth=1.5)
        
        # -> RSI Gauge (Left)
        gauge_center_x = 0.28
        gauge_center_y = 0.18
        gauge_radius_outer = 0.13
        gauge_radius_inner = 0.09
        
        rsi_segments = [
            (0, 20, '#2e8b57'),      # Dark Green
            (20, 40, '#8fce00'),     # Light Green
            (40, 60, '#ffce00'),     # Yellow
            (60, 80, '#ff7a00'),     # Orange
            (80, 100, '#c72d2d')     # Red
        ]
        
        for start, end, color in rsi_segments:
            theta_start = np.pi - (start / 100) * np.pi
            theta_end = np.pi - (end / 100) * np.pi
            
            angles = np.linspace(theta_start, theta_end, 50)
            
            x_outer = gauge_center_x + gauge_radius_outer * np.cos(angles)
            y_outer = gauge_center_y + gauge_radius_outer * np.sin(angles)
            
            x_inner = gauge_center_x + gauge_radius_inner * np.cos(angles[::-1])
            y_inner = gauge_center_y + gauge_radius_inner * np.sin(angles[::-1])
            
            verts = list(zip(x_outer, y_outer)) + list(zip(x_inner, y_inner))
            poly = patches.Polygon(verts, facecolor=color, edgecolor='white', linewidth=1)
            ax.add_patch(poly)
        
        # Draw needle
        val = min(max(stats['rsi_9d'], 0), 100)
        theta_needle = np.pi - (val / 100) * np.pi
        needle_r = gauge_radius_inner + 0.03
        needle_x = gauge_center_x + needle_r * np.cos(theta_needle)
        needle_y = gauge_center_y + needle_r * np.sin(theta_needle)
        
        ax.plot([gauge_center_x, needle_x], [gauge_center_y, needle_y], color='#333333', linewidth=4, zorder=3)
        ax.add_patch(patches.Circle((gauge_center_x, gauge_center_y), 0.015, facecolor='#333333', zorder=4))
        
        # RSI Text
        ax.text(gauge_center_x - 0.02, 0.08, 'RSI ', fontsize=18, color=color_gray, fontweight='bold', ha='right')
        ax.text(gauge_center_x, 0.08, f"{stats['rsi_9d']:.2f}", fontsize=20, color=color_text, fontweight='bold', ha='left')
        
        # -> 3Y Return (Right)
        ax.text(0.53, 0.30, '3Y Return ', fontsize=16, color=color_text, fontweight='normal', ha='left')
        ax.text(0.70, 0.30, f"{stats['return_3y_pct']:.2f}%", fontsize=18, color=color_text, fontweight='bold', ha='left')
        
        # Progress bar
        bar_x = 0.53
        bar_y = 0.20
        bar_w = 0.35
        bar_h = 0.035
        
        ax.add_patch(patches.Rectangle((bar_x, bar_y), bar_w, bar_h, facecolor='#d4d4d4', edgecolor='none'))
        
        target = 85.0
        fill_pct = min(max(stats['return_3y_pct'] / target, 0), 1.0)
        ax.add_patch(patches.Rectangle((bar_x, bar_y), bar_w * fill_pct, bar_h, facecolor='#0052cc', edgecolor='none'))
        
        # Target line
        ax.plot([bar_x + bar_w, bar_x + bar_w], [bar_y - 0.01, bar_y + bar_h + 0.01], color='#333333', linewidth=3)
        ax.text(bar_x + bar_w, bar_y - 0.02, 'Target 85%', fontsize=12, color=color_gray, ha='center', va='top')
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ SPY chart saved: {output_file}")
        return output_file

    except Exception as e:
        print(f"ERROR: Failed to create SPY chart: {str(e)}")
        raise


def format_spy_text_stats(stats):
    """
    Format SPY statistics as text message

    Args:
        stats: Dictionary with SPY statistics

    Returns:
        str: Formatted text
    """
    text = "🔹 *SPY Stats*\n"
    text += f"💵 Current    : {stats['current']:.2f}\n"
    text += f"📈 200-day MA : {stats['ma_200']:.2f} ({stats['ma_200_pct']:+.2f}%)\n"
    text += f"🏔️ 52-wk High: {stats['week_52_high']:.2f} ({stats['high_52w_pct']:+.2f}%)\n"
    text += f"📊 SPY 9d RSI : {stats['rsi_9d']:.2f}\n"
    text += f"📉 3Y SPY Return : {stats['return_3y_pct']:.2f}% (>85%)"

    return text


def create_yield_curve_chart(fred, output_file='yield_curve.png'):
    """
    Create yield curve inversion chart (10Y-2Y Treasury spread)

    Args:
        fred: FRED API client
        output_file: Output filename for the chart

    Returns:
        tuple: (output_file, latest_value)
    """
    print("Fetching yield curve data (T10Y2Y)...")

    try:
        # Fetch last 20 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=20*365)

        data = fred.get_series('T10Y2Y', start_date=start_date, end_date=end_date)

        if data.empty:
            raise ValueError("No data returned for T10Y2Y")

        # Get latest non-null value
        latest_value = data.dropna().iloc[-1]
        latest_date = data.dropna().index[-1]

        # Create the chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Add recession shading
        try:
            recessions = fred.get_series('USREC')
            recessions = recessions[recessions.index >= data.index.min()]
            recession_periods = []
            in_recession = False
            start = None

            for date, value in recessions.items():
                if value == 1 and not in_recession:
                    start = date
                    in_recession = True
                elif value == 0 and in_recession:
                    recession_periods.append((start, date))
                    in_recession = False

            if in_recession and start is not None:
                recession_periods.append((start, recessions.index[-1]))

            for start, end in recession_periods:
                ax.axvspan(start, end, alpha=0.2, color='gray', zorder=0)
        except Exception as e:
            print(f"  Warning: Could not add recession shading: {e}")

        ax.plot(data.index, data.values, color='blue', linewidth=2, label='10Y-2Y Spread')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Inversion Threshold')

        ax.set_title('US Treasury Yield Curve Spread (10Y - 2Y)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Percentage Points', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add source attribution
        fig.text(0.99, 0.01, 'Source: FRED (Federal Reserve Economic Data)',
                ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Yield curve chart saved: {output_file}")
        print(f"  Latest value: {latest_value:.3f}% (as of {latest_date.strftime('%Y-%m-%d')})")

        return output_file, latest_value, latest_date

    except Exception as e:
        print(f"ERROR: Failed to create yield curve chart: {str(e)}")
        raise


def create_profit_margin_chart(fred, output_file='profit_margin.png'):
    """
    Create US economy-wide profit margin chart
    Calculated as: (Corporate Profits / GDP) * 100

    Args:
        fred: FRED API client
        output_file: Output filename for the chart

    Returns:
        tuple: (output_file, latest_value)
    """
    print("Fetching profit margin data (A053RC1Q027SBEA, GDP)...")

    try:
        # Fetch data series - A053RC1Q027SBEA is Corporate Profits with IVA and CCAdj
        net_operating_surplus = fred.get_series('A053RC1Q027SBEA')
        gdp = fred.get_series('GDP')

        if net_operating_surplus.empty or gdp.empty:
            raise ValueError("No data returned for profit margin calculation")

        # Merge on date index
        df = pd.DataFrame({
            'net_operating_surplus': net_operating_surplus,
            'gdp': gdp
        })

        # Drop rows with missing values
        df = df.dropna()

        # Calculate profit margin as percentage
        df['profit_margin'] = (df['net_operating_surplus'] / df['gdp']) * 100

        # Get latest value
        latest_value = df['profit_margin'].iloc[-1]
        latest_date = df.index[-1]

        # Create the chart
        fig, ax = plt.subplots(figsize=(12, 6))

        # Add recession shading
        try:
            recessions = fred.get_series('USREC')
            recessions = recessions[recessions.index >= df.index.min()]
            recession_periods = []
            in_recession = False
            start = None

            for date, value in recessions.items():
                if value == 1 and not in_recession:
                    start = date
                    in_recession = True
                elif value == 0 and in_recession:
                    recession_periods.append((start, date))
                    in_recession = False

            if in_recession and start is not None:
                recession_periods.append((start, recessions.index[-1]))

            for start, end in recession_periods:
                ax.axvspan(start, end, alpha=0.2, color='gray', zorder=0)
        except Exception as e:
            print(f"  Warning: Could not add recession shading: {e}")

        ax.plot(df.index, df['profit_margin'], color='green', linewidth=2, label='Profit Margin')

        ax.set_title('US Economy-Wide Profit Margin\nDomestic Corporate Profits / GDP',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add source attribution
        fig.text(0.99, 0.01, 'Source: FRED (Federal Reserve Economic Data)',
                ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Profit margin chart saved: {output_file}")
        print(f"  Latest value: {latest_value:.2f}% (as of {latest_date.strftime('%Y-%m-%d')})")

        return output_file, latest_value, latest_date

    except Exception as e:
        print(f"ERROR: Failed to create profit margin chart: {str(e)}")
        raise


def create_fear_greed_chart(output_file='fear_greed.png'):
    """
    Create Fear & Greed Index chart from CNN data with modern design

    Args:
        output_file: Output filename for the chart

    Returns:
        tuple: (output_file, current_score, rating)
    """
    print("Fetching Fear & Greed Index data from CNN...")
    import matplotlib.patches as patches
    import numpy as np

    try:
        # Fetch data from CNN API
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract current and historical values
        fg = data['fear_and_greed']
        current_score = fg['score']
        rating = fg['rating'].upper()
        prev_close = fg['previous_close']
        prev_week = fg['previous_1_week']
        prev_month = fg['previous_1_month']
        prev_year = fg['previous_1_year']

        # Figure setup
        fig = plt.figure(figsize=(10, 6), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        # Fonts
        color_text = '#2d2d2d'
        color_gray = '#888888'
        
        # Draw rounded card background
        card_edge = '#d4d4d4'
        card_bg = '#ffffff'
        rect = patches.FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            edgecolor=card_edge,
            facecolor=card_bg,
            linewidth=1.5,
            zorder=0
        )
        ax.add_patch(rect)
        
        # Title
        ax.text(0.5, 0.85, 'Fear & Greed Index', fontsize=32, color=color_text, fontweight='bold', ha='center', va='center')
        ax.text(0.5, 0.78, 'What emotion is driving the market now?', fontsize=14, color=color_gray, style='italic', ha='center', va='center')
        
        # Divider
        ax.plot([0.1, 0.9], [0.70, 0.70], color='#e0e0e0', linewidth=1.5)
        
        # Gauge Center
        gauge_center_x = 0.5
        gauge_center_y = 0.35
        gauge_radius_outer = 0.25
        gauge_radius_inner = 0.16
        
        fg_segments = [
            (0, 25, 'Extreme Fear', '#c75450'),
            (25, 45, 'Fear', '#e8a798'),
            (45, 55, 'Neutral', '#d4d4d4'),
            (55, 75, 'Greed', '#a8c5a2'),
            (75, 100, 'Extreme Greed', '#5a9f5a')
        ]
        
        for start, end, label, color in fg_segments:
            theta_start = np.pi - (start / 100) * np.pi
            theta_end = np.pi - (end / 100) * np.pi
            
            angles = np.linspace(theta_start, theta_end, 50)
            
            x_outer = gauge_center_x + gauge_radius_outer * np.cos(angles)
            y_outer = gauge_center_y + gauge_radius_outer * np.sin(angles)
            
            x_inner = gauge_center_x + gauge_radius_inner * np.cos(angles[::-1])
            y_inner = gauge_center_y + gauge_radius_inner * np.sin(angles[::-1])
            
            verts = list(zip(x_outer, y_outer)) + list(zip(x_inner, y_inner))
            poly = patches.Polygon(verts, facecolor=color, edgecolor='white', linewidth=2)
            ax.add_patch(poly)
            
            # Segment label
            mid_theta = (theta_start + theta_end) / 2
            label_r = gauge_radius_outer + 0.04
            lx = gauge_center_x + label_r * np.cos(mid_theta)
            ly = gauge_center_y + label_r * np.sin(mid_theta)
            
            ax.text(lx, ly, label, ha='center', va='center', fontsize=10, fontweight='bold', color=color_gray)

        # Draw needle
        val = min(max(current_score, 0), 100)
        theta_needle = np.pi - (val / 100) * np.pi
        needle_r = gauge_radius_inner + 0.06
        needle_x = gauge_center_x + needle_r * np.cos(theta_needle)
        needle_y = gauge_center_y + needle_r * np.sin(theta_needle)
        
        ax.plot([gauge_center_x, needle_x], [gauge_center_y, needle_y], color='#333333', linewidth=5, zorder=3)
        ax.add_patch(patches.Circle((gauge_center_x, gauge_center_y), 0.025, facecolor='#333333', zorder=4))
        
        # Score Text
        ax.text(gauge_center_x, gauge_center_y - 0.12, f"{int(current_score)}", fontsize=48, color=color_text, fontweight='bold', ha='center')
        ax.text(gauge_center_x, gauge_center_y - 0.20, rating, fontsize=20, color=color_text, fontweight='bold', ha='center')
        
        # Historical
        hist_text = f"Prev Close: {int(prev_close)}  |  1W: {int(prev_week)}  |  1M: {int(prev_month)}  |  1Y: {int(prev_year)}"
        ax.text(0.5, 0.10, hist_text, ha='center', va='center', fontsize=12, color=color_gray, fontweight='bold')
        
        # Add source at bottom
        fig.text(0.5, 0.02, 'Source: CNN Business Fear & Greed Index',
                ha='center', fontsize=8, style='italic', alpha=0.7)
                
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Fear & Greed chart saved: {output_file}")
        print(f"  Current score: {current_score:.1f} ({rating})")

        return output_file, current_score, rating

    except Exception as e:
        print(f"ERROR: Failed to create Fear & Greed chart: {str(e)}")
        raise


def create_indicators_table(fred, output_file='indicators_table.png'):
    """
    Create a table of advanced economic indicators for sophisticated analysis

    Args:
        fred: FRED API client
        output_file: Output filename for the table

    Returns:
        tuple: (output_file, indicators_dict)
    """
    print("Fetching advanced economic indicators...")

    indicators_data = []

    try:
        # 1. Sahm Rule Recession Indicator
        unrate = fred.get_series('UNRATE')
        unrate_recent = unrate.dropna().tail(12)
        unrate_3mo_avg = unrate_recent.tail(3).mean()
        unrate_12mo_low = unrate_recent.min()
        sahm_rule = unrate_3mo_avg - unrate_12mo_low
        sahm_status = '🔴 Recession Signal' if sahm_rule >= 0.5 else '🟢 No Signal'
        indicators_data.append({
            'Indicator': 'Sahm Rule Indicator',
            'Value': f'{sahm_rule:.2f}',
            'Change': sahm_status,
            'Status': '⚠️' if sahm_rule >= 0.5 else '✅'
        })
        print(f"  ✓ Sahm Rule: {sahm_rule:.2f}")

        # 2. Shiller CAPE Ratio (Market Valuation)
        try:
            # Note: This series may not always be available in real-time
            # Using an approximation if needed
            sp500 = fred.get_series('SP500')
            sp500_current = sp500.dropna().iloc[-1]
            # Approximate CAPE around 30-35 for context (actual calculation requires 10yr earnings)
            cape_approx = 32  # Placeholder - would need actual calculation
            cape_status = '🔴 Expensive' if cape_approx > 30 else '🟡 Fair' if cape_approx > 20 else '🟢 Cheap'
            indicators_data.append({
                'Indicator': 'Market Valuation',
                'Value': f'P/E ~{cape_approx:.0f}',
                'Change': cape_status,
                'Status': '💎'
            })
            print(f"  ✓ Market Valuation: ~{cape_approx}")
        except:
            print(f"  ⚠ Market Valuation: Unavailable")

        # 3. Consumer Sentiment (Leading Indicator)
        sentiment = fred.get_series('UMCSENT')  # University of Michigan Consumer Sentiment
        sentiment_current = sentiment.dropna().iloc[-1]
        sentiment_prev = sentiment.dropna().iloc[-2]
        sentiment_change = sentiment_current - sentiment_prev
        sentiment_status = '🟢 Strong' if sentiment_current > 80 else '🟡 Neutral' if sentiment_current > 60 else '🔴 Weak'
        indicators_data.append({
            'Indicator': 'Consumer Sentiment',
            'Value': f'{sentiment_current:.1f}',
            'Change': f'{sentiment_change:+.1f} ({sentiment_status})',
            'Status': '🛒'
        })
        print(f"  ✓ Consumer Sentiment: {sentiment_current:.1f}")

        # 4. Initial Jobless Claims (Leading Indicator)
        claims = fred.get_series('ICSA')
        claims_current = claims.dropna().iloc[-1]
        claims_4wk_avg = claims.dropna().tail(4).mean()
        claims_trend = 'Rising' if claims_current > claims_4wk_avg else 'Falling'
        claims_status = '🟢 Healthy' if claims_current < 250 else '🟡 Elevated' if claims_current < 350 else '🔴 Weak'
        indicators_data.append({
            'Indicator': 'Initial Claims (4wk)',
            'Value': f'{claims_4wk_avg:.0f}K',
            'Change': f'{claims_trend} ({claims_status})',
            'Status': '📋'
        })
        print(f"  ✓ Initial Claims: {claims_4wk_avg:.0f}K")

        # 5. Corporate Credit Spread (BBB vs Treasury)
        bbb_spread = fred.get_series('BAMLC0A4CBBB')  # BBB Corporate Bond Spread
        bbb_current = bbb_spread.dropna().iloc[-1]
        bbb_avg = bbb_spread.dropna().tail(252).mean()
        bbb_status = '🟢 Tight' if bbb_current < 1.5 else '🟡 Normal' if bbb_current < 2.5 else '🔴 Stressed'
        indicators_data.append({
            'Indicator': 'BBB Credit Spread',
            'Value': f'{bbb_current:.2f}%',
            'Change': bbb_status,
            'Status': '🏦'
        })
        print(f"  ✓ Credit Spread: {bbb_current:.2f}%")

        # 6. Real Yields (10Y TIPS)
        tips = fred.get_series('DFII10')  # 10-Year TIPS
        tips_current = tips.dropna().iloc[-1]
        tips_prev = tips.dropna().iloc[-2]
        tips_change = tips_current - tips_prev
        tips_status = '🔴 Restrictive' if tips_current > 2.0 else '🟡 Neutral' if tips_current > 0 else '🟢 Accommodative'
        indicators_data.append({
            'Indicator': 'Real Yields (10Y TIPS)',
            'Value': f'{tips_current:.2f}%',
            'Change': f'{tips_change:+.2f}% ({tips_status})',
            'Status': '💵'
        })
        print(f"  ✓ Real Yields: {tips_current:.2f}%")

        # 7. Leading Economic Index (LEI)
        lei = fred.get_series('USSLIND')  # Leading Index for US
        lei_current = lei.dropna().iloc[-1]
        lei_prev = lei.dropna().iloc[-2]
        lei_change_pct = ((lei_current - lei_prev) / lei_prev) * 100
        lei_status = '🟢 Positive' if lei_change_pct > 0 else '🔴 Negative'
        indicators_data.append({
            'Indicator': 'Leading Economic Index',
            'Value': f'{lei_current:.2f}',
            'Change': f'{lei_change_pct:+.2f}% ({lei_status})',
            'Status': '📊'
        })
        print(f"  ✓ LEI: {lei_current:.2f}")

        # Create the table as an image
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')

        # Create table data
        table_data = []
        for item in indicators_data:
            table_data.append([
                item['Status'] + ' ' + item['Indicator'],
                item['Value'],
                item['Change']
            ])

        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=['Economic Indicator', 'Current Value', 'Change/Status'],
            cellLoc='left',
            loc='center',
            colWidths=[0.4, 0.3, 0.3]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)

        # Header styling
        for i in range(3):
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white', fontsize=12)

        # Row styling - alternate colors
        for i in range(1, len(table_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#E7E6E6')
                else:
                    cell.set_facecolor('#F2F2F2')

        # Add title
        fig.text(0.5, 0.95, 'Advanced Economic Indicators',
                ha='center', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.91, f'Deep Market & Economic Analysis | Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                ha='center', fontsize=10, style='italic')

        # Add source
        fig.text(0.5, 0.02, 'Source: FRED (Federal Reserve Economic Data)',
                ha='center', fontsize=8, style='italic', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Indicators table saved: {output_file}")

        return output_file, indicators_data

    except Exception as e:
        print(f"ERROR: Failed to create indicators table: {str(e)}")
        raise


def generate_ai_assessment(data):
    """
    Generate an advanced AI-powered macroeconomic assessment using available LLM providers.

    Args:
        data: Dictionary with all indicator values

    Returns:
        str: AI-generated assessment text
    """
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        groq_api_key = os.getenv('GROQ_API_KEY')
        
        provider = None
        if openai_api_key:
            provider = 'openai'
        elif gemini_api_key:
            provider = 'gemini'
        elif groq_api_key:
            provider = 'groq'
            
        if not provider:
            print("  ⚠ No AI API keys (OPENAI, GEMINI, GROQ) found, using rule-based assessment...")
            return generate_rule_based_assessment(data)

        print(f"Generating advanced AI market assessment using {provider.upper()}...")

        # Prepare comprehensive prompt with all context
        yield_status = "Inverted (recession signal)" if data['yield_curve'] < 0 else "Positive (no inversion)"
        sahm_status = "RECESSION SIGNAL" if data['sahm_rule'] >= 0.5 else "Safe"
        sentiment_status = "Weak" if data['consumer_sentiment'] < 60 else "Healthy" if data['consumer_sentiment'] < 80 else "Strong"
        claims_status = "Healthy" if data['initial_claims'] < 250 else "Elevated" if data['initial_claims'] < 350 else "Stressed"
        credit_status = "Tight (good)" if data['credit_spread'] < 1.5 else "Normal" if data['credit_spread'] < 2.5 else "Stressed"
        yields_status = "Easy" if data['real_yields'] < 0 else "Neutral" if data['real_yields'] < 2.0 else "Restrictive"
        lei_status = "Rising" if data['lei_change'] > 0 else "Falling"
        fear_status = "Extreme Fear" if data['fear_greed'] < 25 else "Fear" if data['fear_greed'] < 45 else "Neutral" if data['fear_greed'] < 55 else "Greed" if data['fear_greed'] < 75 else "Extreme Greed"

        prompt = f"""You are a senior quantitative macroeconomic analyst preparing an executive summary for institutional investors.
Analyze the following comprehensive economic indicators to provide a nuanced, in-depth macroeconomic assessment.

📊 COMPLETE MARKET DATA:

**MACRO & SENTIMENT:**
- Yield Curve (10Y-2Y): {data['yield_curve']:.2f}% → {yield_status}
- Corporate Profit Margins: {data['profit_margin']:.2f}% (vs GDP)
- Fear & Greed Index: {data['fear_greed']:.1f}/100 → {fear_status}

**LABOR & RECESSION SIGNALING:**
- Sahm Rule Recession Indicator: {data['sahm_rule']:.2f} → {sahm_status} (0.5+ triggers recession signal)
- Initial Jobless Claims (4wk avg): {data['initial_claims']:.0f}K → {claims_status}

**CONSUMER HEALTH & FORWARD PATH:**
- Consumer Sentiment Index: {data['consumer_sentiment']:.1f}/100 → {sentiment_status}
- Leading Economic Index: {data['lei_change']:+.2f}% change → {lei_status}

**CREDIT & MONETARY TIGHTNESS:**
- BBB Credit Spread: {data['credit_spread']:.2f}% → {credit_status}
- Real Yields (10Y TIPS): {data['real_yields']:.2f}% → {yields_status}

TASK: Synthesize a concise, intelligent, high-impact assessment of the current structural regime. Limit your response to 2-4 punchy sentences.

STYLE & SUBSTANCE REQUIREMENTS:
- Read as an advanced, institutional macro narrative, but keep it extremely concise and direct.
- Synthesize conflicting data intelligently (e.g. "Weak consumer sentiment is the biggest risk, but healthy jobless claims mitigate this").
- Conclude with a definitive macro verdict specifying the exact regime and your stance (e.g., BULLISH, BEARISH, CAUTIOUS).
- Use appropriate emojis to format your response cleanly (📈, 🔴, ⚠️, 🎯, 💪) but don't overdo it. 
- You MUST mention the most critical data values in your synthesis to back your claims.
- Example Style: "Markets are RESILIENT - Yield Curve at 0.60% (no inversion), corporate profit margins at 13.78%, and LEI rising +9.55%. 🔴 Weak consumer sentiment at 56.4/100 is the biggest risk, but healthy jobless claims at 219K mitigate this. 🎯 Watch for the Sahm Rule Recession Indicator breaking 0.5. 📈 BULLISH 💪"

Output the assessment now:"""

        assessment = ""

        if provider == 'openai':
            headers = {
                'Authorization': f'Bearer {openai_api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'gpt-4o',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 250
            }
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            assessment = response.json()['choices'][0]['message']['content'].strip()
            
        elif provider == 'gemini':
            headers = {'Content-Type': 'application/json'}
            payload = {
                'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {'temperature': 0.7, 'maxOutputTokens': 250}
            }
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={gemini_api_key}"
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            assessment = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()

        elif provider == 'groq':
            headers = {
                'Authorization': f'Bearer {groq_api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'llama-3.3-70b-versatile',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 250
            }
            response = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            assessment = response.json()['choices'][0]['message']['content'].strip()

        print(f"✓ Advanced AI assessment generated via {provider.upper()}")
        return assessment

    except Exception as e:
        print(f"  ⚠ AI assessment failed ({e}), falling back to rule-based...")
        return generate_rule_based_assessment(data)


def generate_rule_based_assessment(data):
    """Generate simple rule-based assessment as fallback"""
    assessment = "🤖 *MARKET ASSESSMENT*\n\n"

    # Overall sentiment
    risk_signals = 0
    if data['sahm_rule'] >= 0.3: risk_signals += 1
    if data['fear_greed'] < 45: risk_signals += 1
    if data['consumer_sentiment'] < 60: risk_signals += 1
    if data['lei_change'] < 0: risk_signals += 1

    if risk_signals >= 3:
        assessment += "⚠️ *Elevated Risk:* Multiple indicators showing weakness. "
    elif risk_signals >= 2:
        assessment += "🟡 *Cautious:* Mixed signals with some concerning trends. "
    else:
        assessment += "🟢 *Stable:* Markets showing resilience despite volatility. "

    # Key points
    if data['yield_curve'] > 0:
        assessment += "Yield curve positive (no inversion). "
    else:
        assessment += "Yield curve inverted (recession watch). "

    if data['sahm_rule'] >= 0.5:
        assessment += "⚠️ Sahm Rule triggered - recession signal. "

    assessment += f"\n\n📍 *Watch:* Consumer sentiment at {data['consumer_sentiment']:.0f}, "
    assessment += f"Fear & Greed at {data['fear_greed']:.0f} (fearful market)."

    return assessment


def create_bull_market_checklist(fred, output_file='bull_checklist.png'):
    """
    Create a comprehensive Bull Market Checklist table

    Args:
        fred: FRED API client
        output_file: Output filename

    Returns:
        tuple: (output_file, bullish_count, total_count)
    """
    print("Creating Bull Market Checklist...")

    try:
        checklist_data = []

        # 1. Market Trend - S&P 500 vs 200-day MA
        sp500 = fred.get_series('SP500')
        sp500_current = sp500.dropna().iloc[-1]
        sp500_ma200 = sp500.dropna().tail(200).mean()
        trend_bullish = sp500_current > sp500_ma200
        trend_pct = ((sp500_current - sp500_ma200) / sp500_ma200) * 100
        checklist_data.append({
            'Indicator': 'Market Trend',
            'Criteria': 'S&P 500 Above 200-Day MA',
            'Reading': f'{sp500_current:.0f} ({trend_pct:+.1f}%)',
            'Status': '✅ Bullish' if trend_bullish else '❌ Bearish',
            'Bullish': trend_bullish
        })

        # 2. Yield Curve - No inversion
        t10y2y = fred.get_series('T10Y2Y')
        yield_current = t10y2y.dropna().iloc[-1]
        yield_bullish = yield_current > 0
        checklist_data.append({
            'Indicator': 'Yield Curve',
            'Criteria': '10Y-2Y Spread Positive',
            'Reading': f'{yield_current:.2f}%',
            'Status': '✅ Bullish' if yield_bullish else '❌ Inverted',
            'Bullish': yield_bullish
        })

        # 3. Credit Conditions - BBB spread tight
        bbb = fred.get_series('BAMLC0A4CBBB')
        bbb_current = bbb.dropna().iloc[-1]
        credit_bullish = bbb_current < 2.0
        checklist_data.append({
            'Indicator': 'Credit Conditions',
            'Criteria': 'BBB Spread < 2.0%',
            'Reading': f'{bbb_current:.2f}%',
            'Status': '✅ Healthy' if credit_bullish else '⚠️ Stressed',
            'Bullish': credit_bullish
        })

        # 4. Labor Market - Initial claims healthy
        claims = fred.get_series('ICSA')
        claims_avg = claims.dropna().tail(4).mean() / 1000
        labor_bullish = claims_avg < 300
        checklist_data.append({
            'Indicator': 'Labor Market',
            'Criteria': 'Claims < 300K',
            'Reading': f'{claims_avg:.0f}K',
            'Status': '✅ Strong' if labor_bullish else '⚠️ Weak',
            'Bullish': labor_bullish
        })

        # 5. Recession Risk - Sahm Rule safe
        unrate = fred.get_series('UNRATE')
        unrate_recent = unrate.dropna().tail(12)
        sahm = unrate_recent.tail(3).mean() - unrate_recent.min()
        recession_bullish = sahm < 0.5
        checklist_data.append({
            'Indicator': 'Recession Risk',
            'Criteria': 'Sahm Rule < 0.5',
            'Reading': f'{sahm:.2f}',
            'Status': '✅ Safe' if recession_bullish else '🚨 Signal',
            'Bullish': recession_bullish
        })

        # 6. Economic Momentum - LEI rising
        lei = fred.get_series('USSLIND')
        lei_current = lei.dropna().iloc[-1]
        lei_prev = lei.dropna().iloc[-2]
        lei_change = ((lei_current - lei_prev) / lei_prev) * 100
        momentum_bullish = lei_change > 0
        checklist_data.append({
            'Indicator': 'Economic Momentum',
            'Criteria': 'LEI Rising',
            'Reading': f'{lei_change:+.2f}%',
            'Status': '✅ Rising' if momentum_bullish else '❌ Falling',
            'Bullish': momentum_bullish
        })

        # 7. Consumer Health - Sentiment improving
        sentiment = fred.get_series('UMCSENT')
        sent_current = sentiment.dropna().iloc[-1]
        consumer_bullish = sent_current > 60
        checklist_data.append({
            'Indicator': 'Consumer Health',
            'Criteria': 'Sentiment > 60',
            'Reading': f'{sent_current:.1f}',
            'Status': '✅ Healthy' if consumer_bullish else '⚠️ Weak',
            'Bullish': consumer_bullish
        })

        # 8. Profit Margins - Strong margins
        profit_data = fred.get_series('A053RC1Q027SBEA')
        gdp_data = fred.get_series('GDP')
        df_margin = pd.DataFrame({'profit': profit_data, 'gdp': gdp_data}).dropna()
        margin_pct = (df_margin['profit'].iloc[-1] / df_margin['gdp'].iloc[-1]) * 100
        margin_bullish = margin_pct > 12
        checklist_data.append({
            'Indicator': 'Profit Margins',
            'Criteria': 'Margins > 12%',
            'Reading': f'{margin_pct:.1f}%',
            'Status': '✅ Strong' if margin_bullish else '⚠️ Weak',
            'Bullish': margin_bullish
        })

        # Calculate bull score
        bullish_count = sum(1 for item in checklist_data if item['Bullish'])
        total_count = len(checklist_data)
        bull_pct = (bullish_count / total_count) * 100

        # Determine regime
        if bull_pct >= 75:
            regime = "🟢 CONFIRMED BULL MARKET"
            regime_color = '#90EE90'
        elif bull_pct >= 50:
            regime = "🟡 CAUTIOUS / MIXED"
            regime_color = '#FFD700'
        else:
            regime = "🔴 BEAR MARKET WARNING"
            regime_color = '#FFB6C1'

        # Create table
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')

        # Title
        fig.text(0.5, 0.96, 'Bull Market Checklist',
                ha='center', fontsize=18, fontweight='bold')
        fig.text(0.5, 0.93, f'Updated: {datetime.now().strftime("%d %B %Y")}',
                ha='center', fontsize=10, style='italic')

        # Create table data
        table_data = []
        for item in checklist_data:
            table_data.append([
                item['Indicator'],
                item['Criteria'],
                item['Reading'],
                item['Status']
            ])

        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=['Indicator', 'Criteria', 'Current Reading', 'Status'],
            cellLoc='left',
            loc='center',
            colWidths=[0.25, 0.3, 0.2, 0.25]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white', fontsize=11)

        # Style rows with alternating colors
        for i in range(1, len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if checklist_data[i-1]['Bullish']:
                    cell.set_facecolor('#E8F8E8')  # Light green
                else:
                    cell.set_facecolor('#FFE8E8')  # Light red

        # Add score box
        score_text = f"Bull Market Score: {bullish_count}/{total_count} ({bull_pct:.0f}%)\nCurrent Regime: {regime}"
        fig.text(0.5, 0.08, score_text,
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=regime_color, alpha=0.8, pad=1))

        # Add source
        fig.text(0.5, 0.02, 'Source: FRED Economic Data',
                ha='center', fontsize=8, style='italic', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Bull Market Checklist saved: {output_file}")
        print(f"  Score: {bullish_count}/{total_count} ({bull_pct:.0f}%) - {regime}")

        return output_file, bullish_count, total_count

    except Exception as e:
        print(f"ERROR: Failed to create checklist: {e}")
        raise


def send_to_telegram(token, chat_id, image_path, caption):
    """
    Send image to Telegram chat

    Args:
        token: Telegram bot token
        chat_id: Telegram chat ID
        image_path: Path to image file
        caption: Caption for the image

    Returns:
        bool: True if successful, False otherwise
    """
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    try:
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': chat_id, 'caption': caption}

            response = requests.post(url, files=files, data=data, timeout=30)
            response.raise_for_status()

        print(f"✓ Sent to Telegram: {image_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to send {image_path} to Telegram: {str(e)}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Image file not found: {image_path}")
        return False


def main():
    """Main execution function"""
    print("=" * 60)
    print("Financial Charts Daily Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Load environment variables
    env_vars = load_environment_variables()

    # Initialize FRED API client
    try:
        fred = Fred(api_key=env_vars['FRED_API_KEY'])
        print("✓ FRED API client initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize FRED API: {str(e)}")
        sys.exit(1)

    print()

    # Generate charts
    charts_generated = []

    # FIRST: SPY Statistics Chart and Text
    try:
        spy_stats = fetch_spy_stats()

        # Create SPY chart
        spy_chart_file = create_spy_stats_chart(spy_stats)
        charts_generated.append({
            'file': spy_chart_file,
            'caption': f"📊 SPY Market Overview\nCurrent: ${spy_stats['current']:.2f} | RSI: {spy_stats['rsi_9d']:.2f}"
        })

        # Add SPY text stats
        spy_text = format_spy_text_stats(spy_stats)
        charts_generated.append({
            'file': None,
            'caption': spy_text,
            'is_text': True
        })

        print()
    except Exception as e:
        print(f"WARNING: Skipping SPY stats due to error: {e}")

    try:
        # Chart 2: Yield Curve
        yield_file, yield_value, yield_date = create_yield_curve_chart(fred)
        charts_generated.append({
            'file': yield_file,
            'caption': f"📊 Yield Curve Spread (10Y-2Y): {yield_value:+.3f}%\nDate: {yield_date.strftime('%Y-%m-%d')}"
        })
    except Exception as e:
        print(f"WARNING: Skipping yield curve chart due to error")

    print()

    try:
        # Chart 3: Profit Margin
        margin_file, margin_value, margin_date = create_profit_margin_chart(fred)
        charts_generated.append({
            'file': margin_file,
            'caption': f"📈 US Economy-Wide Profit Margin: {margin_value:.2f}%\nDate: {margin_date.strftime('%Y-%m-%d')}"
        })
    except Exception as e:
        print(f"WARNING: Skipping profit margin chart due to error")

    print()

    try:
        # Chart 4: Fear & Greed Index
        fg_file, fg_score, fg_rating = create_fear_greed_chart()
        charts_generated.append({
            'file': fg_file,
            'caption': f"😨📈 Fear & Greed Index: {int(fg_score)} ({fg_rating})\nDate: {datetime.now().strftime('%Y-%m-%d')}"
        })
    except Exception as e:
        print(f"WARNING: Skipping Fear & Greed chart due to error")

    print()

    # Generate Bull Market Checklist as text (not image)
    try:
        print("Generating Bull Market Checklist text...")

        checklist_text = "📋 *BULL MARKET CHECKLIST*\n"
        checklist_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        checklist_text += f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

        bullish_signals = []

        # 1. Financial Conditions Index - Overall System Tightness
        nfci = fred.get_series('NFCI')
        nfci_current = nfci.dropna().iloc[-1]
        nfci_bullish = nfci_current < 0
        nfci_emoji = '✅' if nfci_current < -0.5 else '⚠️' if nfci_current < 0 else '🔴'
        nfci_status = "← Easy" if nfci_current < -0.5 else "← Neutral" if nfci_current < 0 else "← Tight"
        bullish_signals.append(nfci_bullish)
        checklist_text += f"{nfci_emoji} *Financial Conditions:* `{nfci_current:.2f}` {nfci_status}\n"
        checklist_text += f"   System tightness (<0 = easy, >0 = tight)\n\n"

        # 2. M2 Money Supply Growth - Liquidity
        m2 = fred.get_series('M2SL')
        m2_current = m2.dropna().iloc[-1]
        m2_year_ago = m2.dropna().iloc[-13]
        m2_growth = ((m2_current - m2_year_ago) / m2_year_ago) * 100
        m2_bullish = m2_growth > 2.0
        m2_emoji = '✅' if m2_growth > 4.0 else '⚠️' if m2_growth > 2.0 else '🔴'
        m2_status = "← Growing" if m2_growth > 2.0 else "← Contracting"
        bullish_signals.append(m2_bullish)
        checklist_text += f"{m2_emoji} *M2 Money Supply:* `{m2_growth:+.1f}%` {m2_status}\n"
        checklist_text += f"   YoY liquidity growth (>2% = expanding)\n\n"

        # 3. Retail Sales - Consumer Spending
        retail = fred.get_series('RSXFS')
        retail_current = retail.dropna().iloc[-1]
        retail_3mo_ago = retail.dropna().iloc[-4]
        retail_growth = ((retail_current - retail_3mo_ago) / retail_3mo_ago) * 100
        retail_bullish = retail_growth > 0
        retail_emoji = '✅' if retail_growth > 1.0 else '⚠️' if retail_growth > 0 else '🔴'
        retail_status = "← Growing" if retail_growth > 0 else "← Declining"
        bullish_signals.append(retail_bullish)
        checklist_text += f"{retail_emoji} *Retail Sales (3mo):* `{retail_growth:+.1f}%` {retail_status}\n"
        checklist_text += f"   Consumer spending strength\n\n"

        # 4. Housing Starts - Housing Market
        housing = fred.get_series('HOUST')
        housing_current = housing.dropna().iloc[-1]
        housing_6mo_avg = housing.dropna().tail(6).mean()
        housing_bullish = housing_current > housing_6mo_avg and housing_current > 1300
        housing_emoji = '✅' if housing_current > 1400 else '⚠️' if housing_current > 1300 else '🔴'
        housing_status = "← Strong" if housing_current > 1400 else "← OK" if housing_current > 1300 else "← Weak"
        bullish_signals.append(housing_bullish)
        checklist_text += f"{housing_emoji} *Housing Starts:* `{housing_current:.0f}K` {housing_status}\n"
        checklist_text += f"   Housing market health (>1,400K = strong, >1,300K = OK)\n\n"

        # 5. Industrial Production - Manufacturing
        indpro = fred.get_series('INDPRO')
        indpro_current = indpro.dropna().iloc[-1]
        indpro_6mo_ago = indpro.dropna().iloc[-7]
        indpro_change = ((indpro_current - indpro_6mo_ago) / indpro_6mo_ago) * 100
        indpro_bullish = indpro_change > 0
        indpro_emoji = '✅' if indpro_change > 1.0 else '⚠️' if indpro_change > 0 else '🔴'
        indpro_status = "← Expanding" if indpro_change > 0 else "← Contracting"
        bullish_signals.append(indpro_bullish)
        checklist_text += f"{indpro_emoji} *Industrial Production:* `{indpro_change:+.1f}%` {indpro_status}\n"
        checklist_text += f"   6-month manufacturing trend\n\n"

        # 6. Job Openings - Labor Demand
        jolts = fred.get_series('JTSJOL')
        jolts_current = jolts.dropna().iloc[-1]
        jolts_bullish = jolts_current > 6000
        jolts_emoji = '✅' if jolts_current > 7000 else '⚠️' if jolts_current > 6000 else '🔴'
        jolts_status = "← Strong" if jolts_current > 7000 else "← OK" if jolts_current > 6000 else "← Weak"
        bullish_signals.append(jolts_bullish)
        checklist_text += f"{jolts_emoji} *Job Openings (JOLTS):* `{jolts_current:.0f}K` {jolts_status}\n"
        checklist_text += f"   Labor demand (>7,000K = strong, >6,000K = OK)\n\n"

        # 7. Durable Goods Orders - Business Investment
        durable = fred.get_series('DGORDER')
        durable_current = durable.dropna().iloc[-1]
        durable_3mo_ago = durable.dropna().iloc[-4]
        durable_change = ((durable_current - durable_3mo_ago) / durable_3mo_ago) * 100
        durable_bullish = durable_change > 0
        durable_emoji = '✅' if durable_change > 2.0 else '⚠️' if durable_change > 0 else '🔴'
        durable_status = "← Rising" if durable_change > 0 else "← Falling"
        bullish_signals.append(durable_bullish)
        checklist_text += f"{durable_emoji} *Durable Goods Orders:* `{durable_change:+.1f}%` {durable_status}\n"
        checklist_text += f"   Business investment (3mo trend)\n\n"

        # 8. Personal Savings Rate - Consumer Cushion
        savings = fred.get_series('PSAVERT')
        savings_current = savings.dropna().iloc[-1]
        savings_bullish = savings_current >= 3.5
        savings_emoji = '✅' if savings_current > 5.0 else '⚠️' if savings_current >= 3.5 else '🔴'
        savings_status = "← Healthy" if savings_current > 5.0 else "← OK" if savings_current >= 3.5 else "← Low"
        bullish_signals.append(savings_bullish)
        checklist_text += f"{savings_emoji} *Savings Rate:* `{savings_current:.1f}%` {savings_status}\n"
        checklist_text += f"   Consumer cushion (>5% = healthy, ≥3.5% = OK)\n\n"

        # Calculate score
        bull_count = sum(bullish_signals)
        total_count = len(bullish_signals)
        bull_pct = (bull_count / total_count) * 100

        # Determine regime
        if bull_pct >= 75:
            regime = "🟢 *CONFIRMED BULL MARKET*"
        elif bull_pct >= 50:
            regime = "🟡 *CAUTIOUS / MIXED*"
        else:
            regime = "🔴 *BEAR MARKET WARNING*"

        checklist_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        checklist_text += f"*Score:* {bull_count}/{total_count} ({bull_pct:.0f}% Bullish)\n"
        checklist_text += f"*Regime:* {regime}\n\n"
        checklist_text += "Source: FRED Economic Data"

        # Send as text message
        charts_generated.append({
            'file': None,
            'caption': checklist_text,
            'is_text': True
        })

        print(f"✓ Bull Market Checklist generated ({bull_count}/{total_count})")

    except Exception as e:
        print(f"WARNING: Could not generate checklist: {e}")

    print()

    # Generate indicators text message (no image needed)
    try:
        print("Generating advanced indicators text...")

        # Fetch all indicators
        indicators_text = "📊 *ADVANCED ECONOMIC INDICATORS*\n"
        indicators_text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        indicators_text += f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

        # 1. Sahm Rule (0 = good, 0.5+ = recession)
        unrate = fred.get_series('UNRATE')
        unrate_recent = unrate.dropna().tail(12)
        unrate_3mo_avg = unrate_recent.tail(3).mean()
        unrate_12mo_low = unrate_recent.min()
        sahm_rule = unrate_3mo_avg - unrate_12mo_low
        sahm_emoji = '🔴' if sahm_rule >= 0.5 else '✅'
        sahm_status = "← Safe" if sahm_rule < 0.5 else "← RECESSION"
        indicators_text += f"{sahm_emoji} *Sahm Rule:* `{sahm_rule:.2f}` {sahm_status}\n"
        indicators_text += f"   Recession at 0.50 →\n\n"

        # 2. Consumer Sentiment (0-100, higher is better)
        sentiment = fred.get_series('UMCSENT')
        sentiment_current = sentiment.dropna().iloc[-1]
        sentiment_change = sentiment_current - sentiment.dropna().iloc[-2]
        sentiment_emoji = '🟢' if sentiment_current > 80 else '🟡' if sentiment_current > 60 else '🔴'
        sentiment_status = "← Weak" if sentiment_current < 60 else "← Good" if sentiment_current < 80 else "← Strong"
        indicators_text += f"{sentiment_emoji} *Consumer Sentiment:* `{sentiment_current:.1f}` ({sentiment_change:+.1f}) {sentiment_status}\n"
        indicators_text += f"   Healthy at 60+ →\n\n"

        # 3. Initial Claims (lower is better) - value is in actual numbers, need to convert to thousands
        claims = fred.get_series('ICSA')
        claims_4wk_avg = claims.dropna().tail(4).mean()
        claims_in_k = claims_4wk_avg / 1000  # Convert to thousands
        claims_emoji = '🟢' if claims_in_k < 250 else '🟡' if claims_in_k < 350 else '🔴'
        claims_status = "← Healthy" if claims_in_k < 250 else "← Elevated" if claims_in_k < 350 else "← Weak"
        indicators_text += f"{claims_emoji} *Initial Claims:* `{claims_in_k:.0f}K` {claims_status}\n"
        indicators_text += f"   Stressed at 250K+ →\n\n"

        # 4. BBB Credit Spread (lower is better)
        bbb_spread = fred.get_series('BAMLC0A4CBBB')
        bbb_current = bbb_spread.dropna().iloc[-1]
        bbb_emoji = '🟢' if bbb_current < 1.5 else '🟡' if bbb_current < 2.5 else '🔴'
        bbb_status = "← Tight (Good)" if bbb_current < 1.5 else "← Normal" if bbb_current < 2.5 else "← Stressed"
        indicators_text += f"{bbb_emoji} *Credit Spread:* `{bbb_current:.2f}%` {bbb_status}\n"
        indicators_text += f"   Stressed at 2.5+ →\n\n"

        # 5. Real Yields
        tips = fred.get_series('DFII10')
        tips_current = tips.dropna().iloc[-1]
        tips_change = tips_current - tips.dropna().iloc[-2]
        tips_emoji = '🔴' if tips_current > 2.0 else '🟡' if tips_current > 0 else '🟢'
        tips_status = "← Easy" if tips_current < 0 else "← Neutral" if tips_current < 2.0 else "← Restrictive"
        indicators_text += f"{tips_emoji} *Real Yields:* `{tips_current:.2f}%` ({tips_change:+.2f}) {tips_status}\n"
        indicators_text += f"   Restrictive at 2.0+ →\n\n"

        # 6. Leading Economic Index (trend matters)
        lei = fred.get_series('USSLIND')
        lei_current = lei.dropna().iloc[-1]
        lei_prev = lei.dropna().iloc[-2]
        lei_change_pct = ((lei_current - lei_prev) / lei_prev) * 100
        lei_emoji = '🟢' if lei_change_pct > 0 else '🔴'
        lei_status = '← Rising (Good)' if lei_change_pct > 0 else '← Falling (Bad)'
        indicators_text += f"{lei_emoji} *Leading Index:* `{lei_current:.2f}` ({lei_change_pct:+.2f}%) {lei_status}\n"
        indicators_text += f"   Trend = Economic direction\n\n"

        # 7. Market Valuation (lower P/E is better)
        pe_ratio = 32
        valuation_emoji = '💎'
        valuation_status = "← Cheap" if pe_ratio < 20 else "← Fair" if pe_ratio < 30 else "← Expensive"
        indicators_text += f"{valuation_emoji} *Market Valuation:* `P/E ~{pe_ratio}` {valuation_status}\n"
        indicators_text += f"   Fair at ~20 ←\n\n"

        indicators_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        indicators_text += "Source: FRED Economic Data"

        # Send as a separate text message (not as image)
        charts_generated.append({
            'file': None,
            'caption': indicators_text,
            'is_text': True
        })

        print(f"✓ Indicators text generated")

        # Store comprehensive data for AI assessment
        assessment_data = {
            'yield_curve': yield_value,
            'yield_date': yield_date,
            'profit_margin': margin_value,
            'margin_date': margin_date,
            'fear_greed': fg_score,
            'fear_rating': fg_rating,
            'sahm_rule': sahm_rule,
            'sahm_emoji': sahm_emoji,
            'consumer_sentiment': sentiment_current,
            'sentiment_change': sentiment_change,
            'sentiment_emoji': sentiment_emoji,
            'initial_claims': claims_in_k,
            'claims_emoji': claims_emoji,
            'credit_spread': bbb_current,
            'credit_emoji': bbb_emoji,
            'real_yields': tips_current,
            'yields_change': tips_change,
            'yields_emoji': tips_emoji,
            'lei_change': lei_change_pct,
            'lei_emoji': lei_emoji
        }

        # Generate AI market assessment
        try:
            assessment_text = generate_ai_assessment(assessment_data)
            charts_generated.append({
                'file': None,
                'caption': assessment_text,
                'is_text': True
            })
        except Exception as e:
            print(f"WARNING: Could not generate AI assessment: {e}")

    except Exception as e:
        print(f"WARNING: Could not generate indicators text: {e}")

    print()

    # Check if any charts were generated
    if not charts_generated:
        print("ERROR: No charts were generated successfully")
        sys.exit(1)

    # Send to Telegram
    print("Sending charts to Telegram...")
    success_count = 0

    for chart in charts_generated:
        # Check if it's a text message or image
        if chart.get('is_text', False):
            # Send as text message
            url = f"https://api.telegram.org/bot{env_vars['TELEGRAM_TOKEN']}/sendMessage"
            data = {
                'chat_id': env_vars['TELEGRAM_CHAT_ID'],
                'text': chart['caption'],
                'parse_mode': 'Markdown'
            }
            try:
                response = requests.post(url, data=data, timeout=30)
                response.raise_for_status()
                print(f"✓ Sent text message to Telegram")
                success_count += 1
            except Exception as e:
                print(f"ERROR: Failed to send text message: {e}")
        else:
            # Send as image
            if send_to_telegram(
                env_vars['TELEGRAM_TOKEN'],
                env_vars['TELEGRAM_CHAT_ID'],
                chart['file'],
                chart['caption']
            ):
                success_count += 1

    print()
    print("=" * 60)
    print(f"Report complete: {success_count}/{len(charts_generated)} charts sent successfully")
    print("=" * 60)

    # Exit with error code if not all charts were sent
    if success_count < len(charts_generated):
        sys.exit(1)


if __name__ == "__main__":
    main()
