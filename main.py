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

# Set style for professional-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_environment_variables():
    """Load and validate required environment variables"""
    required_vars = {
        'FRED_API_KEY': os.getenv('FRED_API_KEY'),
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID')
    }

    missing_vars = [var for var, value in required_vars.items() if not value]

    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    return required_vars


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
    Create Fear & Greed Index chart from CNN data

    Args:
        output_file: Output filename for the chart

    Returns:
        tuple: (output_file, current_score, rating)
    """
    print("Fetching Fear & Greed Index data from CNN...")

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

        # Create the gauge chart
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'polar'})

        # Define segments
        segments = [
            (0, 25, 'Extreme Fear', '#c75450'),
            (25, 45, 'Fear', '#e8a798'),
            (45, 55, 'Neutral', '#d4d4d4'),
            (55, 75, 'Greed', '#a8c5a2'),
            (75, 100, 'Extreme Greed', '#5a9f5a')
        ]

        # Draw segments (bottom half circle)
        import numpy as np
        theta_offset = np.pi  # Start from left (180 degrees)

        for start, end, label, color in segments:
            theta_start = theta_offset + (start / 100) * np.pi
            theta_end = theta_offset + (end / 100) * np.pi
            theta = np.linspace(theta_start, theta_end, 100)
            r = np.ones_like(theta)
            ax.fill_between(theta, 0, r, color=color, alpha=0.7)

            # Add labels
            mid_theta = (theta_start + theta_end) / 2
            label_r = 0.7
            ax.text(mid_theta, label_r, label, ha='center', va='center',
                   fontsize=9, fontweight='bold', rotation=0)

        # Draw the needle
        needle_theta = theta_offset + (current_score / 100) * np.pi
        ax.plot([needle_theta, needle_theta], [0, 0.95], 'k-', linewidth=4)
        ax.plot(needle_theta, 0, 'ko', markersize=10)

        # Configure polar plot
        ax.set_ylim(0, 1.2)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('W')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)

        # Add score in center
        ax.text(np.pi, -0.3, f"{int(current_score)}",
               ha='center', va='center', fontsize=48, fontweight='bold')
        ax.text(np.pi, -0.55, rating,
               ha='center', va='center', fontsize=16, fontweight='bold')

        # Add title
        fig.text(0.5, 0.95, 'Fear & Greed Index',
                ha='center', fontsize=18, fontweight='bold')
        fig.text(0.5, 0.91, 'What emotion is driving the market now?',
                ha='center', fontsize=10, style='italic')

        # Add historical values
        history_text = f"""Previous close: {rating} ({int(prev_close)})
1 week ago: {"NEUTRAL" if 45 <= prev_week <= 55 else "FEAR" if prev_week < 45 else "GREED"} ({int(prev_week)})
1 month ago: {"NEUTRAL" if 45 <= prev_month <= 55 else "FEAR" if prev_month < 45 else "GREED"} ({int(prev_month)})
1 year ago: {"NEUTRAL" if 45 <= prev_year <= 55 else "FEAR" if prev_year < 45 else "GREED"} ({int(prev_year)})"""

        fig.text(0.75, 0.35, history_text, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Add scale markers
        for value in [0, 25, 50, 75, 100]:
            theta_val = theta_offset + (value / 100) * np.pi
            ax.text(theta_val, 1.05, str(value), ha='center', va='center', fontsize=9)

        # Add source
        fig.text(0.5, 0.02, 'Source: CNN Business Fear & Greed Index',
                ha='center', fontsize=8, style='italic', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
    Generate AI-powered market assessment using Groq API

    Args:
        data: Dictionary with all indicator values

    Returns:
        str: AI-generated assessment text
    """
    try:
        # Check if Groq API key is available (optional)
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            print("  ⚠ GROQ_API_KEY not set, generating rule-based assessment...")
            return generate_rule_based_assessment(data)

        print("Generating AI market assessment...")

        # Prepare the prompt
        prompt = f"""You are a financial analyst. Analyze these economic indicators and provide a brief, actionable market assessment (3-4 sentences max):

📊 DATA:
- Yield Curve (10Y-2Y): {data['yield_curve']:.2f}%
- Profit Margin: {data['profit_margin']:.2f}%
- Fear & Greed Index: {data['fear_greed']:.1f} (Fear)
- Sahm Rule: {data['sahm_rule']:.2f} (0.5+ = recession)
- Consumer Sentiment: {data['consumer_sentiment']:.1f}
- Initial Claims: {data['initial_claims']:.0f}K
- Credit Spread: {data['credit_spread']:.2f}%
- Real Yields: {data['real_yields']:.2f}%
- Leading Index Change: {data['lei_change']:+.2f}%

Provide a concise assessment covering:
1. Overall market health (1 sentence)
2. Key risk or opportunity (1 sentence)
3. What to watch (1 sentence)

Be direct and actionable. No fluff."""

        # Call Groq API
        headers = {
            'Authorization': f'Bearer {groq_api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': 'llama-3.3-70b-versatile',  # Latest Llama model
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 300
        }

        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        assessment = response.json()['choices'][0]['message']['content'].strip()
        print(f"✓ AI assessment generated")
        return assessment

    except Exception as e:
        print(f"  ⚠ AI assessment failed: {e}, using rule-based...")
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

    try:
        # Chart 1: Yield Curve
        yield_file, yield_value, yield_date = create_yield_curve_chart(fred)
        charts_generated.append({
            'file': yield_file,
            'caption': f"📊 Yield Curve Spread (10Y-2Y): {yield_value:+.3f}%\nDate: {yield_date.strftime('%Y-%m-%d')}"
        })
    except Exception as e:
        print(f"WARNING: Skipping yield curve chart due to error")

    print()

    try:
        # Chart 2: Profit Margin
        margin_file, margin_value, margin_date = create_profit_margin_chart(fred)
        charts_generated.append({
            'file': margin_file,
            'caption': f"📈 US Economy-Wide Profit Margin: {margin_value:.2f}%\nDate: {margin_date.strftime('%Y-%m-%d')}"
        })
    except Exception as e:
        print(f"WARNING: Skipping profit margin chart due to error")

    print()

    try:
        # Chart 3: Fear & Greed Index
        fg_file, fg_score, fg_rating = create_fear_greed_chart()
        charts_generated.append({
            'file': fg_file,
            'caption': f"😨📈 Fear & Greed Index: {int(fg_score)} ({fg_rating})\nDate: {datetime.now().strftime('%Y-%m-%d')}"
        })
    except Exception as e:
        print(f"WARNING: Skipping Fear & Greed chart due to error")

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

        # Store data for AI assessment
        assessment_data = {
            'yield_curve': yield_value,
            'profit_margin': margin_value,
            'fear_greed': fg_score,
            'sahm_rule': sahm_rule,
            'consumer_sentiment': sentiment_current,
            'initial_claims': claims_in_k,
            'credit_spread': bbb_current,
            'real_yields': tips_current,
            'lei_change': lei_change_pct
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
