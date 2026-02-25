"""
Modular entrypoint for the financial-telegram-bot.
Orchestrates data fetching, chart generation, and Telegram reporting.
"""

import sys
import os
from datetime import datetime
from fredapi import Fred

# Import from the modular bot package
from bot.utils import load_environment_variables, send_to_telegram
from bot.fetchers import fetch_google_sheet_indicators, fetch_spy_stats
from bot.charts import (
    create_spy_stats_chart,
    create_yield_curve_chart,
    create_profit_margin_chart,
    create_fear_greed_chart
)
from bot.assessment import generate_ai_assessment

def main():
    """Main execution function"""
    print("=" * 60)
    print("Financial Charts Daily Report (Modular)")
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

    # Generate and send report
    try:
        # 1. Google Sheets
        gs_text = fetch_google_sheet_indicators()
        if gs_text:
            send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], caption=gs_text)

        # 2. SPY Stats
        spy_stats = fetch_spy_stats()
        spy_chart = create_spy_stats_chart(spy_stats)
        spy_caption = f"📊 SPY Overview | Current: ${spy_stats['current']:.2f} | RSI: {spy_stats['rsi_9d']:.2f}"
        send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], image_path=spy_chart, caption=spy_caption)

        # 3. Yield Curve
        yield_file, yield_val, _ = create_yield_curve_chart(fred)
        send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], image_path=yield_file, caption=f"📊 Yield Curve Spread (10Y-2Y): {yield_val:+.3f}%")

        # 4. Fear & Greed
        fg_file, fg_score, fg_rating = create_fear_greed_chart()
        send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], image_path=fg_file, caption=f"😨📈 Fear & Greed: {int(round(fg_score))} ({fg_rating})")

        # 5. Profit Margins
        margin_file, margin_val, _ = create_profit_margin_chart(fred)
        send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], image_path=margin_file, caption=f"📈 Profit Margin: {margin_val:.2f}%")

        # 6. AI Assessment
        # Construct data dict for assessment
        data = {
            'yield_curve': yield_val,
            'profit_margin': margin_val,
            'fear_greed': fg_score,
            'sahm_rule': 0.0, # Placeholder or fetch if needed
            'initial_claims': 200, # Placeholder
            'consumer_sentiment': 70.0, # Placeholder
            'lei_change': 0.5, # Placeholder
            'credit_spread': 1.2, # Placeholder
            'real_yields': 1.5 # Placeholder
        }
        assessment = generate_ai_assessment(data)
        send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], caption=assessment)

    except Exception as e:
        print(f"CRITICAL ERROR in main loop: {e}")

if __name__ == "__main__":
    main()
