"""
Modular entrypoint for the financial-telegram-bot.
Orchestrates data fetching, chart generation, and Telegram reporting.
Includes a Flask health-check server and APScheduler for cloud deployment.
"""

import sys
import os
import time
import pytz
import logging
from datetime import datetime
from threading import Thread
from typing import Dict, Any, Optional

from fredapi import Fred
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

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
from bot.config import TIMEZONE, REPORT_TIME

# Flask app for health checks
flask_app = Flask(__name__)
global_scheduler: Optional[BackgroundScheduler] = None

@flask_app.route('/')
def health_check():
    return {'status': 'running', 'bot': 'financial-telegram-bot'}, 200

@flask_app.route('/health')
def health():
    return {'status': 'healthy'}, 200

def run_report():
    """Execute the full report generation and delivery sequence"""
    print("\n" + "=" * 60)
    print("Generating Daily Financial Report...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    env_vars = load_environment_variables()
    
    try:
        # Initialize FRED
        fred = Fred(api_key=env_vars['FRED_API_KEY'])
        
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
        data = {
            'yield_curve': yield_val,
            'profit_margin': margin_val,
            'fear_greed': fg_score,
            'sahm_rule': 0.0, 'initial_claims': 200, 'consumer_sentiment': 70.0,
            'lei_change': 0.5, 'credit_spread': 1.2, 'real_yields': 1.5
        }
        assessment = generate_ai_assessment(data)
        send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], caption=assessment)
        
        print("\n✓ Report processing complete.")
        return True
    except Exception as e:
        print(f"CRITICAL ERROR in report generation: {e}")
        return False

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /report command"""
    env_vars = load_environment_variables()
    if str(update.effective_chat.id) != env_vars['TELEGRAM_CHAT_ID']:
        return

    await update.message.reply_text("🔄 Generating your financial report... this will take about 30 seconds.")
    
    # Run in a separate thread to avoid blocking the bot's event loop
    def job():
        run_report()
    
    Thread(target=job).start()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text("👋 Financial Report Bot\n\nUse /report to generate a report.")

def run_flask():
    """Run Flask server for health checks"""
    port = int(os.environ.get('PORT', 10000))
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

def main():
    """Start the integrated bot service"""
    global global_scheduler
    print("Starting Integrated Financial Bot Service...")

    env_vars = load_environment_variables()
    tz = pytz.timezone(TIMEZONE)

    # 1. Start Scheduler
    scheduler = BackgroundScheduler(timezone=tz)
    global_scheduler = scheduler
    scheduler.add_job(
        run_report,
        trigger=CronTrigger(hour=REPORT_TIME['hour'], minute=REPORT_TIME['minute'], timezone=tz),
        id='daily_report',
        name=f"Daily Report at {REPORT_TIME['hour']}:{REPORT_TIME['minute']} {TIMEZONE}",
        replace_existing=True
    )
    scheduler.start()
    print(f"✓ Scheduler started (Daily at {REPORT_TIME['hour']}:{REPORT_TIME['minute']})")

    # 2. Start Flask
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("✓ Flask health-check server started")

    # 3. Start Telegram Bot
    telegram_app = Application.builder().token(env_vars['TELEGRAM_TOKEN']).job_queue(None).build()
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(CommandHandler("report", report_command))
    
    print("✓ Telegram bot is polling...")
    telegram_app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    # If one argument 'report' is passed, just run the report once
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        run_report()
    else:
        main()
