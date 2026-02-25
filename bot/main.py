"""
Modular entrypoint for the financial-telegram-bot.
Orchestrates data fetching and lightweight Telegram text reporting.
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

from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Import from the modular bot package
from bot.utils import load_environment_variables, send_to_telegram
from bot.fetchers import fetch_google_sheet_indicators, fetch_spy_stats
from bot.assessment import generate_ai_assessment
from bot.config import TIMEZONE, REPORT_TIME

# Flask app for health checks
flask_app = Flask(__name__)
global_scheduler: Optional[BackgroundScheduler] = None

@flask_app.route('/')
def health_check():
    return {'status': 'running', 'bot': 'financial-telegram-bot-lite'}, 200

@flask_app.route('/health')
def health():
    return {'status': 'healthy'}, 200

def run_report():
    """Execute a lightweight text-only report generation and delivery sequence"""
    print("\n" + "=" * 60)
    print("Generating Lightweight Financial Report...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    env_vars = load_environment_variables()
    
    try:
        # 1. Fetch Google Sheets Indicators (Primary content requested by user)
        gs_text = fetch_google_sheet_indicators()
        if gs_text:
            send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], caption=gs_text)
            print("✓ Sent Google Sheets indicators.")

        # 2. Add a quick text summary for SPY (Disabled per user request)
        # try:
        #     spy = fetch_spy_stats()
        #     spy_text = f"📈 *SPY Market Snapshot*\nPrice: ${spy['current']:.2f} ({spy['change_pct']:+.2f}%)\n9D RSI: {spy['rsi_9d']:.2f}"
        #     send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], caption=spy_text)
        #     print("✓ Sent SPY text summary.")
        # except Exception as e:
        #     print(f"Skipping SPY summary: {e}")
        pass

        # 3. AI Assessment (Disabled per user request)
        # try:
        #     assessment_data = {
        #         'spy_rsi': spy['rsi_9d'] if 'spy' in locals() else 50.0,
        #         'vix_current': 20.0,
        #         'fear_greed': 50,
        #     }
        #     assessment = generate_ai_assessment(assessment_data)
        #     send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], caption=assessment)
        #     print("✓ Sent AI Market Assessment.")
        # except Exception as e:
        #     print(f"Skipping AI assessment: {e}")
            
        print("\n✓ Lightweight report processing complete.")
        return True
    except Exception as e:
        print(f"CRITICAL ERROR in report generation: {e}")
        return False

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /report command"""
    env_vars = load_environment_variables()
    if str(update.effective_chat.id) != env_vars['TELEGRAM_CHAT_ID']:
        return

    # Removed "Generating" reply per user request
    # await update.message.reply_text("🔄 Generating your financial summary...")
    
    # Run in a separate thread to avoid blocking the bot's event loop
    def job():
        run_report()
    
    Thread(target=job).start()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text("👋 Financial Report Bot (Lite)\n\nUse /report for a quick market summary.")

def run_flask():
    """Run Flask server for health checks"""
    port = int(os.environ.get('PORT', 10000))
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

def main():
    """Start the integrated bot service"""
    global global_scheduler
    print("Starting Lightweight Financial Bot Service...")

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
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        run_report()
    else:
        main()
