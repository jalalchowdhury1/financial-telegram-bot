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
from typing import Dict, Any, Optional

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Import from the modular bot package
from bot.utils import load_environment_variables, send_to_telegram, report_marker
from bot.fetchers import fetch_google_sheet_indicators

from bot.config import TIMEZONE, REPORT_TIME

# Flask app for health checks
flask_app = Flask(__name__)
global_scheduler: Optional[BackgroundScheduler] = None

@flask_app.route('/')
def health_check():
    return {'status': 'running', 'bot': 'financial-telegram-bot-lite', 'telegram_polling': False}, 200

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
        if not gs_text:
            print("⚠ Google Sheets returned empty result — nothing to send.")
            print(report_marker(False, reason="empty_content"))
            return False

        sent = send_to_telegram(env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'], caption=gs_text, silent=True)  # overnight report — no buzz (11 Sep 2026)
        if not sent:
            print(report_marker(False, sections=1, reason="telegram_delivery"))
            return False
        print("✓ Sent Google Sheets indicators.")

        pass

        pass
            
        print("\n✓ Lightweight report processing complete.")
        print(report_marker(True, sections=1, errors=0))
        return True
    except Exception as e:
        print(f"CRITICAL ERROR in report generation: {e}")
        print(report_marker(False, reason="exception"))
        return False

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

    load_environment_variables()  # fail fast on missing env, as before
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

    # 2. Flask in the foreground keeps the web process (Render) alive.
    # NO Telegram polling here, ever (removed 2026-10-10). TELEGRAM_TOKEN is @TweetSyn_bot,
    # the alerts bot, whose updates belong to health-hub's webhook (/api/defensive: the
    # Silent-digest card buttons). run_polling() calls deleteWebhook on start, and Render
    # re-ran this on EVERY push to main: 26 wiped webhooks 26 Sep-10 Oct, each one a
    # dead-button window until health-hub's 5-min tick re-set it. /report and /start never
    # got an update anyway (the webhook owns them). tests/test_no_telegram_polling.py guards it.
    print("✓ Flask health-check server starting (no Telegram polling)")
    run_flask()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        sys.exit(0 if run_report() else 1)
    else:
        main()
