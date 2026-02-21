"""
Telegram Bot with Web Server for Cloud Deployment
Combines the Telegram bot with a simple Flask web server for health checks
Includes automatic daily scheduler for reports at 4:15 AM EST
"""

import os
import sys
import asyncio
import requests
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
from flask import Flask
from threading import Thread
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = 'jalalchowdhury1/financial-telegram-bot'

# Flask app for health checks
app = Flask(__name__)

# Global scheduler reference for status checking
global_scheduler = None

@app.route('/')
def health_check():
    return {'status': 'running', 'bot': 'financial-telegram-bot'}, 200

@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

@app.route('/scheduler-status')
def scheduler_status():
    """Check if scheduler is running and show next scheduled jobs"""
    est = pytz.timezone('America/New_York')
    current_time = datetime.now(est).strftime('%Y-%m-%d %I:%M:%S %p %Z')

    if global_scheduler is None:
        return {'error': 'Scheduler not initialized', 'current_time': current_time}, 500

    jobs = []
    for job in global_scheduler.get_jobs():
        next_run = job.next_run_time
        if next_run:
            next_run_est = next_run.astimezone(est).strftime('%Y-%m-%d %I:%M:%S %p %Z')
        else:
            next_run_est = 'None'

        jobs.append({
            'id': job.id,
            'name': job.name,
            'next_run': next_run_est
        })

    return {
        'current_time': current_time,
        'scheduler_running': global_scheduler.running,
        'jobs': jobs
    }, 200


def validate_environment():
    """Check that all required environment variables are set"""
    required_vars = {
        'TELEGRAM_TOKEN': TELEGRAM_TOKEN,
        'GITHUB_TOKEN': GITHUB_TOKEN,
        'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID
    }

    missing = [var for var, val in required_vars.items() if not val]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    return True


def trigger_github_workflow():
    """Trigger the GitHub Actions workflow via repository_dispatch"""
    url = f'https://api.github.com/repos/{GITHUB_REPO}/dispatches'
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {GITHUB_TOKEN}'
    }
    data = {
        'event_type': 'telegram-report'
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error triggering workflow: {e}")
        if hasattr(response, 'text'):
            print(f"Response: {response.text}")
        return False


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /report command"""
    # Verify the message is from the authorized chat
    if str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
        print(f"Unauthorized access attempt from chat_id: {update.effective_chat.id}")
        return

    # Send acknowledgment
    await update.message.reply_text("🔄 Generating financial report...")

    # Trigger the workflow
    if trigger_github_workflow():
        await update.message.reply_text(
            "✅ Report generation started!\n"
            "📊 Charts will arrive in ~40-50 seconds."
        )
    else:
        await update.message.reply_text(
            "❌ Failed to trigger report.\n"
            "Please check your GitHub token and try again."
        )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    if str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
        return

    await update.message.reply_text(
        "👋 Financial Report Bot\n\n"
        "Available commands:\n"
        "/report - Generate financial report now\n"
        "/start - Show this help message\n\n"
        "📅 Daily reports are automatically sent at 8:00 AM EST"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    if str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
        return

    await update.message.reply_text(
        "📊 Financial Report Bot Commands:\n\n"
        "/report - Trigger report generation immediately\n"
        "/help - Show this help message\n"
        "/start - Show welcome message\n\n"
        "The bot automatically sends reports daily at 8 AM EST."
    )


def scheduled_report():
    """Trigger report automatically on schedule"""
    est = pytz.timezone('America/New_York')
    current_time = datetime.now(est).strftime('%Y-%m-%d %I:%M:%S %p %Z')
    print(f"🕐 Scheduled report triggered at {current_time}")

    if trigger_github_workflow():
        print("✅ Scheduled report triggered successfully")
    else:
        print("❌ Scheduled report failed to trigger")


def run_flask():
    """Run Flask server"""
    port = int(os.environ.get('PORT', 10000))
    print(f"✓ Starting Flask server on port {port}")
    # Disable Flask debug mode and request logs for production
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


def main():
    """Start Flask server, scheduler, and Telegram bot"""
    global global_scheduler

    print("Starting Financial Telegram Bot with Web Server and Scheduler...")

    # Validate environment
    validate_environment()

    # Set up scheduler for automatic reports
    try:
        est = pytz.timezone('America/New_York')
        scheduler = BackgroundScheduler(timezone=est)
        global_scheduler = scheduler

        # Production schedule: Daily at 4:15 AM EST
        scheduler.add_job(
            scheduled_report,
            trigger=CronTrigger(hour=4, minute=15, timezone=est),
            id='daily_report',
            name='Daily Financial Report at 4:15 AM EST',
            replace_existing=True
        )

        # TESTING: Hourly reports for today (REMOVE AFTER TESTING)
        scheduler.add_job(
            scheduled_report,
            trigger=CronTrigger(hour='*', minute=0, timezone=est),  # Every hour on the hour
            id='hourly_test',
            name='Hourly Test Report',
            replace_existing=True
        )

        # TESTING: Quick diagnostic in 5 minutes
        test_time = datetime.now(est) + timedelta(minutes=5)
        scheduler.add_job(
            scheduled_report,
            trigger='date',
            run_date=test_time,
            id='quick_test',
            name='Quick Diagnostic Test (5 min)',
            replace_existing=True
        )

        scheduler.start()

        # Log scheduler info
        print("✓ Scheduler started - Daily reports at 4:15 AM EST")
        current_time_est = datetime.now(est).strftime('%Y-%m-%d %I:%M:%S %p %Z')
        print(f"  Current time: {current_time_est}")

        for job in scheduler.get_jobs():
            next_run = job.next_run_time
            if next_run:
                next_run_est = next_run.astimezone(est).strftime('%Y-%m-%d %I:%M:%S %p %Z')
                print(f"  → Next report: {next_run_est}")
    except Exception as e:
        print(f"ERROR: Failed to initialize scheduler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Start Flask in background thread
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Give Flask a moment to start before Render checks health
    time.sleep(2)
    print("✓ Web server started and ready")

    # Create Telegram bot application (disable built-in job queue since we use APScheduler)
    telegram_app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .job_queue(None)  # Disable built-in job queue to avoid conflicts
        .build()
    )

    # Add command handlers
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("report", report_command))

    print("✓ Telegram bot is running! Send /report to generate a financial report.")

    # Start Telegram bot polling (this is a blocking call)
    # run_polling() already handles event loop creation internally
    try:
        telegram_app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    except Exception as e:
        print(f"ERROR: Telegram bot polling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
