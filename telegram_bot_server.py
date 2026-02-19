"""
Telegram Bot with Web Server for Cloud Deployment
Combines the Telegram bot with a simple Flask web server for health checks
Includes automatic daily scheduler for reports at 4:15 AM EST
"""

import os
import sys
import asyncio
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
from flask import Flask
from threading import Thread
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from datetime import datetime

# Load environment variables
load_dotenv()

# Environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = 'jalalchowdhury1/financial-telegram-bot'

# Flask app for health checks
app = Flask(__name__)

@app.route('/')
def health_check():
    return {'status': 'running', 'bot': 'financial-telegram-bot'}, 200

@app.route('/health')
def health():
    return {'status': 'healthy'}, 200


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
    app.run(host='0.0.0.0', port=port)


def main():
    """Start Flask server, scheduler, and Telegram bot"""
    print("Starting Financial Telegram Bot with Web Server and Scheduler...")

    # Validate environment
    validate_environment()

    # Set up scheduler for automatic reports
    est = pytz.timezone('America/New_York')
    scheduler = BackgroundScheduler(timezone=est)

    # Production schedule: Daily at 4:15 AM EST
    scheduler.add_job(
        scheduled_report,
        trigger=CronTrigger(hour=4, minute=15, timezone=est),
        id='daily_report',
        name='Daily Financial Report at 4:15 AM EST',
        replace_existing=True
    )

    # TEST SCHEDULE: Run at 9:20 AM EST today for testing
    # Remove this after successful test!
    scheduler.add_job(
        scheduled_report,
        trigger=CronTrigger(hour=9, minute=20, timezone=est),
        id='test_report',
        name='TEST: Report at 9:20 AM EST',
        replace_existing=True
    )

    scheduler.start()
    print("✓ Scheduler started")
    print("  → Daily reports: 4:15 AM EST")
    print("  → TEST: 9:20 AM EST (remove after testing)")

    # Start Flask in background thread
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("✓ Web server started")

    # Create Telegram bot application
    telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command handlers
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("report", report_command))

    print("✓ Telegram bot is running! Send /report to generate a financial report.")

    # Fix for Python 3.10+ asyncio event loop issue
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Start Telegram bot polling
    telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
