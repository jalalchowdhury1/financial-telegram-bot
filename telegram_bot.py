"""
Telegram Bot - Trigger GitHub Workflow on Command
Listens for /report command and triggers the daily financial report workflow
"""

import os
import sys
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Personal Access Token with repo scope
GITHUB_REPO = 'jalalchowdhury1/financial-telegram-bot'  # Your repo


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
        print("\nRequired setup:")
        print("  export TELEGRAM_TOKEN='your_telegram_bot_token'")
        print("  export TELEGRAM_CHAT_ID='your_telegram_chat_id'")
        print("  export GITHUB_TOKEN='your_github_personal_access_token'")
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


def main():
    """Start the bot"""
    print("Starting Telegram Bot...")

    # Validate environment
    validate_environment()

    # Create application
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("report", report_command))

    print("✓ Bot is running! Send /report to generate a financial report.")
    print("  Press Ctrl+C to stop")

    # Start polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
