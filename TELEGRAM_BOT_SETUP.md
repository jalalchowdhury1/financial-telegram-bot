# Telegram Bot Setup - Trigger Reports on Command

This guide shows you how to set up a Telegram bot that lets you trigger financial reports by sending `/report` in Telegram.

## Prerequisites

1. Your existing Telegram bot token (same one used for reports)
2. A GitHub Personal Access Token with `repo` scope

## Step 1: Create GitHub Personal Access Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name like "Telegram Bot Workflow Trigger"
4. Select scope: **✅ repo** (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

## Step 2: Set Environment Variables

Add this to your `~/.bashrc` or `~/.zshrc`:

```bash
export TELEGRAM_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_telegram_chat_id"
export GITHUB_TOKEN="your_github_personal_access_token"
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Step 3: Install Dependencies

```bash
cd /Users/jalalchowdhury/PycharmProjects/financial-telegram-bot
pip3 install -r requirements.txt
```

## Step 4: Run the Bot

### Option A: Run in foreground (for testing)
```bash
python3 telegram_bot.py
```

### Option B: Run in background (recommended)
```bash
nohup python3 telegram_bot.py > telegram_bot.log 2>&1 &
```

### Option C: Keep it running with screen
```bash
screen -S telegram-bot
python3 telegram_bot.py
# Press Ctrl+A then D to detach
# To reattach: screen -r telegram-bot
```

## Step 5: Test It!

Open Telegram and send these commands to your bot:

- `/start` - Shows welcome message
- `/help` - Shows available commands
- `/report` - **Triggers the financial report immediately!** 📊

## How It Works

1. You send `/report` in Telegram
2. The bot receives your message
3. Bot triggers GitHub Actions workflow via API
4. Workflow runs and generates all charts
5. Reports are sent to your Telegram (30-50 seconds)

## Stopping the Bot

If running in background:
```bash
ps aux | grep telegram_bot.py
kill <process_id>
```

## Troubleshooting

**Bot not responding:**
- Check if it's running: `ps aux | grep telegram_bot.py`
- Check logs: `tail -f telegram_bot.log`
- Verify environment variables are set: `echo $GITHUB_TOKEN`

**"Failed to trigger report" error:**
- Verify your GitHub token has `repo` scope
- Check the token hasn't expired
- Ensure the repository name in `telegram_bot.py` matches your repo

**Unauthorized access:**
- The bot only responds to your authorized chat ID
- Check that `TELEGRAM_CHAT_ID` matches your actual chat ID

## Daily Schedule Still Works!

The scheduled 8 AM EST reports will still run automatically. This bot just gives you an **on-demand** option! 🎯
