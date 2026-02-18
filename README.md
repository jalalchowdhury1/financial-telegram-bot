# Financial Charts Telegram Bot

Automated daily financial charts delivered to your Telegram chat every morning at 8:00 AM EST.

## Features

- **Yield Curve Inversion Chart**: 10Y-2Y Treasury spread over the last 20 years
- **US Economy-Wide Profit Margin**: Corporate Net Operating Surplus / GDP
- **Automated Daily Delivery**: GitHub Actions runs every morning
- **Professional Visualizations**: High-quality charts with matplotlib and seaborn

## Setup Instructions

### 1. Get Your API Keys

**FRED API Key:**
1. Visit https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account
3. Request an API key

**Telegram Bot:**
1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` and follow the prompts
3. Save your bot token (looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz123456789`)

**Telegram Chat ID:**
1. Start a chat with your bot and send any message
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Look for `"chat":{"id":123456789}` in the response
4. Save this number

### 2. Configure GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions → New repository secret

Add these three secrets:
- `FRED_API_KEY`
- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`

### 3. Test Locally (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FRED_API_KEY="your_key"
export TELEGRAM_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Run the script
python main.py
```

### 4. Deploy to GitHub

Push this repository to GitHub and the workflow will run automatically every day at 8:00 AM EST.

To test immediately: Go to Actions tab → Daily Financial Report → Run workflow

## File Structure

```
financial-telegram-bot/
├── main.py                           # Main script
├── requirements.txt                  # Python dependencies
├── .github/
│   └── workflows/
│       └── daily_report.yml         # GitHub Actions workflow
└── README.md                        # This file
```

## Generated Charts

- `yield_curve.png` - Yield curve spread visualization
- `profit_margin.png` - Profit margin trend

## Troubleshooting

**Charts not sending?**
- Check that all three secrets are correctly set in GitHub
- View the Actions logs for error messages
- Verify your Telegram bot token and chat ID

**Data fetch errors?**
- FRED API may be temporarily unavailable
- Check if your API key is valid
- Some data series update quarterly (slight delays are normal)

## License

MIT
