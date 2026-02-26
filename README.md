# 📊 Financial Telegram Bot & Dashboard

> [!IMPORTANT]
> **AI MAINTAINERS**: Read [AI_CONTEXT.md](file:///Users/jalalchowdhury/PycharmProjects/financial-telegram-bot/financial-telegram-bot/AI_CONTEXT.md) before making changes.

A professional financial monitoring system delivered via a lightweight Telegram bot and a deep-dive interactive dashboard.

## 🌟 Features

- **Automated Daily Summaries**: Compact text-only reports delivered every morning.
- **Interactive Dashboard**: Next.js-powered UI with real-time FRED and Stooq data.
- **AI-Powered Assessment**: Quantitative market analysis using LLMs (Groq/OpenAI/Gemini).
- **Core Indicators**:
  - Yield Curve (10Y-2Y Spread) with recession shading.
  - Economy-wide Profit Margins.
  - SPY Stats (200D MA, 52W High, 9D RSI).
  - Fear & Greed Index.
  - Proprietary Sheets Data (NotSoBoring, FrontRunner).

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+** (for the Bot)
- **Node.js 18+** (for the Dashboard)
- **FRED API Key**: [Get it here](https://fred.stlouisfed.org/docs/api/api_key.html)
- **Telegram Bot**: Created via [@BotFather](https://t.me/botfather)

### 2. Configuration
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### 3. Local Development

**Run the Dashboard:**
```bash
cd dashboard
npm install
npm run dev
```

**Run the Bot:**
```bash
pip install -r requirements.txt
python -m bot.main
```

## 📂 Project Structure

- `bot/`: Python Telegram Bot (Report Service)
- `dashboard/`: Next.js Web Application
- `docs/`: Technical Documentation & [AI Maintainer Guide](docs/AI_MAINTAINER_GUIDE.md)
- `scripts/`: Development and deployment utilities
- `requirements.txt`: Bot dependencies

---

## 🏗️ Architecture

- **Dashboard**: Next.js deployed on Vercel.
- **Bot**: Python service deployed on Render.

## 📜 Commands

Send these to your Telegram bot:
- `/report`: Triggers an immediate financial report generation.
- `/start`: Shows the welcome message and help guide.

## 🤝 Contributing

This project is optimized for AI maintainability with centralized configurations and explicit type hinting.

## 📄 License
MIT
