# 📊 Financial Telegram Bot & Dashboard

> [!IMPORTANT]
> **AI MAINTAINERS**: Read [AI_CONTEXT.md](docs/AI_MAINTAINER_GUIDE.md) before making changes.

A professional financial monitoring system featuring a real-time interactive dashboard deployed on Vercel, and a lightweight Telegram bot delivering daily market summaries.

---

## 🌟 Features

### 📈 Dashboard (`/dashboard`)
A Next.js application with live data from **FRED**, **Stooq**, and the **ExchangeRate API**.

#### Core Market Sections
| Section | Data |
|---|---|
| **SPY Chart** | Price, 200D MA, 52W High, RSI, Volume |
| **Fear & Greed Index** | CNN F&G via Stooq, with gauge visualization |
| **Economic Indicators** | Yield Curve, LEI, Profit Margin, Consumer Sentiment, Credit Spread, Real Yields, Claims |
| **Bull Market Checklist** | 8-factor checklist: NFCI, M2, Retail Sales, Housing Starts, Industrial Production, JOLTS, Durables, Savings Rate |
| **🌐 Global Markets** | FX, Commodities, Crypto, Real Estate & Rates (see below) |

#### Global Markets Widget
| Left Column — Real Estate & Rates | Right Column — FX, Commodities & Crypto |
|---|---|
| ZRI — US Median Monthly Rent | USD/CAD |
| MTGPMT — Estimated Monthly Mortgage | USD/INR |
| MORT30 — 30-Year Fixed Mortgage Rate | USD/BDT |
| TNX — 10-Year Treasury Yield | INR/BDT |
| T2Y — 2-Year Treasury Yield | CAD/INR |
| DXY — US Dollar Index | GOLD (Spot) |
| CL — Crude Oil WTI | BTC/USD |

#### Data Sources
- **[FRED](https://fred.stlouisfed.org/)** — Rates, Real Estate, Oil, Treasuries, Economic Indicators
- **[Stooq](https://stooq.com/)** — BTC, Gold, SPY (live market data)
- **[ExchangeRate-API](https://open.er-api.com/)** — USD/BDT, DXY basket calculation (free, no key required)
- **[Google Sheets](https://sheets.google.com/)** — Proprietary indicators (NotSoBoring, FrontRunner, AAII)
- **CNN** — Fear & Greed Index

### 🤖 Telegram Bot (`/bot`)
- **Daily Reports**: Automated morning market summaries sent to your Telegram channel.
- **AI Assessment**: Quantitative analysis powered by Groq/OpenAI/Gemini.
- **Commands**: `/report`, `/start`

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (for the Bot)
- **Node.js 18+** (for the Dashboard)
- **[FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)**
- **Telegram Bot** — Create one via [@BotFather](https://t.me/botfather)

### Dashboard (Local Development)
```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

### Bot (Local Development)
```bash
pip install -r requirements.txt
python -m bot.main
```

### Environment Variables
Create `dashboard/.env.local`:
```env
FRED_API_KEY=your_fred_api_key_here
```

Create `.env` in the root for the bot:
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
FRED_API_KEY=your_fred_api_key
# Optional: GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
```

---

## 📂 Project Structure

```
financial-telegram-bot/
├── bot/                        # Python Telegram bot
│   ├── main.py                 # Bot entry point
│   └── ...
├── dashboard/                  # Next.js web dashboard
│   ├── app/
│   │   ├── page.js             # Main dashboard page
│   │   ├── globals.css         # Design system & CSS variables
│   │   └── api/
│   │       ├── fred/           # FRED economic indicators (batched)
│   │       ├── market-extra/   # Global Markets: FX, Oil, Rates
│   │       ├── spy/            # SPY price & technicals (Stooq)
│   │       ├── fear-greed/     # CNN Fear & Greed Index
│   │       ├── assessment/     # AI market assessment
│   │       ├── sheets/         # Google Sheets proprietary data
│   │       └── last-run/       # Bot last-run timestamp
│   ├── components/
│   │   ├── ExtraMarketsGrid.js # 🌐 Global Markets widget
│   │   ├── BullChecklist.js    # 8-factor bull market checklist
│   │   ├── SpyChart.js         # Interactive SPY price chart
│   │   ├── MiniChart.js        # Reusable sparkline chart
│   │   ├── EconomicIndicatorGrid.js
│   │   ├── Gauge.js            # Fear & Greed gauge
│   │   ├── ErrorBoundary.js
│   │   ├── Skeleton.js
│   │   └── ...
│   └── lib/
│       ├── constants.js        # FRED series IDs, API URLs
│       └── fetcher.js          # Standardized HTTP fetcher
├── docs/                       # Technical documentation
├── scripts/                    # Dev & deployment utilities
├── requirements.txt            # Bot Python dependencies
└── README.md
```

---

## 🏗️ Architecture

```
Vercel (Dashboard)          Render (Bot)
┌─────────────────┐         ┌─────────────┐
│  Next.js App    │         │ Python Bot  │
│  ┌───────────┐  │         │ ┌─────────┐ │
│  │ page.js   │  │ ◄──────►│ │ main.py │ │
│  └───────────┘  │         │ └─────────┘ │
│  API Routes:    │         └─────────────┘
│  /api/fred      ├── FRED API
│  /api/market-   ├── Stooq
│    extra        ├── ExchangeRate-API
│  /api/spy       └── Google Sheets
└─────────────────┘
```

### Key Design Decisions
- **Batched FRED Requests**: All FRED API calls are sent in batches of 3 with 200ms delays to avoid Vercel Datacenter IP blocking (HTTP 429).
- **Multi-Source Data**: FX pairs use FRED daily spot rates; BTC/Gold use Stooq (live); BDT/DXY use ExchangeRate-API (no key required).
- **DXY Calculation**: Computed server-side using the official ICE basket formula from live exchange rates.
- **Rent Proxy**: `CUUR0000SEHA` (FRED Rent of Primary Residence index) × 4.41 to approximate US median monthly rent (~$1,950).
- **Mortgage Payment**: Calculated from FRED median home price (`MSPUS`) × 80% (20% down) + current 30-year rate.

---

## 📜 Bot Commands

| Command | Description |
|---|---|
| `/report` | Triggers an immediate financial report |
| `/start` | Shows the welcome message and guide |

---

## 📄 License
MIT
