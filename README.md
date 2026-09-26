# Financial Telegram Bot & Dashboard

A personal market intelligence tool:

- **Daily Telegram report** — an AWS Lambda (`financial-telegram-report`) assembles a
  market brief (Google-Sheet indicators + SPY snapshot) and sends it to Telegram every
  morning via an EventBridge schedule.
- **Live dashboard** — a Next.js app (in [`/dashboard`](./dashboard)) showing SPY, FX,
  commodities, rates, FRED economic indicators, CNN Fear & Greed, trending Polymarket
  markets, and a factor row (value, momentum, quality, small caps, low vol vs the market,
  1 month to 40 years). It opens instantly from the last visit's numbers (tagged 🕐 until
  live ones land), leads with a "What moved" line (the 5 most unusual moves since the last
  close), and on a phone has pull-to-refresh, a ticking "updated 3 min ago" and an offline
  banner. Deployed on Vercel: <https://financial-telegram-bot-beryl.vercel.app/>

Data comes from a resilient multi-source waterfall (yfinance, Polygon, Finnhub, Stooq,
FRED, Google Sheets, and more), so the dashboard never goes blank when a source fails.

---

## 🛠 Maintainers & AI agents: read **[AGENTS.md](./AGENTS.md)** first

**[`AGENTS.md`](./AGENTS.md) is the single source of truth** for how to safely change,
deploy, and operate this project — Lambda packaging rules, the API-Gateway architecture,
the never-throw dashboard routes, known gotchas, and a pre-commit checklist. Read it
before touching the backend or the dashboard.
