# Financial Telegram Bot & Dashboard

A personal market intelligence tool:

- **Daily Telegram report** — an AWS Lambda (`financial-telegram-report`) assembles a
  market brief (Google-Sheet indicators + SPY snapshot) and sends it to Telegram every
  morning via an EventBridge schedule.
- **Live dashboard** — a Next.js app (in [`/dashboard`](./dashboard)) showing SPY, FX,
  commodities, rates, FRED economic indicators, CNN Fear & Greed, and trending Polymarket
  markets. Deployed on Vercel: <https://financial-telegram-bot-beryl.vercel.app/>

Data comes from a resilient multi-source waterfall (yfinance, Polygon, Finnhub, Stooq,
FRED, Google Sheets, and more), so the dashboard never goes blank when a source fails.

---

## 🛠 Maintainers & AI agents: read **[AGENTS.md](./AGENTS.md)** first

**[`AGENTS.md`](./AGENTS.md) is the single source of truth** for how to safely change,
deploy, and operate this project — Lambda packaging rules, the API-Gateway architecture,
the never-throw dashboard routes, known gotchas, and a pre-commit checklist. Read it
before touching the backend or the dashboard.
