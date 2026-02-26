# 🤖 AI Maintainer Guide

This document is for AI agents (like Antigravity, Claude, or ChatGPT) assisting with this project. It provides immediate context to ensure continuity across different chat sessions.

## 📌 Project Identity
- **Repository**: `https://github.com/jalalchowdhury1/financial-telegram-bot`
- **Main Branch**: `main`
- **Structure**: Monorepo with `bot/` and `dashboard/` at the root.

## 🏗️ Deployment Architecture
- **Web Dashboard**: Next.js (`dashboard/` folder). Deployed on **Vercel**.
- **Bot Service**: Python (`bot/` folder). Deployed on **Render**.
  - *Start Command*: `python -m bot.main`
  - *Internal Scheduler*: APScheduler runs in `bot/main.py`.

## 📂 Source of Truth Locations
| Goal | File / Directory |
| :--- | :--- |
| **JS Constants** | `dashboard/lib/constants.js` |
| **Python Config** | `bot/config.py` |
| **API Endpoints** | `dashboard/app/api/` |
| **UI Components** | `dashboard/components/` (Modular structure v7.0) |
| **Main Page** | `dashboard/app/page.js` (Orchestration only) |
| **Bot Logic** | `bot/` (Modular package - "Lite" version) |
| **Deployment** | `render.yaml` (Render), Dashboard Settings (Vercel) |

## 🛡️ Data Resilience & Sourcing
The project implements a cascading fallback strategy for critical metrics.

| Metric Group | Primary Source | Fallback Mechanism |
| :--- | :--- | :--- |
| **SPY Data** | Stooq API | Yahoo Finance API |
| **S&P 500 P/E** | multpl.com | Yahoo Finance (Operating P/E * 1.07) |
| **F&G Index** | CNN Business | RapidAPI -> Yahoo VIX Proxy |
| **AI Assessment** | OpenRouter (Free) | GPT-4o -> Groq -> Rule-Based Fallback |
| **Eco Data** | FRED API | Graceful skip (Promised-based isolation) |

## 🛠️ Common AI Tasks
1. **Adding a New Indicator**:
   - Add Series ID to `bot/config.py` and `dashboard/lib/constants.js`.
   - Update `bot/fetchers.py` or new Dashboard API route.
   - Create or update relevant components in `dashboard/components/`.
   - Update `dashboard/app/page.js` to include the new component.
2. **Updating the Bot Schedule**:
   - Edit `REPORT_TIME` in `bot/config.py`.
3. **Core AI Workflow**:
   - Always check `.gemini/antigravity/brain/` for `task.md` and `implementation_plan.md` to understand current work state.
   - Use `task_boundary` and `notify_user` tools to maintain transparency.

## ⚠️ Important Constraints
- **Lightweight Reporting**: The Telegram bot is strictly "Lite" (text-only).
  - No plotting libraries (removed Matplotlib/Seaborn).
  - Isolated: Only the Google Sheet indicator block is sent via `/report`.
  - Fast & Low Resource: Designed to run efficiently on Render's free/basic tiers.
- **Modular Dashboard**:
  - Keep `dashboard/app/page.js` clean. Extract all logic into components.
  - Follow the v7.0 design: Glassmorphism, dark mode, vibrant status badges.
- **Data Integrity**:
  - Use `clean_val` helper in `bot/fetchers.py` for sheet data.
  - Use `proxyFetch` in Dashboard routes for robust fetching.
- **Environment**: Always update `requirements.txt` when adding Python dependencies.
