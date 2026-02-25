# 🤖 AI Maintainer Guide

This document is for AI agents (like Antigravity, Claude, or ChatGPT) assisting with this project. It provides immediate context to ensure continuity across different chat sessions.

## 📌 Project Identity
- **Repository**: `https://github.com/jalalchowdhury1/financial-telegram-bot`
- **Main Branch**: `main`

## 🏗️ Deployment Architecture
- **Web Dashboard**: Next.js (Dashboard folder). Deployed on **Vercel**.
- **Bot Service**: Python (Bot folder). Deployed on **Render**.
  - *Start Command*: `python -m bot.main` (or `python telegram_bot_server.py` for legacy compatibility).
  - *Internal Scheduler*: APScheduler runs in `bot/main.py`.

## 📂 Source of Truth Locations
| Goal | File / Directory |
| :--- | :--- |
| **JS Constants** | `dashboard/lib/constants.js` |
| **Python Config** | `bot/config.py` |
| **API Endpoints** | `dashboard/app/api/` |
| **UI Components** | `dashboard/components/` (Modular structure) |
| **Main Page** | `dashboard/app/page.js` (Orchestration only) |
| **Bot Logic** | `bot/` (Modular package) |
| **Deployment** | `render.yaml` (Render), Dashboard Settings (Vercel) |

## 🛠️ Common AI Tasks
1. **Adding a New Indicator**:
   - Add Series ID to `bot/config.py` and `dashboard/lib/constants.js`.
   - Update `bot/fetchers.py` or new Dashboard API route.
   - Create or update relevant components in `dashboard/components/`.
   - Update `dashboard/app/page.js` to include the new component.
2. **Updating the Bot Schedule**:
   - Edit `REPORT_TIME` in `bot/config.py`.
3. **Commit & Push**:
   - Always verify the remote with `git remote -v`.
   - Use descriptive commit messages following the "Phase X: [Description]" format.

## ⚠️ Important Constraints
- **Lightweight Reporting**: The Telegram report is strictly "Lite" (text-only).
  - Isolated: Only the Google Sheet indicator block is sent via `/report`.
  - No Status: Avoid sending "Generating..." or other transition messages to keep the chat clean.
- **Data Cleaning**: Use the `clean_val` helper in `bot/fetchers.py` to strip trailing digits or junk characters from Google Sheet cells.
- **Do not** add heavy plotting libraries to the Bot.
- **Keep it modular**: Extract new UI features into standalone components in `dashboard/components/`.
- **Ensure** `requirements.txt` is updated after any new Python imports.
