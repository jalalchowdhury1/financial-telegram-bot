# 🏗️ System Architecture

This project consists of two primary services: an **Interactive Visualization Dashboard** and a **Lightweight Text Reporter**.

## 🛰️ Data Flow Diagram

```mermaid
graph TD
    subgraph "External Providers"
        FRED["FRED API (Economic Data)"]
        Stooq["Stooq API (SPY Stats)"]
        CNN["CNN Business (Fear & Greed)"]
        Sheets["Google Sheets (Custom Data)"]
    end

    subgraph "Cloud Services"
        subgraph "Vercel (Dashboard)"
            DASH["Next.js Web App"]
            API["API Routes /lib"]
            VIZ["Visualization Logic (D3/SVG)"]
        end

        subgraph "Render (Bot Lite)"
            BOT["Python Text Reporter"]
            FLASK["Flask Health Check"]
            SCHED["APScheduler (Daily Jobs)"]
        end
    end

    subgraph "User Delivery"
        TELEGRAM["Telegram Chat / Bot (Text Only)"]
        WEB["Browser Dashboard (Charts & Charts)"]
    end

    %% Data Extraction
    FRED --> API
    Stooq --> API
    CNN --> API
    Sheets --> API

    Stooq --> BOT
    Sheets --> BOT

    %% Delivery paths
    API --> DASH
    DASH --> WEB
    
    BOT --> TELEGRAM
    SCHED --> BOT
    TELEGRAM -- "/report" --> BOT
```

## 🧩 Component Breakdown

### 1. The Dashboard (Next.js - Full Visual Studio)
- **Deep Dive Visualization**: Handles all complex charting (Yield Curve, Profit Margins, RSI Gagues).
- **Standardized Fetcher**: Uses `lib/fetcher.js` with consistent timeout logic.
- **Business Logic**: Math for RSI and Moving Averages is isolated in `lib/finance.js`.

### 2. The Bot Lite (Python - High Speed)
- **Compact Text Reports**: Focused on immediate, high-value text summaries.
- **Dependency Optimized**: Running without heavy plotting libraries (matplotlib/seaborn) for ultra-fast startup and execution.
- **Integrated Server**: The entrypoint `bot/main.py` runs a Flask server for Render's health checks and an internal scheduler.
- **AI-Friendly**: Fully type-hinted and modular.
