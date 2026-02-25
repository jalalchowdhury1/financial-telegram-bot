# 🏗️ System Architecture

This project consists of two primary services interacting with external financial data sources and delivering insights to the user.

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
        end

        subgraph "Render (Bot)"
            BOT["Python Bot Logic"]
            FLASK["Flask Health Check"]
            SCHED["APScheduler (Daily Jobs)"]
        end
    end

    subgraph "User Delivery"
        TELEGRAM["Telegram Chat / Bot"]
        WEB["Browser Dashboard"]
    end

    %% Data Extraction
    FRED --> API
    Stooq --> API
    CNN --> API
    Sheets --> API

    FRED --> BOT
    Stooq --> BOT
    CNN --> BOT
    Sheets --> BOT

    %% Delivery paths
    API --> DASH
    DASH --> WEB
    
    BOT --> TELEGRAM
    SCHED --> BOT
    TELEGRAM -- "/report" --> BOT
```

## 🧩 Component Breakdown

### 1. The Dashboard (Next.js)
- **Standardized Fetcher**: Uses a centralized utility in `lib/fetcher.js` with consistent timeout logic.
- **Business Logic**: Math for RSI and Moving Averages is isolated in `lib/finance.js`.
- **Constants**: All FRED Series IDs and external URLs are managed in `lib/constants.js`.

### 2. The Bot Package (Python)
- **Modular Design**: Broken into `fetchers.py`, `charts.py`, `assessment.py`, and `utils.py`.
- **Integrated Server**: The entrypoint `bot/main.py` runs a Flask server for Render's health checks and an internal scheduler.
- **AI-Friendly**: Fully type-hinted to ensure reliable AI-assisted updates.
