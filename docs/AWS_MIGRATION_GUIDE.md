# AWS Lambda Migration Guide: Financial Telegram Bot

## Objective
Migrate the daily financial report generation from **GitHub Actions** (`.github/workflows/daily_report.yml`) to **AWS Lambda**. This guide focuses on including a comprehensive **SPY Market Overview** and **Currency/FX** metrics in the Telegram report, triggered by an AWS EventBridge daily cron schedule.

## 1. Migration Overview
Currently, the Telegram bot runs on GitHub Actions and sends a lightweight text report. The goal is to:
1. Move the execution environment to AWS Lambda for better reliability and faster execution.
2. Re-enable and enhance the SPY Market Overview message.
3. Add comprehensive Currency (FX), Commodities, and Crypto metrics to the daily Telegram message.

## 2. Updated Report Format

### 📊 SPY Market Overview
The Telegram bot will generate a richly formatted summary using Telegram MarkdownV2 or HTML:
```text
📊 *SPY Market Overview*
Neutral
$650.34
▲ 2.91% today
▼ 1.76% below 200d MA
-6.81% from 52wk high ($697.84)

• *200d MA:* $662.02
• *52w High:* $697.84
• *3Y Return:* +58.25%
• *9d RSI:* 44.73 
(RSI Scale: 0 - 30 - 50 - 70 - 100) -> NEUTRAL
```

### 💱 Currencies, Crypto & Commodities
```text
💱 *FX, Commodities & Crypto*
• USD/BDT: 122.72
• USD/CAD: 1.3908
• GOLD: $4,681.56
• BTC: $67,796
• CAD/BDT: 88.16
• USD/INR: 93.45
• INR/BDT: 1.3075
• CAD/INR: 68.32
```

## 3. Implementation Plan

### Step 1: Create the AWS Lambda Handler (`lambda_function.py`)
Create a new entry point that AWS Lambda can invoke:
- Initialize the data fetchers (`bot.fetchers`).
- Retrieve SPY data using `web.DataReader('SPY.US', 'stooq')`.
- Retrieve Currency & Commodity data (likely using the same sources as your Next.js Dashboard, e.g., FRED or Yahoo Finance/ExchangeRate-API).
- Format the data into the structure above.
- Call the `telegram` API to send the message to your `TELEGRAM_CHAT_ID`.

### Step 2: Environment Variables needed in AWS
Configure the following Environment Variables in the AWS Lambda Console:
- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`
- `FRED_API_KEY`
- `ALPHA_VANTAGE_API_KEY` (if used)

### Step 3: Deployment Package
Since this bot uses `pandas`, `pandas-datareader`, and `requests`, we will need to:
1. Package the dependencies using Docker or a Lambda Layer because libraries like `pandas` might exceed the base unzipped size or require specific C-compiled binaries for Amazon Linux.
2. Zip the `bot/` directory and upload it to AWS Lambda.

### Step 4: AWS EventBridge (CloudWatch Events)
Replace the GitHub Actions `cron` schedule by creating an Amazon EventBridge rule:
- **Schedule Expression:** `cron(M H * * ? *)` (Matches your `REPORT_TIME`).
- **Target:** The newly created Lambda function.

## 4. Next Steps
If this plan aligns with your expectations, I will:
1. Write the Python Lambda Handler code (`bot/lambda_function.py`) to fetch and format the SPY and Currency metrics.
2. Update the existing `fetchers.py` if missing currency fetching logic.
3. Provide the exact deployment commands to package and upload this code to AWS Lambda.
