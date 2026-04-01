# Financial Telegram Bot & Dashboard - AI Maintainer Guide

> **CRITICAL READING FOR ALL LLMs**: Do NOT modify, deploy, or touch the backend of this repository without reading this document entirely. This project seamlessly integrates a Next.js `dashboard` with an **AWS Lambda Python Backend**. Overlooking these rules will result in production crashes, 500 API Gateway errors, and rate-limit IP bans.

---

## 1. Project Architecture
- **Frontend**: A Next.js application located in `/dashboard`. It fetches live data from the AWS Lambda backend.
- **Backend (Python)**: An AWS Lambda function (`financial-telegram-report`) deployed via zip packaged code.
- **Fetchers (`bot/fetchers.py`)**: A multi-tier waterfall system that fetches from Yahoo Finance (yfinance), Polygon, Finnhub, Stooq, FRED, and Google Sheets. 

---

## 2. AWS Lambda Deployment Rules (Crucial!)

**You cannot simply push local dependencies to AWS Lambda.** If you build `numpy` or `pandas` on a Mac, it will crash `Runtime.ImportModuleError` on Lambda`s Linux environment.

**How to package dependencies for deployment:**
When adding or updating `aws/requirements-lambda.txt`, you MUST compile for the `manylinux2014_x86_64` platform using Python 3.11:

```bash
# 1. Install to a fresh package/ folder
pip install \
    --platform manylinux2014_x86_64 \
    --target=package \
    --implementation cp \
    --python-version 3.11 \
    --only-binary=:all: \
    -r aws/requirements-lambda.txt

# 2. Zip the dependencies
cd package && zip -r ../deployment_full.zip . && cd ..

# 3. Add the application code
zip -g deployment_full.zip lambda_handler.py bot/*.py

# 4. Deploy
aws lambda update-function-code \
    --function-name financial-telegram-report \
    --zip-file fileb://deployment_full.zip
```
*(Always clean up the `package/` and `.zip` so they do not pollute the git history).*

---

## 3. The `yfinance` Cache & AWS Rate Limits

**The Problem**:
AWS Lambda runs a read-only file system (except for `/tmp`). The `yfinance` library requires writable directories to store TZ and Session Cookies. If it cannot store cookies, EVERY scrape hit to Yahoo Finance registers as a fresh, cookieless scrape. Yahoo Finance will instantly **RATE-LIMIT (HTTP 429)** and IP-ban the Lambda function.

**The Fix**:
You MUST explicitly tell `yfinance` to cache in the `/tmp` folder before making any calls. This codebase handles it in `_fetch_yfinance`:

```python
import yfinance as yf
import os

cache_dir = '/tmp/yfinance'
os.makedirs(cache_dir, exist_ok=True)
yf.set_tz_cache_location(cache_dir)
# Now you can safely call yfinance
```

---

## 4. API Gateway Proxy 500 Errors (The `NaN` JSON Bug)

**The Problem**:
If the Lambda succeeds (returns HTTP 200 locally) but the live API Gateway URL returns `{ "message": "Internal Server Error" }` (HTTP 500), the Gateway Proxy Parser is failing. 

AWS HTTP API Gateway (Payload format 2.0) requires exact JSON strings if you use proxy mappings. **Python's `json.dumps()` converts missing pandas floats directly into unquoted `NaN` variables.** Standard JSON parsers (including Vercel Edge functions and API Gateway) will violently crash trying to parse `NaN`.

**The Fix**:
Any float returned by pandas or math must be sanitized before `json.dumps`. We use `_clean_nans` in `lambda_handler.py` to recursively override `NaN` and `Infinity` into standard `null` (`None`). 

---

## 5. Finnhub Real-time Spot Override

**The Problem**:
If `yfinance` is completely blocked on AWS, the waterfall falls back to Polygon's free API. Polygon's free aggregates strictly delay data by a full day, causing the dashboard to show 1-day stale prices (e.g. returning yesterday's close instead of today's spot).

**The Fix**:
Do not accept stale data. In `fetch_spy_with_fallback`, we use `Finnhub` (`_fetch_finnhub_quote`) as a bulletproof spot-price override hook. Finnhub reliably returns second-by-second live SPY data. We overwrite the stale Polygon `current` spot variable dynamically before it reaches the dashboard.

---

## 6. AWS IAM API Gateway Integration

If you hook up a new API Gateway to the Lambda function and it receives an instant `500` error with NO logs generated in CloudWatch, it is because AWS Lambda restricts execution via Resource-based policies. 

You must manually grant the API Gateway permission to trigger the Lambda:
```bash
aws lambda add-permission \
    --function-name financial-telegram-report \
    --statement-id AllowMyAPIGateway \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:us-east-1:<ACCOUNT_ID>:<API_ID>/*/*"
```
Ensure `.env.local` in `dashboard/` points to the fully authorized gateway.

---

### Final Checklist for AI Modifications
1. Did you double check that your Python maths don't leak `NaN` into the API Gateway payload?
2. Are your `yfinance` fetches caching to `/tmp`?
3. Did you bundle your lambda deployment strictly for `manylinux2014_x86_64`?
4. Are you gracefully falling back to Finnhub for live spot overrides if Polygon returns 1-day stale data? 
5. Does the API Gateway have IAM Resource Policy invoke permissions?

By adhering to this guide, you guarantee a 100% stable, rate-limit resilient, and bug-free production environment.
