# Alpha Vantage API Setup

## Get Your Free API Key

1. **Sign up** at: https://www.alphavantage.co/support/#api-key
   - Takes less than 20 seconds
   - Enter your email and organization name
   - Free tier includes 25 API requests per day (perfect for daily reports)

2. **Copy your API key** from the confirmation page

## Add to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add:
   - **Name:** `ALPHA_VANTAGE_API_KEY`
   - **Value:** Your API key from Alpha Vantage

## Test Locally (Optional)

If you want to test locally, add to your `.env` file:

```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## Why Alpha Vantage?

- ✅ **Reliable:** Official API, not web scraping
- ✅ **Free:** 25 requests/day (we only use 1 per report)
- ✅ **Works in CI/CD:** No IP blocking like Yahoo Finance
- ✅ **Stable:** Doesn't break when websites change
- ✅ **Fast:** Typical response time < 1 second

## Free Tier Limits

- **25 API requests per day**
- **5 API requests per minute**

Our daily report uses **1 request per day**, so we're well within limits.
