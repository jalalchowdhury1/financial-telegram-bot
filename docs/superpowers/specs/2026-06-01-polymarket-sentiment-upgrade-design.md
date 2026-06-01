# Design: Polymarket → "Market Sentiment" upgrade

**Date:** 2026-06-01
**Status:** Approved ("do the best you can") — implementing autonomously
**Goal:** Turn the Polymarket section from a wall of 1%/100% longshots into a curated
"what is the crowd betting on" board — meaningful-probability markets across diverse topics.

## Problem
`fetch_polymarket_trending` orders individual **markets** by 24h volume, which surfaces
longshot sub-markets of multi-candidate events (live proof: #1 by 24h vol = "Will Cape
Verde win the World Cup?" @ 0.05%). No filter for genuine uncertainty → walls of 1% and
100%. Display is thin (just % + a mis-formatted volume: "29844k" should be "$30M").

## Approach: Consensus board (chosen)
Show only markets with **genuine uncertainty**, one per topic, most-traded first, each with
a 30-day momentum arrow. Two implementations stay in sync (resilience pattern): the Lambda
fetcher (primary) and the dashboard JS fallback.

### Smart selection (the core fix)
Both `bot/fetchers.py: fetch_polymarket_trending` (Python) and
`dashboard/app/api/polymarket/route.js: fallbackPoly` (JS):
1. Fetch a **broad pool** (~300–500 markets) ordered by **weekly volume** (`volume1wk`,
   recent interest, less churny than 24h), `active=true&closed=false`.
2. Keep only markets that are: **binary Yes/No** (`outcomes == ["Yes","No"]`), **non-sports**
   (existing tag + keyword filter), probability (first outcome price) in **[0.08, 0.92]**,
   and volume ≥ ~$50k.
3. **De-dupe by event** (`events[0].ticker`) — keep the highest-volume market per event.
4. **Tag a topic** by keyword (Crypto 🪙 / Geopolitics 🌍 / Politics 🏛️ / Tech 🤖 /
   Economy 📉 / World 🌐), **cap 2 per topic** for diversity.
5. Rank survivors by volume → take **top 8**.

### Richer per-market data contract
Each bet: `{ name, odds (0–1), volume ($), change (oneMonthPriceChange, ±frac | null),
topic (str), endDate (ISO | null), eventSlug (str | null) }`. Backward-compatible:
`name/odds/volume` (existing) preserved so `MarketModal` keeps working.

### Frontend (`PolymarketTable.js`)
Each row: **topic emoji** + question + clear **%** with color bar (existing `getOddsColor`)
+ **▲▼ Npt** 30-day momentum (only when |change| ≥ 0.02) + **volume in $M/$k** (fixes the
"29844k" bug) + a muted **"Nd"** resolves-in (from `endDate`). Keep click→modal. Header →
"📊 Market Sentiment · what the crowd's betting on". Badge → "REAL-TIME · TOP 8".

## Plumbing & safety
- Lambda `/api/polymarket` (`lambda_handler.handle_http_api`) already returns `{bets,...}`;
  it passes the richer dicts through `_ok()` (never-throw, NaN-sanitized).
- Dashboard route stays never-throw (Lambda → JS fallback → last-known-good → empty).
- bot/fetchers change deploys via the gated **Deploy to AWS Lambda** PR + smoke test.
- `fetch_polymarket_trending` stays fully wrapped (returns `[]` on any failure).

## Testing
- Python: extend `tests/test_polymarket_fetcher.py` — filters extremes (drop <8%/>92%),
  filters sports (tag + the World-Cup-style keyword), de-dupes by event, caps per topic,
  parses the richer fields, returns ≤8, never raises on bad/empty API data.
- JS: `dashboard/__tests__/PolymarketTable.test.js` still passes with the richer rows.
- **Live verification:** run the new fetcher against the live Gamma API and eyeball the 8
  selected markets (must be diverse + meaningful, no 1%/100%); check the PR's Vercel preview.

## Out of scope (v2)
Multi-candidate "favorites" (e.g., "Dem 2028: Newsom 22%"); a fresher momentum window via
the CLOB price-history endpoint (only 30-day change is exposed in the markets API).

## Success criteria
The section shows ~8 diverse, meaningful-probability markets (no 1%/100% walls, no sports,
no duplicate-event candidates), with topic + momentum + correctly-formatted volume; the
endpoint stays never-throw; daily health-check's `endpoint_polymarket` stays green.
