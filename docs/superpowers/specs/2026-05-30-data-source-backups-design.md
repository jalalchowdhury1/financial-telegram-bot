# Data-Source Backups — Design

**Date:** 2026-05-30
**Status:** Approved (approach), implementing
**Goal:** No dashboard data ever goes blank. Every feed gets independent fallback sources beneath its current primary, and fallbacks must return the **same numbers** as the primary (verified live before merge).

## Principle

Keep current primaries (AWS Lambda for markets, FRED for economics) so normal operation produces today's exact numbers. Add direct-source fallback chains in the Next.js layer that activate only when the primary fails. Apply 30-min Data Cache throughout.

## Current redundancy (audit)

| Route | Primary | Has fallback? |
|---|---|---|
| `/api/spy` | Lambda (`/api/spy`) | 🔴 none |
| `/api/spy-daily-move` | Lambda | 🔴 none (Lambda currently throwing `'SPY_DAILY_MOVE'`) |
| `/api/market-extra` | Lambda | 🔴 none (crude `cl` + `mortgagePayment` null) |
| `/api/polymarket` | Lambda | 🔴 none |
| `/api/fred` | FRED + 30-min cache | 🟡 one provider |
| `/api/fear-greed` | CNN→RapidAPI→YahooVIX→FREDVIX→cache | 🟢 deep |
| `/api/sheets` | Live→cache→altURL→FREDproxy→static | 🟢 deep |

## Response shapes to preserve (from live Lambda)

- **spy:** `{ current, dailyChange:{value,pct}, ma200:{value,pct}, week52High:{value,pct}, rsi, return3y, chartHistory:[{date,price,ma50,ma200}], _meta }`
- **spy-daily-move:** `{ value, source }` (value is a string like "+0.25%")
- **market-extra:** `{ fx:{usdcad,usdinr,usdbdt,inrbdt,cadinr,cadbdt,dxy}, commodities:{cl,gc,btc}, rates:{tnx,t2y,mortgageRate}, realEstate:{rentIndex,mortgagePayment,atnhpi}, _meta }` — each metric `{ current, dailyChange:{value,pct}, history:[{date,price}], lastDate }`
- **polymarket:** `{ bets:[{name,odds,volume}], timestamp }`

## Fallback chains

Shared helper `lib/sources.js` (each with retry + 30-min cache):
- `yahooChart(ticker, range)` → `{current, history:[{date,price}], dailyChange}` — SPY, FX (`CAD=X`,`INR=X`), `GC=F`, `CL=F`, `BTC-USD`, `DX-Y.NYB`, `^TNX`, `^IRX`/`^FVX`
- `stooqDaily(symbol)` → daily CSV (spy.us, etc.)
- `coingeckoSimple('bitcoin')` → BTC USD (keyless)
- `coinbaseSpot('BTC-USD')` → BTC (keyless)
- `erApiRate(base, quote)` → ExchangeRate-API (existing key) 
- `frankfurterRate(base, quote)` → ECB rates (keyless)
- `dbnomicsFred(seriesId)` → FRED mirror, keyless, **identical numbers**

Per route:
- **spy:** Lambda → Yahoo (`SPY`, 5y → current/MA200/52wHigh/RSI(9)/3yReturn/chartHistory via `finance.js`) → Stooq
- **spy-daily-move:** Lambda → derive from Yahoo SPY last-2 closes (fixes the current breakage)
- **market-extra:** Lambda → per-metric direct fallback (FX: ER-API→Frankfurter→Yahoo; gold/crude/dxy: Yahoo→Stooq; btc: CoinGecko→Coinbase→Yahoo; rates: FRED `DGS10`/`DGS2`; mortgage: FRED `MORTGAGE30US`). Fill only the metrics the Lambda left null/missing; never overwrite a good Lambda value.
- **fred:** FRED → DBnomics per failed series → cache
- **polymarket:** Lambda → Polymarket Gamma API (`gamma-api.polymarket.com`) top-10 by volume
- **fear-greed / sheets:** already redundant; modernize caching only (low priority)

## Equality verification (the hard requirement)

Before merging any route, deploy to preview and run a comparison: fetch the **production Lambda value** (primary) and the **fallback value** at the same moment, compute the delta. Acceptance:
- Live prices (SPY, FX, metals, BTC, rates, DXY): within **0.5%** (sources quote at slightly different instants).
- FRED economics via DBnomics: **exact** match.
A fallback is not merged until its delta is within tolerance. A temporary `?debug=compare` mode on each route returns primary vs each fallback side-by-side for this check.

## Rollout

Branch `feat/data-source-backups`. Order: spy → spy-daily-move → polymarket → market-extra → fred(DBnomics). Each verified on preview against production before merge. Production untouched until verified. Reversible via Vercel rollback.

## Testing

- Unit tests for `lib/sources.js` parsers (Yahoo/Stooq/DBnomics/CoinGecko shape → normalized `{current,history}`), using captured sample payloads.
- Preview equality checks logged in the PR/commit messages.

## Non-goals

- Not modifying the AWS Lambda itself.
- Not changing the dashboard UI (response shapes preserved exactly).

## Final implementation notes (after live verification on Vercel)

**Two-layer defense-in-depth.** The Lambda (in AWS, clean egress IP) already has 4–5 source chains per metric (yfinance→Polygon→Finnhub→Stooq→FRED for prices; FRED for rates; Gamma for Polymarket). It stays PRIMARY. The dashboard-layer fallbacks below engage only if the whole Lambda is unreachable.

**Measured source reliability from Vercel egress** (8+ hits each, repeated):
- Reliable: ER-API, Frankfurter, Coinbase, Gamma, FRED (when cached, not bursty). Yahoo is reliable for *single small* queries but 429s on *large/bursty* historical queries. Stooq now requires an API key (dead). CoinGecko rate-limits.
- Therefore: price-history feeds (SPY, Gold, exact FX) use **Polygon** (`POLYGON_KEY`) + **Finnhub** (`FINNHUB_KEY`) — server-friendly, the same vendors the Lambda uses. Keyless fallbacks: FX→ER-API, DXY→computed from the ER-API basket, crude+rates→FRED, BTC→Coinbase.

**Verified equality (preview vs live Lambda):** SPY price/52w/RSI exact, MA200 0.4%; USD/CAD exact (Polygon); USD/BDT + rates + DXY exact; Gold/BTC within ~0.5%; crude filled (Lambda's was null).

**Env vars added (Vercel, Production+Preview):** `POLYGON_KEY`, `FINNHUB_KEY`. Never hardcoded (public repo).

**Lambda bugs the dashboard layer now masks** (fix at the Lambda for a permanent cure — needs AWS redeploy):
- `fetch_spy_daily_move()` uses `URLS['SPY_DAILY_MOVE']` which is missing from `bot/config.py` → always errors. Dashboard now serves it via Finnhub.
- Crude (`cl`) has no Stooq layer → nulls when 3 sources miss. Dashboard fills from FRED `DCOILWTICO`.

**Dropped:** DBnomics (per-series dataset path unresolved; FRED is reliable via cache+retry). Stooq (now key-gated).
