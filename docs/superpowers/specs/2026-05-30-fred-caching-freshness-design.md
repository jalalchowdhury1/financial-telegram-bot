# FRED Caching & Per-Metric Freshness — Design

**Date:** 2026-05-30
**Status:** Approved (design), pending implementation
**Scope:** `dashboard/app/api/fred/route.js`, `dashboard/lib/fetcher.js`, `dashboard/lib/constants.js`, and the components that render FRED numbers (`EconomicIndicatorGrid.js`, `BullChecklist.js`, `app/page.js`).

## Problem (verified live)

The deployed dashboard shows ~12 of 18 FRED-driven metrics as **N/A** or **0**. The live `/api/fred` `_meta.messages` proves the cause: **`429 Too Many Requests`** on most series. Which series fail is random per load — the fingerprint of rate-limiting, not bad series IDs or a missing key.

Two root causes in code:

1. **`export const dynamic = 'force-dynamic'`** in the route. Per Next.js docs this forces every `fetch()` to `{ cache: 'no-store', revalidate: 0 }` — overriding the `revalidate` the code already passes. So **caching never happens**; every page load (every visitor, every 5-min auto-refresh) fires 19 live FRED requests, blowing past FRED's ~120 req/min + shared-IP throttle.
2. **Dead fallback cache.** `if (!hasErrors) saveCache(...)` only saves on a perfect 18/18 load, which almost never happens, so the stale-cache safety net is always empty.

Plus: the FRED **API key is leaked** in `_meta.messages` (echoed in the failing URLs).

## Goals

- Stop the 429s by caching FRED data and refreshing at most every **30 minutes**.
- **Never display stale numbers.** Each metric has its own freshness deadline; past it, show N/A + a subtle "couldn't refresh" state rather than an old value.
- Always show **when each number is from** (hover tooltip per metric) and a global "Data as of HH:MM".
- Back off politely on 429 (stagger + retry).
- Stop leaking the API key.

## Design

### 1. Caching (the core fix)

`force-dynamic` makes the route override every `fetch()` to `no-store`, so per-fetch `revalidate` is ignored. Rather than remove `force-dynamic` (which could let the route be prerendered at build time and cache a 500 when the key isn't present at build), wrap the FRED fetching in **`unstable_cache(..., { revalidate: 1800 })`**. `unstable_cache` stores the *function result* in Vercel's Data Cache with its own 30-min TTL, independent of `force-dynamic`/`fetchCache`. The handler still runs per request (for assembly + freshness stamping); only the upstream FRED calls are served from the 30-min cache.

- Result: each series is cached fleet-wide for 30 min. FRED is hit at most ~once per series per 30 min, regardless of visitor count or the 5-min client poll. No new infra. The P/E scrapes (multpl.com / Yahoo) are wrapped the same way.

### 2. Polite back-off

- `fetchSeries` wraps the request in a retry helper: on `429`, wait `[400ms, 900ms, 1800ms]` and retry (max 3 attempts) before giving up.
- Keep batched requests (2 per batch) with a ~400ms stagger between batches to smooth cold-start bursts.

### 3. Per-metric freshness (`asOf` + deadlines)

`constants.js` gains a `FRED_FRESHNESS` map of each metric → max acceptable age (days), keyed by update cadence:

| Cadence | Metrics | Deadline (days) |
|---|---|---|
| Daily | yieldCurve, realYields, creditSpread | 5 |
| Weekly | claims | 10 |
| Monthly | sentiment, lei, sahm, nfci, m2, retail, housing, indpro, jolts, durable, savings, peRatio | 50 |
| Quarterly | profitMargin | 130 |

(Monthly/quarterly are generous because the government reports those weeks late — e.g. current-quarter GDP doesn't exist yet. The deadline only trips on a genuinely broken/delayed feed.)

Each metric in the API response gains:
- `asOf`: ISO date of the latest underlying observation used.
- `stale: true` when `today − asOf > deadline`.

When a series **fails to fetch** (429 after retries) → value is `null`, `stale: false`, `unavailable: true`.
When a series is **too old** → value is forced to `null`, `stale: true` (so the UI shows N/A, never the old number).

Computed metrics use the oldest contributing observation for `asOf` (e.g. profitMargin uses min(corpProfits date, gdp date); sahm uses unrate date; claims uses the 4-week window's latest date).

### 4. Response envelope

Add `_meta.fetchedAt` (ISO timestamp of assembly). Keep `_meta.messages` but **mask the API key** in any URL it contains (`api_key=***`).

### 5. Fallback cache (made safe)

- Save **per-series last-good** values (value + observation date) to the `/tmp` cache on every run, for the series that succeeded (no longer gated on a perfect load).
- When a series fails this run, fill from last-good **only if within that metric's freshness deadline**; otherwise N/A. This honors "remember a copy, but never show numbers that are too old."

### 6. Frontend (subtle, reuses existing tooltip system)

The app already uses `className="tooltip-trigger" data-tooltip="…"` (CSS hover tooltips). We append a freshness line to each metric's existing tooltip:
- Fresh: `… • As of <Mon D, YYYY>`
- Stale/unavailable: value renders as `N/A` in a muted **amber** (`var(--yellow)`); tooltip reads `… • ⚠ Last data <date> — couldn't refresh`.

Global: add a small "Data as of HH:MM" line (from `_meta.fetchedAt`) next to the existing "Updated …" badge in the header, so the user always sees the server data time (distinct from the browser poll time).

Touched components: `EconomicIndicatorGrid.js` (6 indicators + peRatio), `BullChecklist.js` (8 checklist items), `app/page.js` (yieldCurve + profitMargin cards + header timestamp).

### 7. Testing

- Extract `isStale(asOfDate, deadlineDays, now)` as a pure helper and unit-test it (fresh, exactly-at-deadline, past-deadline, missing date) with jest.
- `npm run build` must pass; existing jest suite must stay green.

## Non-goals

- No new database/KV (Vercel Data Cache is sufficient).
- The "MKTS unavailable: 3 metrics" box (separate AWS Lambda) is out of scope for this change.
