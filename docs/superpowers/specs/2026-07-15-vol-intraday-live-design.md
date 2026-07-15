# Intraday live IV for the Volatility Metrics card — design

**Date:** 2026-07-15 · **Status:** approved by owner · **Scope:** dashboard only (no Lambda change)

## Problem

The 🌡️ Volatility Metrics card (`/api/vol` + `VolMetricsTable.js`) shows IV / IV rank /
IV percentile / RV 21d / VRP for SPY, QQQ, TQQQ, SQQQ, UVXY — but every source in its
cascade (CBOE daily-history CSV, CNBC daily bars, FRED, Yahoo) is a **daily-close series**.
During the trading day the newest point is yesterday's close, so the card reads
"As of <yesterday>" all day. The owner bases trading judgement on these numbers and wants
them fresh during market hours.

## Decision

Add a **live-spot override tier** (Option A of three considered): fetch live `.VIX` /
`.VXN` / `.VVIX` quotes from CNBC's keyless quote endpoint (`cnbcQuotes`, already used by
copper/gold and datacenter-proven from Vercel) and, **only when the quote is strictly newer
than the last EOD close**, use the live level as "current" in the metric math. This mirrors
the repo's existing Finnhub-overrides-stale-Polygon pattern for SPY.

Rejected alternatives: (B) promoting CNBC daily bars above CBOE — demotes the canonical
source and relies on unverified partial-bar behavior; (C) a scheduled twice-a-day refresh —
extra infrastructure for strictly worse freshness (serve() runs the producer per request;
freshness is set by per-fetch `revalidate`, so a 5-min revalidate gives ≤5-min-old data).

## Design

### 1. `lib/sources.js` — additive
`cnbcQuotes()` also returns the full `last_time` string as `lastTime`
(e.g. `2026-07-15T13:42:31.000-0400`) alongside the existing date-only `asOf`.
Backward-compatible; copper/gold callers unaffected.

### 2. `app/api/vol/route.js`
One extra fetch alongside the history fetches:
`cnbcQuotes(['.VIX', '.VXN', '.VVIX'], { revalidate: 300 })`
— gated behind the existing **`vol_cnbc`** fault gate (per-SOURCE semantics: tripping it
disables daily bars AND the live quote), wrapped in try/catch so any failure appends a
`_meta.messages` note and the card falls back to today's EOD behavior. When an override is
applied, `_meta.source` shows it per index: `VIX:cboe+live`.

### 3. `lib/vol.js`
`buildVolMetrics(indexSeries, etfCloses, liveQuotes?)` — new optional third param, keyed by
index name: `{ VIX: { value, date, lastTime }, … }`. Per ticker, when the live quote's
**date is strictly newer** than the last EOD point (and the value is finite and > 0):

- **IV** = mult × live level (scaled, unchanged rule)
- **Rank / %ile** = UNSCALED live level vs. the unchanged 1y window of UNSCALED EOD closes
- **RV 21d** = untouched — EOD closes only, never a partial day
- **VRP** = live IV − EOD RV
- row gains `live: true` and `asOf` = the quote's date

Quote date ≤ last close (evenings, weekends, holidays) → no override, `live: false`,
identical to today. Payload top level gains `live_at` (full timestamp of the newest applied
quote, else null).

### 4. `components/VolMetricsTable.js`
When any row is live, the footnote shows
"As of **2026-07-15, 1:42 PM ET · intraday**" with a small green dot; otherwise it stays
exactly "As of <date>". No layout changes.

### 5. Error handling
Never-throw preserved. Live-quote failure is a degradation note, not an error —
`hasErrors` logic unchanged. No new fault gate names.

## Reliability requirements (owner: "I will base my judgement on it")

- The override must never *replace* good EOD math with garbage: guard on finite, positive
  quote values and a parseable, strictly-newer date; otherwise skip the override entirely.
- Rank/percentile must keep using the UNSCALED index level against the UNSCALED window
  (a constant multiplier changes neither — the existing invariant).
- Verification before merge: live CNBC quote values cross-checked against an independent
  source (CBOE's own delayed quotes / Google Finance) during market hours; full Jest suite;
  `npm run build` must list `/api/vol` as `λ` (dynamic), never `○`.
- Post-deploy: curl the production endpoint during market hours and confirm `live: true`,
  sane values, and correct `_meta.source` labels; confirm `?_fail=vol_cnbc` on prod serves
  the EOD fallback with `live: false`.

## Testing

- **Jest units (`lib/vol.js`)**: override applied when quote date > last EOD date; skipped
  when equal/older/missing/NaN/≤0; rank/%ile computed against the EOD window with the live
  current; RV untouched; VRP = live IV − EOD RV; `live_at` set/null correctly.
- **Jest units (`lib/sources.js`)**: `lastTime` passthrough.
- **Route-level**: `?_fail=vol_cnbc` path (no live override, EOD values still served);
  `_meta.source` `+live` labeling.
- **Build**: route remains dynamic (λ).
- **Live**: manual cross-check + post-deploy prod curl as above.

## Out of scope

- Intraday RV, intraday history charts, Telegram delivery of the vol table.
- Any Lambda/backend change.
- AGENTS.md §3 vol section gets updated as part of the implementation PR.
