# Graceful staleness for FRED indicators

**Date:** 2026-06-21
**Status:** Approved (design), pending implementation plan
**Author:** Claude (with owner)

## Problem

Dashboard FRED tiles show **N/A** whenever a series' newest data point is older than
its freshness deadline. For monthly macro series this happens *every month* during the
normal FRED reporting gap: the series is dated the 1st, FRED publishes the next reading
weeks late, so the newest point legitimately ages past the deadline for ~a week before
the next print. Concretely, on 2026-06-21:

- `M2SL` (M2 Money Supply), `DGORDER` (Durable Goods), `PSAVERT` (Savings Rate) all have
  their latest FRED observation at **2026-04-01** = **81 days old** vs an **80-day** deadline.
  FRED has not published the May reading yet (those release ~June 23–27). Verified against
  FRED's own keyless CSVs; our fetch is healthy (`/api/fred` `_meta`: "Loaded 17/17 series",
  `hasErrors: false`). **The data lag is entirely on FRED's side, not ours.**

This is the same class of bug fixed for `UMCSENT` (deadline 80→95) on 2026-06-20. We want a
general solution so monthly lag never shows N/A, while genuine breaks are still surfaced.

## Goals

1. **Never show N/A for merely-stale data.** If we have a last-known value, show it.
2. **Make staleness visible**, not hidden: a stale value renders in **orange with a 🕐 clock**
   and an "as of <date>" note.
3. **Alert only on genuine overdue-ness**: warn (Telegram health alert) when a series is stale
   **more than 3 days past its deadline**, so the owner can fix it. Normal lag never alerts.
4. **N/A is reserved for true unavailability** — no value at all (fetch returned nothing).
5. Apply consistently across **all FRED tiles** (Bull Checklist + Economic Indicators grid +
   Yield Curve + Profit Margin).

Non-goals (this spec): the Google-Sheet last-known-good fallback for total FRED outages. That
is a separate fast-follow (the daily-scraped `financial-dashboard-history` sheet), explicitly
deferred per owner. It only matters when the live fetch returns *nothing*; merely-stale data is
already in hand.

## Design

### 1. `lib/freshness.js` — stop discarding stale values; expose staleness magnitude

`withFreshness(value, asOfDate, deadlineDays, now)` changes so a stale value is **kept**, not
nulled. Value becomes `null` only when genuinely unavailable (undefined/null/NaN input).

New return shape adds `staleDays` (whole days past the deadline; 0 when fresh/unavailable):

```js
{ value, asOf, stale, unavailable, staleDays }
//  value:      kept when stale; null ONLY when unavailable
//  stale:      had data, older than deadline
//  unavailable:no data at all (fetch returned nothing)
//  staleDays:  max(0, floor(ageDays - deadlineDays)) when stale, else 0
```

`freshnessNote(metric)` returns a `tone` instead of a bare `amber` boolean:

```js
{ suffix, tone }   // tone ∈ 'fresh' | 'stale' | 'unavailable'
```

- `stale`       → suffix `" • 🕐 As of <date> (stale)"`, tone `'stale'`
- `unavailable` → suffix `" • ⚠ Unavailable — source busy, try again shortly"`, tone `'unavailable'`
- `fresh`       → suffix `" • As of <date>"`, tone `'fresh'`

(Keep a back-compat `amber` getter — `tone !== 'fresh'` — so any un-migrated caller still works.)

### 2. `lib/constants.js` — deadlines reflect real publish cadence

Bump the three late-month monthlies to **95** (matching the UMCSENT fix), so normal lag is not
flagged stale and the 3-day alert baseline is meaningful:

```
M2SL: 80 → 95     # H.6 release ~4th Tue of following month
DGORDER: 80 → 95  # Advance Durable Goods ~25th of following month
PSAVERT: 80 → 95  # Personal Income & Outlays ~last business day of following month
```

Other series unchanged: mid-month monthlies (`RSXFS`/`HOUST`/`INDPRO`, ~15th release, max age
~77) stay 80 — graceful staleness absorbs any rare slip; `UNRATE` 80; `JTSJOL` 110; quarterly
200; weekly 14; daily 7. `UMCSENT` already 95.

### 3. UI — orange + 🕐 for stale, N/A only for unavailable

A value is shown whenever present; styling keys off `tone`:

- **`components/BullChecklist.js`**: row icon becomes `unavailable ? '⚪' : stale ? '🕐' : (bullish ? '✅' : '🔴')`.
  Value text: number when present (already), `'N/A'` when null. Value color: `--orange` when stale,
  `--yellow` when unavailable (kept), normal positive/negative otherwise. Score still counts each
  item's last-known `bullish` (unchanged — per owner Q1).
- **`components/EconomicIndicatorGrid.js`**: value color `--orange` when `tone==='stale'`, and
  prefix the **value text** with `🕐 `; `--yellow`/N/A when unavailable. Each tile already renders
  `value ?? 'N/A'`, so a kept stale value now shows automatically.
- **`app/page.js` Yield Curve & Profit Margin tiles**: today the stale note lives in the
  `current == null` branch (only reached because stale was nulled). Restructure into three states:
  `unavailable` (`current == null`) → N/A block (kept); `stale` (present + `stale`) → value block
  styled `--orange` + 🕐 + "as of" note; `fresh` → normal green/red value block.

### 4. `app/globals.css` — add orange

```
--orange: #f59e0b;
--orange-bg: rgba(245, 158, 11, 0.12);
```

(`--yellow` #eab308 stays for the unavailable/busy state, visually distinct from stale-orange.)

### 5. `scripts/health_check.py` — alert on overdue, not on lag; sweep checklist too

Replace the current "null/unavailable ⇒ warn" rule (`check_indicators_na`) with:

- **unavailable** (value null) → warn ("indicator N/A — fetch failed").
- **staleDays > 3** → warn ("indicator overdue — stale N days past deadline; fix the source").
- otherwise → ok.
- `KNOWN_DISCONTINUED` still suppresses (e.g. a deliberately frozen series).

The sweep must inspect **both** `indicators` and `checklist` from `/api/fred` (today it only
inspects `indicators`, so checklist staleness is invisible to the alerter). Top cards
(`yieldCurve`/`profitMargin`) are daily/quarterly and out of scope for the 3-day alert; their
outright outages are still caught by the existing endpoint-health probe.

This kills the recurring false WARN (normal lag is no longer stale, and even a brief slip is
shown gracefully and not alerted until 3 days overdue), while a genuine multi-day break still
pages the owner.

## Trade-offs / decisions

- **Discontinued series show old orange data + alert until handled.** Per owner Q1/Q2 choice
  (alert after 3 days; no hard display cutoff), a truly dead series keeps showing its last value
  in orange and warns daily until it's replaced (as LEI→Copper/Gold) or allowlisted in
  `KNOWN_DISCONTINUED`. Accepted.
- **Score counts stale items' last-known status.** A 1-month-old macro reading is still
  decision-relevant; this also makes the X/8 score consistent with the now-visible numbers.

## Testing

- `lib/__tests__/freshness.test.js`: stale value is **kept** (not nulled); `unavailable` still
  nulls; `staleDays` math; `freshnessNote` tone + 🕐 suffix; regression that `M2SL`/`DGORDER`/
  `PSAVERT` are not stale at their realistic worst-case age (day before next print, ~85 days).
- Component tests (jest): BullChecklist renders 🕐 + orange value for a stale item, ⚪ + N/A for
  unavailable, and the score counts a stale-but-bullish item.
- `scripts/health_check.py` tests: `staleDays > 3` ⇒ warn; `staleDays ≤ 3` ⇒ ok; unavailable ⇒
  warn; checklist items are swept.
- Full suites green: `cd dashboard && npm test` (jest) + `pytest`; `npm run build`.

## Files touched

- `dashboard/lib/freshness.js`
- `dashboard/lib/constants.js`
- `dashboard/components/BullChecklist.js`
- `dashboard/components/EconomicIndicatorGrid.js`
- `dashboard/app/page.js`
- `dashboard/app/globals.css`
- `scripts/health_check.py`
- tests for the above
- `AGENTS.md` (document the new staleness semantics + the 95-day deadlines + the 3-day alert rule)

## Rollout

Branch + PR (branch protection on `main`); CI gates (`backend-tests`, `dashboard-tests`) + Vercel
preview. Verify on the live dashboard after merge that the three tiles render their April values
(orange/🕐 only if still past deadline; otherwise normal) and that the health check no longer
WARNs on them. Sheet-fallback fast-follow tracked separately.
