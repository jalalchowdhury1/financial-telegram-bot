# Sheet last-known-good fallback (case #3)

**Date:** 2026-06-21
**Status:** Approved (owner: "take over and get it done"), pending implementation
**Spans two repos:** `financial-telegram-bot` (dashboard reader) + `financial-dashboard-history` (helper-tab writer)

## Problem / goal

The dashboard's `/api/fred` never shows N/A for FRED reporting lag (graceful staleness) or for an
outage on a **warm** instance (`serve()` serves `/tmp` last-known-good for 7 days). The one
remaining gap (**case #3**): a total FRED outage that lands on a **cold** serverless instance (no
`/tmp` copy, no Redis) → `{ error: 'FRED temporarily unavailable' }` → dashboard-wide N/A.

Close it for **zero cost** using the existing `financial-dashboard-history` Google Sheet as a
durable last-resort store: a self-describing **helper tab** the scraper writes each run, and the
dashboard reads (public gviz, no auth, no new secret) only when everything else has failed.

**Fallback order:** live FRED → `/tmp` last-known-good → **Google Sheet helper tab** → error.

## Why this design (vs the raw sheet)

The existing `Sheet1` is position-mapped flat scalars (brittle) with no `asOf`/status (lossy). The
helper tab fixes both: **self-describing key→value rows** (order-independent) carrying value +
`asOf` + status (+ bullish/label for checklist). The scraper builds it from the **rich, already
null-recovered `fred` JSON** it fetches each run — not from `Sheet1`'s flattened row.

## Helper tab: `dashboard_lkg`

Two columns, `key | value`, one field per row (header row `key,value`). Public-readable via
`https://docs.google.com/spreadsheets/d/1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc/gviz/tq?tqx=out:csv&sheet=dashboard_lkg`.

Keys written (a metric is **omitted entirely** if its value is null/`N/A` — "cut out the N/A"):
```
updated_at                       <ISO timestamp of the snapshot>
peRatio                          <number>
yieldCurve.current               <number>
yieldCurve.asOf                  <YYYY-MM-DD>
profitMargin.current             <number>
profitMargin.asOf                <YYYY-MM-DD>
indicators.<k>.value             for k in sahmRule,sentiment,claims,creditSpread,realYields,copperGold
indicators.<k>.asOf
indicators.<k>.status
checklist.<k>.value              for k in nfci,m2,retail,housing,indpro,jolts,durable,savings
checklist.<k>.asOf
checklist.<k>.status
checklist.<k>.bullish            "true" | "false"
checklist.<k>.label
```
All values are simple scalars (numbers, `YYYY-MM-DD`, short words, `true/false`, labels w/o commas)
→ clean CSV, no escaping pitfalls. Fidelity = **values + asOf only; no chart history** (charts
render empty in the rare fallback — acceptable for a last resort).

## Writer — `financial-dashboard-history/scraper.py`

Add `build_lkg_pairs(fred)` (pure) → list of `[key, value]` pairs from the rich `fred` dict,
dropping any metric whose value is `None`/`"N/A"` (reuse `cur_or_hist` for yieldCurve/profitMargin;
`clean_numeric_string` is NOT applied — store raw numeric values the dashboard consumes directly).
Add `write_helper_tab(doc, pairs)`: get-or-create the `dashboard_lkg` worksheet, `clear()`, then
`update('A1', [['key','value'], *pairs])`. Call it in `main()` after the rich fetch, wrapped so a
helper-tab failure **never** breaks the core append (non-fatal, logged). The writer uses the same
service-account auth already in place.

## Reader — `financial-telegram-bot`

New pure-ish module `dashboard/lib/sheetLkg.js`:
- `parseLkgCsv(text)` → `{ [key]: rawString }` map (tiny CSV parse: 2 columns, skip header).
- `reconstructFred(map, now)` → a payload shaped exactly like `/api/fred` minus history:
  - `yieldCurve {current, asOf, stale:true, history:[]}`, `profitMargin {…, history:[]}`,
    `peRatio`, `peRatioAsOf`, `recessions:[]`.
  - `indicators.<k> = {value, asOf, status, stale:true, unavailable:false, staleDays:4}` (staleDays
    >3 so the health check flags the outage).
  - `checklist.<k> = {value, asOf, status, bullish(bool), label, stale:true, unavailable:false, staleDays:4}`.
  - `_meta {source:'Google Sheet (last-known-good <updated_at>)', stale:true, hasErrors:true,
    fetchedAt: now, messages:['live FRED + /tmp cache unavailable; served Google-Sheet last-known-good'],
    loadedCount:0}`.
  - Returns `null` if the map is empty / has no usable metrics.
  - Typing by key suffix: `.value`/`.current`/`peRatio` → Number (drop if NaN); `.bullish` → boolean;
    else string.
- `fetchSheetLkg(now)` → never-throws: `fetch` the gviz CSV (short timeout) → `parseLkgCsv` →
  `reconstructFred`; returns `null` on any error/empty.

`dashboard/lib/store.js` `serve()`: add an optional `opts.lastResort` (async, never-throws). In
BOTH the "produced not-good" branch and the `catch` branch, AFTER a `/tmp` last-good miss and
BEFORE returning `fallback`, call `lastResort()`; if it returns a payload, serve it via
`withStale({data, savedAt})`. In fault-test mode it's skipped unless explicitly allowed.

`dashboard/app/api/fred/route.js`: pass `lastResort: () => fetchSheetLkg(new Date())`. Add the gviz
URL to `lib/constants.js` (`EXTERNAL_URLS.SHEET_LKG`).

## Health-check interaction

When served from the sheet, every metric has a value (not N/A) but `staleDays:4` and `_meta.stale`/
`hasErrors:true`. The existing `check_endpoint` flags `_meta.hasErrors` → WARN, and
`check_indicators_na` flags `staleDays>3` → WARN. So a real total outage still alerts the owner —
it just shows last-known values (orange/🕐) instead of N/A.

## Testing (thorough)

- **Reader unit (jest)** — `parseLkgCsv` (quoting, blank lines), `reconstructFred` (typing, N/A
  omission, empty→null, checklist bullish bool, _meta shape, staleDays:4).
- **store.js (jest)** — `serve()` lastResort: produce throws + no `/tmp` + lastResort returns payload
  → serves it stale; lastResort returns null → fallback error; warm `/tmp` present → lastResort NOT
  called (precedence).
- **Live integration (script, not committed)** — fetch the real `dashboard_lkg` gviz CSV →
  reconstruct → assert ≥12 metrics present, correct types, no N/A.
- **Writer (python)** — `build_lkg_pairs` drops N/A, includes asOf/status/bullish, shape correct.
- **End-to-end** — populate the tab (MCP for first snapshot; scraper thereafter), then drive the
  dashboard reader against the live tab; confirm a simulated case #3 serves sheet values, not N/A.
- Full suites green (jest, pytest, build) + a code-review pass.

## Rollout / safety

Checkpoint tag `checkpoint-pre-sheet-fallback` (pushed). Dashboard ships via PR (branch protection)
→ CI → merge. Scraper ships to its `main` (solo repo). AGENTS.md updated in **both** repos. The
reader is strictly additive and never-throws, so it cannot regress the existing route behavior.
