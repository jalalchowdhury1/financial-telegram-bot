# Jev pills v3 — backup chains for every pill input

Owner rule: every number a pill reads must have 3+ sources or a last-known-good,
like the established cards. Audit on 2026-09-19 (Claude) found four gaps, all on
the FRED side. Everything else already cascades:

| Input (pill) | Route | Chain today | Gap |
|---|---|---|---|
| SPY vs 200d, F&G (regime, conflict) | /api/spy, /api/fear-greed | 3+ tiers + last-good | none |
| RSP/SPY, IWM/SPY, HYG/LQD (breadth, regime, conflict) | /api/breadth | polygon → cnbc → last-good | none |
| IV percentile, VRP (hedging) | /api/vol | cboe → cnbc → fred → yahoo + last-good | none |
| Yield curve 2s10s (recession, conflict) | /api/fred `yieldCurve` | fred → treasury → fredcsv → /tmp → Sheet | none |
| **Jobless claims** (recession) | /api/fred `indicators.claims` | primary FRED only. `horsemen.claims` IS repaired (fredcsv) but `indicators.claims` is not recomputed | **A** |
| **Sahm rule** (recession) | /api/fred `indicators.sahmRule` | primary UNRATE only. `horsemen.unemployment` IS repaired (bls → fredcsv) but Sahm is not recomputed | **B** |
| **NFCI** (recession) | /api/fred `checklist.nfci` | primary FRED only, no repair tier | **C** |
| **3m10y** (conflict pair "2s10s vs 3m10y") | /api/jev-pills `fetchT10y3m` | FRED API only, no CSV tier, no last-good | **D** |

## Design (do NOT edit app/api/fred/route.js — it is 679 lines of production-critical cascade code)

All repairs live in the pills layer. New pure module `dashboard/lib/jevInputs.js`
+ glue in `dashboard/app/api/jev-pills/route.js`. Reuse existing helpers:

- `resolveHorseman(sources, faults, now)` from `lib/horsemen.js` — tries sources in
  order, skips a source when `faults.has('hm_<name>')`, when it throws, returns < 2
  points, or its newest point is older than `freshnessDays`. Returns
  `{ history, current, currentDate, source, tried }` (history ASCENDING).
- `parseFredGraphCsv(csv)` from `lib/horsemen.js` → ascending `[{date,value}]`.
- `fredGraphCsv(seriesId)` from `lib/sources.js` — keyless CSV
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`. Verified live today:
  T10Y3M 2026-09-18 = 0.87, NFCI 2026-09-11 = -0.560.
- `fredObservations(seriesId, apiKey, {limit})` from `lib/sources.js` — returns
  DESCENDING `[{date,value}]` (newest first). Reverse it before handing to
  resolveHorseman.
- `saveLastGood(key, data)` / `loadLastGood(key, maxAgeMs)` from `lib/store.js`.
  Rule from `serve()`: when `faults.size > 0` never WRITE last-good; when
  `faults.has('lastgood')` never READ it.

### lib/jevInputs.js (pure, unit-tested)

```js
export const FRESH = { T10Y3M: 7, NFCI: 14, ICSA: 14, UNRATE: 45 }; // days

/** 4-week average of weekly ICSA in THOUSANDS, from an ascending history. null if < 4 points. */
export function claims4wkFromHistory(history)  // e.g. [..., 231000] → 231 (÷1000, not rounded)

/** Sahm: mean of the latest 3 months minus the min of the latest 12 months, from an ascending monthly UNRATE history. null if < 12 points. */
export function sahmFromHistory(history)

/**
 * Generic cascade for one pill input.
 * sources: [{ name, freshnessDays, fetch }] — passed straight to resolveHorseman.
 * lastGoodKey: e.g. 'jev-t10y3m'. maxStaleMs default 7 days.
 * Returns { value, asOf, source, tried } where source ∈ source names | 'lastgood' | null.
 * Never throws. Writes last-good on any live success (unless faults.size > 0).
 */
export async function resolvePillInput({ sources, faults, now, lastGoodKey, maxStaleMs })
```

### route glue: `repairPillInputs(raw, { fredKey, faults, now })` in jev-pills/route.js

Runs AFTER the sibling fetches, BEFORE `toData`. Mutates `raw` and returns
`inputSources` for `_meta`.

1. **t10y3m (D)** — always resolved here (replaces `fetchT10y3m`):
   sources `[ fred (if fredKey): fredObservations('T10Y3M', key, {limit: 30}) reversed,
              fredcsv: parseFredGraphCsv(await fredGraphCsv('T10Y3M')) ]`, freshness 7,
   lastGoodKey `jev-t10y3m`. Set `raw.t10y3m = value` (number|null — keep the
   existing contract, `toData` reads `raw.t10y3m` as a number).
2. **nfci (C)** — only when `raw.fred?.checklist?.nfci?.value` is not a finite number:
   sources `[ fredcsv NFCI ]`, freshness 14, lastGoodKey `jev-nfci`. On success set
   `raw.fred.checklist.nfci = { value, asOf, stale: false, unavailable: false, source }`
   (create `raw.fred`/`checklist` objects if missing).
3. **claims (A)** — only when `raw.fred?.indicators?.claims?.value` is not finite:
   first try DERIVE from `raw.fred?.horsemen?.claims?.history` via
   `claims4wkFromHistory` (source `'derived:horsemen'`, asOf = last history date);
   else cascade `[ fredcsv ICSA ]` → `claims4wkFromHistory(history)`; lastGoodKey
   `jev-claims`. Set `raw.fred.indicators.claims = { value, asOf, ..., source }`.
4. **sahm (B)** — same shape: derive from `raw.fred?.horsemen?.unemployment?.history`
   via `sahmFromHistory` (source `'derived:horsemen'`), else `[ fredcsv UNRATE ]`,
   lastGoodKey `jev-sahm`. Set `raw.fred.indicators.sahmRule = {...}`.

Nothing here may throw; a total failure leaves the field null exactly as today.

`_meta.inputSources = { t10y3m, nfci, claims, sahm }` where each value is
`'fred-route'` (the sibling already had it), a source name, `'derived:horsemen'`,
`'lastgood'`, or `null`. Add it in `assemblePills` (lib/jevPills.js) from a new
opt `inputSources` (default `{}`) so tests can assert it.

### Popup (components/JevPillModal.js)

In the Sources section, when `_meta.inputSources` has any entry that is not
`'fred-route'`, append one line: `Backups in use: nfci fredcsv · claims derived:horsemen`
(only the non-default ones). No other UI change.

## Tests (jest, `cd dashboard && npx jest`) — must stay 100% green (597 today)

- `lib/__tests__/jevInputs.test.js`: claims4wkFromHistory (4+ points, <4 → null,
  ÷1000), sahmFromHistory (12 points, <12 → null, known numbers), resolvePillInput
  (first source wins; throwing source skipped; stale source skipped; `hm_<name>`
  fault skips; last-good read when all fail; `lastgood` fault blocks the read;
  no write when faults present). Mock `lib/store` with jest.mock.
- `lib/__tests__/jevPillsRoute.test.js`: assemblePills carries `inputSources`
  into `_meta`.
- Route glue is exercised through a small exported helper: export
  `repairPillInputs` from the route file and test it with mocked
  `lib/sources` (fredGraphCsv / fredObservations) and `lib/store`.
- Component test: "Backups in use" line renders when inputSources has a non-default
  entry, absent otherwise.

## Definition of done

- `npx jest` green, `npm run build` shows `λ /api/jev-pills`.
- Fault proofs to be run by Claude on production after deploy:
  `?_fail=fred` → claims & sahm `derived:horsemen`, nfci `fredcsv`, t10y3m `fredcsv`, verdicts still computed;
  `?_fail=fred,hm_fredcsv,hm_bls,hm_treasury` → `lastgood` entries.
- AGENTS.md: one bullet under the Jev pills section naming the four repairs and the
  fault names.

## v3.1 (same day) — what the live proof found and the fix

Prod `?_fail=fred` after the v3 deploy returned **n/a for NFCI, claims and Sahm**
(`inputSources` null) and `?_fail=…hm_fredcsv…` left t10y3m null. Three root causes:

1. **`fredgraph.csv` is a phantom on Vercel** (AGENTS.md already said so: works in local
   dev, hangs from a server). v3 leaned on it for every input. Local success was the trap.
2. **The sibling `/api/fred` takes ~18 s under a FRED outage** (two 8 s CSV tries per
   series) and the pills route stops waiting at 10 s → `raw.fred` was null, so even the
   `horsemen` tier had nothing to derive from.
3. **Last-good was never seeded.** It was written only when a repair succeeded, i.e. only
   during an outage — on a healthy day the tier stayed empty. And /tmp is per instance.

Fix (this doc's contract, amended):

- t10y3m: fred API → **treasury** (`parseTreasurySpreadCsv(csv, '3 mo', '10 yr')`,
  origin publisher, already prod-proven for 2s10s) → fredcsv → last-good.
- nfci: **sheet** (`fetchSheetLkg().checklist.nfci`) → fredcsv → last-good.
  Chicago Fed's own CSV was probed and rejected: the file ends 2026-04-24 at every URL.
- claims / sahm: horsemen → **sheet** (`indicators.claims` / `indicators.sahmRule`) →
  fredcsv → last-good.
- `resolvePillInput` gains `read()` single-value sources, an injectable `store`, and
  error text in `tried` (surfaced as `_meta.inputTried`).
- `lib/jevStore.js`: /tmp + Upstash KV (`ftb:jev:lg:<key>`), 14-day max age, seeded from
  the sibling on every healthy, fault-free call.
- `fredGraphCsv`: 1 try × 5 s (was 2 × 8 s) so the sibling's outage path fits in 10 s.
- `export const maxDuration = 30` on the pills route.
- Fault names: `hm_<tier>` shared with the sibling; `jev_lastgood` for the pills' last-good;
  the sibling's `lastgood`/`sheetlkg` are not applied to the pills, so a proof can empty the
  sibling and still exercise the Sheet and KV tiers.

- `FRESH.UNRATE` 45 → 75 days: the series is dated by observation month, so its newest
  point is ~65 days old just before the next print; at 45 every Sahm tier read as stale for
  the last three weeks of each month (prod proof P2 showed `sheet:stale(2026-08-01)`).

### v3.1 proofs (production, 2026-09-19, commit 5ec297e + freshness fix)

| Call | inputSources | time |
|---|---|---|
| baseline | t10y3m `fred`, rest `fred-route`; popup shows no "Backups in use" line | 1.3 s |
| `?_fail=fred` | sibling answers in ~6 s (was ~18 s), serves its own cache → all `fred-route` | 6.6 s |
| `?_fail=hm_fred` | t10y3m `treasury` (`fred:off`, `treasury:ok`), value 0.87 = FRED | 1.1 s |
| `?_fail=fred,lastgood,sheetlkg,hm_bls,hm_treasury,hm_fredcsv` (sibling emptied) | nfci `sheet`, claims `sheet`, sahm `sheet`; popup "Backups in use: nfci sheet · claims sheet · sahm sheet" | 2.0 s |
| … + `hm_sheet,hm_fred` | all four `lastgood` from KV (`lastgood:ok(<savedAt>)`), verdicts unchanged | 2.0 s |
| … + `jev_lastgood` | all four null → rows show n/a, verdicts still computed | 2.2 s |
