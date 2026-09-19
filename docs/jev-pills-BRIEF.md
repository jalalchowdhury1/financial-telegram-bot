# Jev regime pills — build brief (2026-09-19)

Read this whole file before writing code. It is the contract. Build ONLY the chunk you
were assigned; do not touch other chunks' files, do not `git commit`, `git push`, or deploy.
Repo conventions live in `AGENTS.md` (§ Dashboard). Tests: from `dashboard/`,
`npx jest --testPathPattern <name>`; the full suite (`npx jest`) must stay green (382 tests
today) and `npm run build` must list every new `/api/*` route as `λ`, never `○`.

## Goal (owner's words)

Three additions to the Vercel dashboard (https://financial-telegram-bot-beryl.vercel.app/),
built so that flipping ONE env var (`JEV_PILLS=off`) restores exactly today's site:

1. **Breadth + hedging-cost pills** — new `/api/breadth` (RSP/SPY, IWM/SPY, XLK/XLU, HYG/LQD
   from Polygon with keyless CNBC backup) plus the vol payload feed two pills.
2. **Conflict / divergence pill** — deterministic pairs (sentiment vs price, 2s10s vs 3m10y,
   credit vs equities, breadth vs index); the pill names the pair(s).
3. **"What changed since yesterday"** — today's verdicts vs yesterday's saved snapshot →
   `none | softening | hardening | mixed`; every day's verdicts are logged so they can be
   scored after 30 days (a scorer script), and a one-line chip can join the 4:15 AM
   Telegram brief (separate, gated step — NOT part of the dashboard chunks).

Plus the base pills: **regime**, **recession**, and each pill shows whether Jev or the rule
decided it.

## Backup ladder (non-negotiable — "always keep a backup to what works today")

- Rules ALWAYS produce a verdict for every pill (`ruleVerdicts`). Jev may only override a
  rule when its confidence `p >= 0.6`. Jev off / no key / timeout / HTTP error / unknown
  verdict / p < 0.6 ⇒ the rule verdict, marked `by: "rule"`.
- `JEV_PILLS` env var: `off` ⇒ `/api/jev-pills` returns `{ enabled: false }` and the
  component renders nothing (today's UI). `rules` ⇒ no Jev call, rules only. `on` or unset
  ⇒ Jev with rule backup.
- Every route wraps its whole body in `serve()` from `lib/store.js` (never throws, always
  HTTP 200 JSON, last-known-good in /tmp). Opening line of every GET handler:
  `request.headers.get('user-agent');` (see `app/api/vol/route.js`).
- Storage for the daily log is best-effort (Upstash REST via `KV_REST_API_URL` /
  `KV_REST_API_TOKEN`); missing KV ⇒ `since` is null and `logged: false`, nothing breaks.
- Git: tag `pre-jev-pills` = the site before this work.

## Privacy

State text sent to Jev (TypeSafe, a third party) may contain ONLY public market numbers:
FRED series, CNN fear/greed, CBOE vol, SPY stats, ETF ratios, AAII bull-bear spread.
NEVER include: the Google Sheet custom indicators (`NotSoBoring`, `FrontRunner`), anything
from `/api/rubber-band` or `/api/sheets` other than `AAIIDiff`, any account, name, or
holding. Keys only via `process.env` (repo is public).

## Jev API (real, verified 19 Sep 2026, jev-1.13.0)

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY>      Content-Type: application/json
{ "state": "<text>", "model": "jev-latest",
  "questions": { "<k>": { "type": "choice", "instructions": "<question>",
                          "criteria": { "<verdict>": "<description>", ... } } , ... } }
→ { "answers": { "<k>": { "choice": "<verdict>", "confidence": 0.83,
                          "probabilities": { ... } } }, "usage": {...} }
```
Several questions ride in ONE call (one call per dashboard refresh). Timeout 6 s. Jev is
proven at classifying a described state; it is NOT trusted for dates or deltas, so the
"since yesterday" direction is computed by code, never asked of Jev.

## Data contract — the `data` object every pure function receives

All numbers may be `null`/`undefined` (source down); functions must cope and say `n/a`.

```js
{
  spy:      { price, chgPct, ma200Pct, high52Pct, rsi },            // /api/spy: current, dailyChange.pct, ma200.pct, week52High.pct, rsi
  fg:       { score, rating, prevWeek, prevMonth },                  // /api/fear-greed: score, rating, previousWeek, previousMonth
  vol:      { spy: { iv, ivRank1y, ivPctile1y, rv21, vrp }, qqq: {…} },   // /api/vol tickers[] by ticker
  fred:     { yieldCurve, sahmRule, claims, creditSpread, realYields, copperGold, sentiment, nfci },
            // /api/fred: yieldCurve.value (T10Y2Y, %), indicators.sahmRule.value, indicators.claims.value (thousands),
            // indicators.creditSpread.value (BBB OAS %), indicators.realYields.value, indicators.copperGold.value,
            // indicators.sentiment.value (UMCSENT), checklist.nfci.value
  t10y3m:   number|null,                                             // FRED T10Y3M (%); null when unavailable
  breadth:  { rspSpy: { ratio, chg20Pct, chg60Pct, vs50dPct }, iwmSpy: {…}, xlkXlu: {…}, hygLqd: {…} },   // /api/breadth pairs
  aaiiDiff: number|null                                              // /api/sheets AAIIDiff "24.50%" → 24.5 (bull minus bear, pts)
}
```

## Pills and rules (exact thresholds — do not invent others)

| pill | verdicts (ordered mild → severe) | rule |
|---|---|---|
| regime | `risk-on`, `neutral`, `risk-off` | score = (ma200Pct>0 ? +1 : -1) + (fg.score>=50 ? +1 : fg.score<30 ? -1 : 0) + (hygLqd.chg20Pct>0 ? +1 : hygLqd.chg20Pct<-1 ? -1 : 0). score>=2 ⇒ risk-on; score<=-1 ⇒ risk-off; else neutral. Missing inputs count 0. |
| recession | `low`, `rising`, `high` | high if sahmRule>=0.5, or (yieldCurve<0 and claims>=260). rising if sahmRule>=0.2, or yieldCurve<0, or claims>=260, or nfci>0. else low. |
| breadth | `broad`, `narrow`, `rolling-over` | broad if rspSpy.chg20Pct>0 and iwmSpy.chg20Pct>0. rolling-over if rspSpy.chg20Pct<-1.5 and rspSpy.vs50dPct<0. else narrow. No breadth data ⇒ `narrow` with reason "no breadth data". |
| hedging | `cheap`, `fair`, `expensive` | uses vol.spy: cheap if ivPctile1y<20 and vrp<6. expensive if ivPctile1y>70 or vrp>10. else fair. Missing ⇒ fair. |
| conflict | `aligned`, `mild-divergence`, `major-divergence` | pairs = `conflictPairs(data)`; 0 ⇒ aligned, 1 ⇒ mild, ≥2 ⇒ major. |

`conflictPairs(data)` returns `[{ pair, detail }]` for each that fires:
- `sentiment vs price`: (fg.score<35 and ma200Pct>3) or (fg.score>65 and ma200Pct<-3).
- `2s10s vs 3m10y`: yieldCurve and t10y3m have opposite signs (both non-null).
- `credit vs equities`: ma200Pct>0 and hygLqd.chg20Pct<-1.5.
- `breadth vs index`: high52Pct>-2 and rspSpy.chg20Pct<-1.5.

Each rule verdict carries a one-sentence `reason` quoting the numbers it used.

`diffSinceYesterday(today, yesterday)` (both `{ pillKey: { verdict } }`): list every pill whose
verdict changed `{ pill, from, to }`; rank by the ordered verdict list above; all moves toward
severe ⇒ `hardening`, all toward mild ⇒ `softening`, both ⇒ `mixed`, none ⇒ `none`.
`yesterday == null` ⇒ `{ direction: 'none', changed: [], noBaseline: true }`.

`mergeVerdicts(rule, jev, floor = 0.6)` ⇒ per pill `{ verdict, p, by: 'jev'|'rule', reason }`;
`jev` may be null. Jev verdict must be one of the pill's verdicts.

`buildState(data)` ⇒ compact plain text, one numbered fact per line, e.g.
`1. SPY 761.69, +6.3% vs 200-day avg, -2.1% from 52-week high, RSI 48.5, today -0.12%` …
`n/a` for missing numbers. Nothing else goes in.

`JEV_QUESTIONS` ⇒ `{ regime: { instructions, criteria }, recession…, breadth…, hedging…, conflict… }`
with criteria = the verdict descriptions (write them as a careful analyst would).

## Chunks

### A — `dashboard/lib/jev.js` + `dashboard/lib/jevBrief.js` (+ tests)
- `lib/jev.js`: `JEV_TIMEOUT_MS = 6000`, `JEV_P_FLOOR = 0.6`, `defaultPoster(url, body, timeoutMs)`
  (fetch + AbortController), `judgeMany(state, questions, post = defaultPoster)` →
  `{ k: { verdict, p } }` (only answers whose `choice` is in that question's criteria) or `null`
  on no `TYPESAFE_API_KEY` / any error (log one short line, never throw).
- `lib/jevBrief.js`: pure, no fetch: `PILLS`, `JEV_QUESTIONS`, `buildState`, `ruleVerdicts`,
  `conflictPairs`, `mergeVerdicts`, `diffSinceYesterday`, `toData(raw)` where `raw =
  { spy, fg, vol, fred, breadth, sheets, t10y3m }` are the raw route payloads → the data contract.
- Tests `dashboard/lib/__tests__/jev.test.js`, `jevBrief.test.js`: reply shape parsed; no key ⇒
  null + no request; every error ⇒ null; unknown verdict dropped; each rule threshold at its
  edge; every conflict pair; merge floor; diff directions incl. no baseline; `buildState` never
  contains "NotSoBoring"/"FrontRunner"; `toData` on real-shaped payloads (see contract comments).

### B — `dashboard/app/api/breadth/route.js` + `dashboard/lib/breadth.js` (+ tests)
- `lib/breadth.js` pure: `PAIRS = { rspSpy: ['RSP','SPY'], iwmSpy: ['IWM','SPY'], xlkXlu: ['XLK','XLU'], hygLqd: ['HYG','LQD'] }`;
  `ratioSeries(histA, histB)` (align on common dates, `{date, ratio}` ascending);
  `pairStats(series)` → `{ ratio, chg20Pct, chg60Pct, vs50dPct, asOf }` (chg over 20/60 trading
  rows; vs50dPct = ratio vs mean of last 50 rows, %; null when too short).
- Route: `request.headers.get('user-agent');` first; `faultsFrom`/`gate` (`breadth_polygon`,
  `breadth_cnbc`); per ticker cascade `polygonDaily(t, process.env.POLYGON_KEY, { years: 1 })`
  → `cnbcHistory(t, { range: '1Y' })` (see `lib/sources.js` for both; each returns `{ history: [{date, price}] }`);
  `serve('breadth', produce, { fallback: { pairs: {}, _meta: { source: 'none', hasErrors: true, messages: ['no breadth source'] } } })`;
  payload `{ updated_at, pairs: { rspSpy: {...}, ... }, _meta: { source, hasErrors, messages } }`.
  A pair with one missing leg is omitted, the rest still ship. `export const fetchCache = 'default-cache';`
- Tests `dashboard/lib/__tests__/breadth.test.js`: alignment with a missing date, 20/60/50
  math on a synthetic series, too-short ⇒ nulls.

### C — `dashboard/components/JevPills.js` (+ test)
- `'use client'`; props `{ data, loading }` where `data` is the `/api/jev-pills` payload (below)
  or null. Renders NOTHING when `!data || data.enabled === false`. While loading renders a
  one-row `Skeleton` (see `components/Skeleton.js`).
- One row of pills in a `card` (match `MarketPulse.js` styling/classes): Regime, Recession,
  Breadth, Hedges, Conflict. Colour by severity: mild green, middle amber, severe red
  (`badge-green` / `badge-yellow` / `badge-red` — check `app/globals.css` for the existing names).
  Each pill: label, verdict, tiny source tag `Jev 0.82` or `rule`, `title=` reason. The conflict
  pill's title lists the pairs. Trailing chip: `Since yesterday: none` / `softening (breadth narrow → broad)`
  / `no baseline yet`. Footer: `as of <asOf>` and `mode: rules` when `data.mode === 'rules'`.
- Test `dashboard/components/__tests__/JevPills.test.js`: renders nothing when disabled; renders
  5 pills + chip from a sample payload; shows `rule` tag; loading shows skeleton.

### D — `dashboard/app/api/jev-pills/route.js` + `dashboard/lib/jevLog.js` (after A and B)
- Payload:
```js
{ enabled: true, mode: 'on'|'rules', asOf, state,
  pills: { regime: { verdict, p, by, reason }, recession, breadth, hedging, conflict },
  conflictPairs: [{ pair, detail }],
  since: { date, direction, changed: [{ pill, from, to }], noBaseline } | null,
  _meta: { jev: 'ok'|'off'|'rules'|'error: …', logged: bool, sources: {…} } }
```
- `JEV_PILLS=off` ⇒ `{ enabled: false }` immediately. Assemble raw data by fetching the sibling
  routes on the same origin (`new URL(request.url).origin`), each with a 10 s timeout and
  null on failure: `/api/spy`, `/api/fear-greed`, `/api/vol`, `/api/fred`, `/api/breadth`,
  `/api/sheets`, plus `T10Y3M` via `fredObservations('T10Y3M', process.env.FRED_API_KEY, { limit: 5 })`.
- `lib/jevLog.js`: Upstash REST like `health-hub/lib/kv.js` (GET `/get/<key>`, POST `/set/<key>`
  body JSON, `/lrange`). Keys `ftb:jev:<YYYY-MM-DD>` (verdicts + state hash + spy price),
  `ftb:jev:days` (LPUSH date once per day, NX guard `ftb:jev:seen:<date>` with 40-day TTL).
  `yesterday()` = newest logged day before today. All failures ⇒ null / false, never throw.
- Wrap in `serve('jev-pills', produce, { maxStaleMs: 6*3600e3, fallback: { enabled: true, mode: 'rules', pills: null, _meta: { jev: 'error: no data' } } })`.
- `dashboard/scripts/jev-score.mjs`: reads the logged days from KV, fetches SPY + RSP daily closes
  (Polygon), scores regime vs SPY 5-trading-day return sign, breadth vs RSP/SPY 5-day sign,
  prints a hit-rate table per pill and per `by`. Manual run: `node scripts/jev-score.mjs`.

### E — page wiring + Telegram line (owner's reviewer does this, not a chunk)
`app/page.js` fetches `/api/jev-pills` alongside the others and renders `<JevPills>` right
under `<MarketPulse>`; Lambda gets an optional `JEV_PILLS_URL` step.

## Done means
- `npx jest` green, `npm run build` shows `λ` for `/api/breadth` and `/api/jev-pills`.
- No file outside your chunk touched. No secrets in code. Report which files you wrote and
  the exact test command + last lines of its output.
