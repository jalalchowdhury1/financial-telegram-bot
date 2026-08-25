# Fresh Print Marks — design

**Date:** 2026-08-25
**Status:** IMPLEMENTED 2026-08-25 (treatment B · Breath). See the "What changed during implementation" section at the end — four defects were found by end-to-end verification that the design did not anticipate.
**Mockup:** https://claude.ai/code/artifact/1c5feac5-2829-4595-a064-d1d2c4514f52

## The problem

Every number on the dashboard is presented as equally current. But a number changes for
one of two reasons: **a price ticked**, or **a new print landed**. Only the second is news,
and today nothing on the page distinguishes them. The owner wants to scroll the dashboard
and be able to tell, at a glance and without being shouted at, which numbers are new — then
double-click one to see what it was before.

## The insight that drives the design

Do not diff values and mark whatever moved. Mark by **release cadence**: was this number
even *allowed* to move today?

Measured over 167 days of the `financial-dashboard-history` sheet, changes per day per column:

| Band | Columns | Reading |
|---|---|---|
| 0.00–0.19 | ATNHPI, Sentiment, Rent, Housing, JOLTS, Savings, Profit Margin, Sahm, M2, IndPro, Retail, Durable, NFCI, AAII, Mortgage rate, Mortgage payment, Claims | Changes only when an agency publishes. A change **is** the news. |
| 0.35–0.67 | Credit Spread, Copper/Gold, Yield Curve, Real Yields, P/E | Moves most days. "Changed" says nothing; an outsized move does. |
| 0.78–1.00 | VIX ×3, DXY, Gold, BTC, Crude, every FX pair | Changes daily by definition. Never mark. |

The classification is **hardcoded** (explicit and reviewable), with the measured rates recorded
alongside each entry as the justification. It is not derived at runtime — a series frozen by a
dead upstream would otherwise look "slow" and then fire loudly on recovery.

Resulting frequency, measured over 166 day-pairs: **52% of days nothing lights, 18% one mark,
11% two, 14% three-to-five, 5% six or more.** The silence is the point.

## Two marks, one colour

| Mark | Glyph | Fires when |
|---|---|---|
| **New print** | `●` | A slow-cadence metric's value differs from the last snapshot of a prior calendar day. |
| **Outsized move** | `⌃` / `⌄` | A daily metric's move exceeds 2σ of its own trailing daily moves. Directional. Requires **≥20 prior daily moves**; below that the metric is simply unmarked. |

Both use **cyan `#22d3ee`** — the only hue the dashboard's semantic palette has not claimed
(green/red = bullish/bearish, yellow = unavailable, orange = stale, indigo = chart lines).
Colour means *noteworthy*; **shape** says which kind. A mark is never green or red, so it can
never be misread as "good" or "bad".

Backtested fire rates for the 2σ tier: yield curve 6% of days, P/E 5%, real yields 3%,
credit spread 1%, copper/gold 10%. Expected ≈0.25 move-marks per day across all five.

## Treatment: B · Breath

- A 6px cyan dot sits after the number, plus a 1px hairline underline that scales in.
- On entering the viewport (`IntersectionObserver`, threshold 0.9, `unobserve` after firing)
  the dot breathes **three times** — 1.9s period, opacity 0.32↔1, scale 1↔1.28 — then rests
  at opacity 0.72.
- Firing on scroll-into-view is deliberate: scanning is exactly when the owner is looking.
  Nothing pulses forever, so a page with four marks never strobes.
- `@media (prefers-reduced-motion: reduce)` drops all animation; the dot and hairline remain.

Rejected: A · Whisper (too easy to miss while scrolling), C · Beacon (never stops — four of
them reads as an alarm), D · Sweep (elegant but the mark vanishes once the animation ends).

## The reveal

Double-click the value, or single-click/tap the dot. `Enter`/`Space` when focused. `Esc` or an
outside click closes. One open at a time.

For a monthly series, "yesterday's value" is almost always identical and therefore useless.
The popover shows the **change event** instead:

```
BEFORE THIS PRINT
14.83%                    ▲ +0.09
held 28 days · since May 28
▁▂▃▅▆  (sparkline, last N distinct prints, endpoint emphasised)
LAST 5 PRINTS
```

For a move mark the eyebrow reads `YESTERDAY` and the sub-line reads
`Moved more than 2σ of its own daily range`.

### Positioning — non-negotiable

The popover **must** be portalled to `document.body` with `position: fixed`.

`.card` sets `backdrop-filter: blur(12px)`, which creates a stacking context. An
absolutely-positioned child is trapped inside it and the *next* card paints straight over it —
verified in the mockup, and `globals.css:953` already carries a comment from a previous
run-in with this same class of bug.

Placement rules, all verified in the mockup:
- Prefer below the trigger; **flip above** when the viewport runs out below.
- Clamp horizontally to the viewport with 12px padding.
- Arrow x-offset follows the trigger's centre (`--ax`).
- Reposition on scroll and resize; close once the trigger scrolls out of view.
- Never overlap the trigger.

## Data architecture

### `/api/history` (new route)

Follows the existing conventions exactly:
- Opens with `request.headers.get('user-agent')` so Next does not statically prerender it
  (AGENTS.md §3 — this bit `/api/vol` on 2026-07-05; verify `λ` not `○` in the build output).
- Entire body wrapped in `lib/store.js: serve()` — never throws, always 200 + valid JSON.
- Reads the **public CSV export** of `Sheet1`
  (`.../1lA-_yjLMc3qDTt9sogSPQrCohNULIk5wwJYfb5wIHfc/export?format=csv&gid=0`), no credentials.
  Confirmed publicly readable. Reuses `parseCsvLine` from `lib/sheetLkg.js`.
- Returns a **compact digest**, not the sheet: one entry per tracked metric —
  `{ kind, prev, changedOn, heldFrom, heldDays, runs[] }`. ~2KB against a ~100KB CSV.
  `runs` differs by kind and the popover labels it accordingly: for a print mark it is the last
  8 **distinct** values (`LAST N PRINTS`), for a move mark the last 8 **daily snapshots**
  (`LAST 7 SESSIONS`).
- Cached like every other route; a failure degrades to no marks at all, never to wrong marks.

The client fetches it alongside the existing six calls in `page.js: fetchAll`.

### Metrics not in the sheet

`spEps`, `horsemen.unemployment`, `horsemen.bankruptcies` and `horsemen.claims` are **not**
`Sheet1` columns, but `/api/fred` already returns full `history[]` arrays for them. Their
previous-print payload is derived client-side from data already on the page — no new fetch.
`yieldCurve` and `profitMargin` likewise carry native history.

`NotSoBoring` and `FrontRunner` are deliberately **out of scope** (owner decision, 2026-08-25):
they are not in the sheet, and `trading-algorithm-` already alerts on their flips.

## Guards against false marks

Three real hazards, all present in the live history:

1. **Unit changes.** On 2026-03-18 the scraper switched conventions — claims `212000 → 212`,
   housing `1487000 → 1487`, JOLTS `6542000 → 6946`. A naive diff lights three false marks.
   **Reject any transition whose ratio is within 10% of 1000× or 1/1000.**
2. **Zero as a missing sentinel.** Older rows write a bare `0`; on 2026-05-08 seven metrics went
   `0 → real value`. **Treat a run of exact zeros as absent data**, not a value.
3. **N/A ↔ value transitions.** A metric recovering from `N/A` is a plumbing event, not
   economic news. **Never mark a transition where either side is blank or `N/A`.**

**Baseline selection.** The baseline is the last snapshot row whose Date is strictly earlier
than today — robust whether or not today's row has been written yet (duplicate-date rows are
normal, since the scraper runs twice daily).

"Today" **must be computed in `America/New_York`**, not in the server's locale. The route runs
on Vercel in UTC, and the sheet's Date column is stamped by a scraper on an ET cron; between
8pm and midnight ET the UTC date is already tomorrow, so a naive `new Date()` would resolve
"today" to the wrong day and silently select the wrong baseline every evening. Use
`Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York' })` to get a `YYYY-MM-DD` string
and compare lexically against the sheet's dates.

**Equality.** Sheet values are already rounded to display precision, so a plain inequality is
the right test — compared with an epsilon of `1e-9` for float safety only. A relative-delta
threshold was considered and rejected: it would suppress genuinely small prints (NFCI moves
±0.01) while the three guards above already cover the pathological cases.

## Components

- **`lib/marks.js`** — pure, no React. `classifyMetric(key)`, `previousPrint(series, today)`,
  `isOutsizedMove(series)`, plus the three guards. All logic that can be unit-tested lives here.
- **`components/Delta.js`** (`'use client'`) — wraps a rendered value. Takes
  `{ value, mark, prev, heldDays, heldFrom, runs, kind }`; renders the number, the glyph, the
  hairline, and owns the portalled popover. Knows nothing about where its data came from.
- **`components/MarkProvider.js`** — context holding the `/api/history` digest plus the
  fred-derived entries, so `Delta` call sites stay one prop wide.

Existing components gain `<Delta>` wrappers only — no restructuring.

**Print marks (`●`)**

| Component | Metrics |
|---|---|
| `EconomicIndicatorGrid` | sahmRule, sentiment, claims |
| `BullChecklist` | all 8 |
| `ExtraMarketsGrid` | ZRI (rent), MTGPMT, MORT30, **ATNHPI** |
| `CustomIndicatorBar` | AAII only — not VIX, not NotSoBoring/FrontRunner |
| Profit Margin card, S&P 500 EPS card | the hero value |
| `FourHorsemen` stat chips | unemployment, bankruptcies, claims |

**Move marks (`⌃`/`⌄`)**

| Component | Metrics |
|---|---|
| `EconomicIndicatorGrid` | creditSpread, realYields, copperGold, peRatio |
| Yield Curve card | the hero value |

Everything else on the page is explicitly unmarked: SPY, Fear & Greed, the volatility
table, the Polymarket board, and every FX / commodity / crypto row in `ExtraMarketsGrid`.

## Header affordance

A small chip beside the existing "Updated …" badge, counting both kinds:
`● 3 new prints · 1 outsized move`. Each clause is omitted when its count is zero, and the whole
chip is omitted when both are — so a quiet day adds nothing to the page. Clicking it scrolls to
the next marked value.

## Error handling

- `/api/history` unreachable, or the sheet malformed → the digest is empty → **no marks render**
  and every number displays exactly as it does today. The feature fails invisible.
- A metric present in the digest but absent from the live payload, or vice versa → skipped.
- The popover never blocks: if `runs` has fewer than two points the sparkline is omitted and the
  rest still renders.

## Testing

Jest, in `dashboard/lib/__tests__/marks.test.js` unless noted:
1. Unit-jump guard — `212000 → 212` produces **no** mark; `212 → 214` does.
2. Zero-sentinel guard — `0 → 14.44` produces no mark; `14.44 → 14.83` does.
3. N/A guard — `'' → 6.49` and `6.49 → 'N/A'` produce no mark.
4. Baseline selection — with duplicate rows for today and yesterday, the baseline is
   yesterday's **last** row; with no row for today, it is still yesterday's last row.
5. `heldDays` — counts back through identical values to the first occurrence.
6. 2σ classifier — a flat series never fires; a 3σ spike does; <20 points never fires.
7. Classification — every FX pair, VIX, DXY, gold, BTC and crude return `none`.
8. `Delta.test.js` — renders the glyph only when marked; double-click opens exactly one
   popover; `Esc` closes it; the popover mounts on `document.body`, not inside the card.
9. Route test — a malformed CSV yields an empty digest and a 200.

`npm test` and `npm run build` must pass, and `/api/history` must appear as `λ` in the build
output. CI (`ci.yml`) is the merge gate.

## Out of scope

- Adding `NotSoBoring` / `FrontRunner` to the history sheet (owner decision).
- Threshold-crossing marks (yield curve crossing zero, Sahm crossing 0.50). Considered and
  rejected for v1 — it would give a mark a third meaning.
- Marking SPY, Fear & Greed, the Polymarket board or the volatility table.
- Any change to `financial-dashboard-history`. `Sheet1`'s append-only column invariant is
  untouched; this feature is a **reader only**.


---

## What changed during implementation

Four things the design got wrong, all caught by running it against real data rather than by
unit tests. Recorded here because each one is a trap the next person would fall into too.

1. **The digest must not decide the mark.** As designed, `/api/history` compared the sheet's
   newest row against its previous row. That is wrong twice: the sheet is a 10am/10pm
   snapshot while the dashboard renders LIVE values, so a print landing at 8:30am went
   unmarked for hours; and when today's row did not exist yet the comparison degenerated to
   a row against itself and returned nothing. Split into `historyFor()` (what history knows)
   and `markFor()` (compares against the live value).

2. **Precision — the one that would have ruined the feature.** The scraper writes rounded
   values (`-0.56`) and `/api/fred` returns full precision (`-0.559`). Comparing them
   directly marked **six metrics every single day**, all false: nfci, m2, retail, indpro,
   durable, profitMargin. Every unit test passed because they used consistent precision on
   both sides. Fixed by comparing at the sheet's stored precision, floored at 2 dp, with a
   regression test built from the exact live/baseline pairs observed on 2026-08-25.

3. **Four metrics are not markable at all.** The design said `spEps` and the Four Horsemen
   could derive their baseline from `/api/fred`'s `history[]`. They cannot: a FRED history
   is a list of OBSERVATIONS, so its newest point IS the current value and the baseline can
   never differ from it. They would have shipped as inert marks that silently never fire.
   Removed, with the reasoning recorded in `lib/marks.js`. Enabling them means four new
   far-right columns in the scraper.

4. **The header chip disagreed with the page.** Cards suppress marks on stale or unavailable
   values; the chip counted them anyway. On a Sheet-last-known-good load it read
   "4 new prints" above a page with no marks on it. `collectLiveValues` now applies the same
   freshness filter, and a regression test covers it.

Also worth noting: `heldFrom` is retained on move-tier entries even though only print marks
display it — a few bytes against the cost of a second shape to reason about.

**Verification performed:** 334 Jest tests; `npm run build` with `/api/history` confirmed as
`λ` (not `○`); the digest built against the real sheet (22 metrics, 4.1KB); an end-to-end
pass combining the live `/api/fred`, `/api/market-extra` and `/api/sheets` payloads with the
real digest (0 marks — 2026-08-25 was genuinely a quiet day, matching an independent analysis
of the raw sheet); and a browser pass against a production build confirming the marks render,
the chip count matches the rendered marks exactly, and all four popovers portal to
`document.body`, stay inside the viewport, and have nothing painted over them.
