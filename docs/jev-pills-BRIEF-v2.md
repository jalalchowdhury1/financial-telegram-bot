# Jev regime pills — v2 brief: mobile polish + tap-to-detail (2026-09-19)

Read this whole file AND `docs/jev-pills-BRIEF.md` (v1, the data contract) before writing
code. Build ONLY the chunk you were assigned. Do not `git commit`, `git push`, or deploy.
Tests: from `dashboard/`, `npx jest --testPathPattern <name>`; the full suite (`npx jest`)
must stay green. `npm run build` must succeed (from `dashboard/`).

## Owner's words

> "Can we make sure it looks great on mobile too. Also if I click it, it should show me
> all the stats that pill is taking nicely."

So: (1) the "🧪 Jev Regime Pills" card must look great at phone width (390 px) and on
desktop; (2) tapping a pill opens a detail view listing EVERY input the pill's rule reads,
the threshold each input is tested against, whether it fired, and what Jev said.

Nothing about the verdict logic changes. `ruleVerdicts`, `mergeVerdicts`, `conflictPairs`,
the Telegram line and the KV log keep their exact behaviour. `JEV_PILLS=off` still renders
nothing.

## Payload additions (the contract between chunk F and chunk G)

`/api/jev-pills` (via `assemblePills` in `lib/jevPills.js`) gains:

```jsonc
{
  "pills": {
    "regime": {
      "verdict": "neutral", "p": null, "by": "rule", "reason": "Score 0: ...",   // unchanged
      "jev": { "verdict": "neutral", "p": 0.41 }   // NEW: Jev's raw answer for this pill,
                                                    // or null when Jev gave no answer
    }
    // ... same for every pill
  },
  "factors": {                                     // NEW, always present (never null)
    "regime": {
      "summary": "Score 0 → neutral (risk-on needs ≥ 2, risk-off ≤ −1)",
      "rows": [
        { "label": "SPY vs 200-day avg", "value": "+6.3%", "test": "> 0 → +1, else −1", "hit": true,  "effect": "+1" },
        { "label": "Fear & Greed",       "value": "29",    "test": "≥ 50 → +1 · < 30 → −1", "hit": true,  "effect": "−1" },
        { "label": "HYG/LQD 20d",        "value": "−0.0%", "test": "> 0 → +1 · < −1 → −1", "hit": false, "effect": "0" }
      ]
    },
    "recession": { "summary": "...", "rows": [ ... ] },
    "breadth":   { "summary": "...", "rows": [ ... ] },
    "hedging":   { "summary": "...", "rows": [ ... ] },
    "conflict":  { "summary": "...", "rows": [ ... ] }
  }
}
```

Row shape (every row, every pill): `{ label: string, value: string, test: string,
hit: boolean, effect: string }`. `value` is already formatted for display (`'n/a'` when the
input is missing). `hit` = this row's threshold fired. `effect` = what firing does
(`'+1'`, `'−1'`, `'0'`, `'high'`, `'rising'`, `'rolling-over'`, `'broad'`, `'expensive'`,
`'cheap'`, `'divergence'`, or `''`).

Rows per pill (order fixed, thresholds copied from `ruleVerdicts` — NEVER invent a new one):

- **regime** (3 rows): SPY vs 200-day avg (`> 0 → +1, else −1`); Fear & Greed
  (`≥ 50 → +1 · < 30 → −1 · else 0`); HYG/LQD 20d (`> 0 → +1 · < −1 → −1 · else 0`).
  `summary` = `Score N → <verdict> (risk-on needs ≥ 2, risk-off ≤ −1)`.
- **recession** (5 rows, in chain order): Sahm rule vs `≥ 0.5 → high`; Yield curve (2s10s)
  + claims vs `< 0 and claims ≥ 260k → high`; Sahm rule vs `≥ 0.2 → rising`; Yield curve
  (2s10s) vs `< 0 → rising`; Jobless claims vs `≥ 260k → rising`; NFCI vs `> 0 → rising`.
  (That is 6 rows; fine.) `summary` = `First rule that fires wins; none fired → low` or
  `<rule that fired> → <verdict>`.
- **breadth** (3 rows): RSP/SPY 20d (`< −1.5% with RSP vs 50d < 0 → rolling-over · > 0 with
  IWM > 0 → broad`); RSP/SPY vs 50d avg (`< 0 (with 20d < −1.5%) → rolling-over`); IWM/SPY 20d
  (`> 0 (with RSP 20d > 0) → broad`). `summary` = one sentence naming the verdict and why.
- **hedging** (2 rows): IV percentile (1y) (`> 70 → expensive · < 20 (with VRP < 6) → cheap`);
  VRP (`> 10 → expensive · < 6 (with IV pct < 20) → cheap`). `summary` likewise.
- **conflict** (4 rows, one per pair, always all four): label = pair name (`sentiment vs
  price`, `2s10s vs 3m10y`, `credit vs equities`, `breadth vs index`); `value` = the two
  numbers being compared, e.g. `F&G 29 · SPY +6.3% vs 200d`; `test` = the pair's rule in
  words, e.g. `F&G < 35 with SPY > +3% · or F&G > 65 with SPY < −3%`; `hit` = that pair is in
  `conflictPairs(data)`; `effect` = `'divergence'` when hit else `''`. `summary` =
  `N of 4 pairs diverge → <verdict>`.

Consistency test (chunk F must write it): for a battery of data objects (the fixtures in
`lib/__tests__/jevBrief.test.js`, plus the all-null case), the verdict implied by the
factors' `hit` flags equals `ruleVerdicts(data)[pill].verdict` for every pill.

## Chunk F — factors in the lib (DeepSeek)

Files: `dashboard/lib/jevBrief.js` (add `export function pillFactors(data)`),
`dashboard/lib/jevPills.js` (add `factors: pillFactors(data)` to the payload and set
`pills[pill].jev = jevAnswers?.[pill] ? { verdict, p } : null` for every pill, AFTER
`mergeVerdicts` — do NOT change `mergeVerdicts`), tests in `lib/__tests__/jevBrief.test.js`
and `lib/__tests__/jevPillsRoute.test.js`. `pillFactors(null)` returns all five pills with
`summary: 'no data'` and rows whose values are `'n/a'` and `hit: false`. Pure, no throw.
Use the same `safeNum`/formatting helpers already in the file; format percentages with one
decimal and a sign (`+6.3%`), Sahm/NFCI/yield curve with two decimals, claims as `203k`,
IV percentile as an integer, VRP with one decimal, F&G as an integer.

## Chunk G — component + modal + mobile CSS (DeepSeek)

Files: `dashboard/components/JevPills.js` (rewrite), NEW `dashboard/components/JevPillModal.js`,
`dashboard/app/globals.css` (append a `/* === Jev pills === */` block; do not edit existing
rules), `dashboard/components/__tests__/JevPills.test.js` (update + extend). Do not touch
`app/page.js` or anything in `lib/`. Chunk G must work when `data.factors` is missing or
`pill.jev` is missing (old payloads, the /tmp last-good fallback): the modal then shows the
`reason` text and "inputs unavailable".

### Card (JevPills.js)

- Header: `🧪 Jev Regime Pills` + the since chip. Chip text SHORT: no baseline → `No baseline
  yet` (badge-blue); no change → `Unchanged since yesterday` (badge-green); changes →
  `Softening ↓` / `Hardening ↑` / `Mixed` (green/red/yellow) and the chip's `title` lists the
  changes (`breadth narrow → broad`). Header wraps on narrow screens (`flex-wrap: wrap`).
- Pills: five `<button type="button" className="jev-pill">` elements in a `.jev-grid`.
  Each shows `.label` (Regime / Recession / Breadth / Hedges / Conflict), a verdict badge with
  a FRIENDLY label (`risk-on → Risk-on`, `rolling-over → Rolling over`, `mild-divergence →
  Mild divergence`, `major-divergence → Major divergence`, others capitalised), a small
  source tag (`Jev 0.82` / `Jev` / `rule`), and a chevron `›` on the right. Keep the
  `title` behaviour (conflict lists pairs, others show the reason).
- Layout via CSS classes (no inline grid): desktop `.jev-grid { display:grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px }`, each `.jev-pill`
  styled like `.indicator-pill` (same background/border/radius/padding, `cursor: pointer`,
  `text-align: left`, `font: inherit`, `color: inherit`, hover lift, `:focus-visible`
  outline). At `max-width: 640px`: `.jev-grid { grid-template-columns: 1fr; gap: 8px }` and
  `.jev-pill` becomes ONE ROW: `display:flex; align-items:center; justify-content:
  space-between; min-height: 52px; padding: 10px 14px;` label on the left, badge + source tag
  + chevron on the right, nothing wraps (`white-space: nowrap` on the badge). Tap targets
  ≥ 44 px tall.
- Footer: `as of 7:42 AM ET` (format `data.asOf` with
  `toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute:
  '2-digit' })` inside a try/catch; fall back to the raw string) and, when `mode === 'rules'`,
  ` · rules only`; when `_meta.jev === 'off'`, ` · Jev off`. The card only renders after the
  client fetch, so there is no SSR/hydration concern, but keep everything deterministic.
- State: `const [open, setOpen] = useState(null)` (pill key). Clicking a pill opens
  `<JevPillModal pillKey=… data={data} onClose=…>`.

### Modal (JevPillModal.js)

Copy the look of `components/MarketModal.js` exactly (backdrop `rgba(30,41,59,0.8)` + blur,
glass card `rgba(17,24,39,0.9)`, 16 px radius, 28 px padding, header with `h2` + 32 px ×
close button, section labels `.85rem 600 uppercase`, JetBrains Mono for numbers, `fadeIn`
/ `fadeInUp` animations). Add: close on Escape (`useEffect` keydown listener), close on
backdrop click, `role="dialog" aria-modal="true" aria-labelledby`. On phones (`max-width:
640px`): container `width: calc(100vw - 24px)`, `max-height: 88vh`, `overflow-y: auto`,
padding 20 px.

Sections, top to bottom:

1. Header: `<label> · <Friendly verdict badge>`; sub-line `Decided by the rule` or
   `Decided by Jev (confidence 0.82)`.
2. **Why** — `pill.reason` in a muted paragraph. For the conflict pill, each firing pair's
   `detail` on its own line.
3. **Inputs the rule checks** — `factors[pillKey].summary` then a table with columns
   Input · Value · Rule · fired? — one row per factor row; `hit` rows get a ✓ in green
   (`var(--green)`) with the `effect` text, non-hit rows a muted `—`. Value column in
   JetBrains Mono. On phones the table becomes stacked cards (label bold, value right,
   rule text below in `.72rem` muted) — never horizontally scrolling.
4. **Jev's view** — from `pill.jev` + `data.mode` + `data._meta?.jev`:
   - `pill.jev` present and `by === 'jev'`: `Jev decided: <verdict> (0.82)`.
   - `pill.jev` present and `by === 'rule'`: `Jev said <verdict> (0.41) — below the 0.6
     confidence floor, so the rule stands` (or `— agrees with the rule` when verdicts match).
   - `mode === 'rules'`: `Jev not consulted (rules-only mode)`.
   - `_meta.jev === 'off'`: `Jev off — no API key configured`.
   - otherwise: `Jev gave no answer this refresh`.
5. **Sources** — small muted line from `data._meta?.sources` limited to the keys the pill
   uses: regime → spy, fg, breadth; recession → fred; breadth → breadth; hedging → vol;
   conflict → spy, fg, fred, breadth. Trim each value to 60 chars with `…`.

### Tests (`components/__tests__/JevPills.test.js`)

Keep every existing case (update expected strings to the friendly labels / new chip text /
new footer). Add: clicking the Regime pill opens a dialog containing the factor rows'
labels and values and the "Decided by the rule" line; Escape closes it; backdrop click
closes it; a payload without `factors` opens the modal and shows "inputs unavailable";
Jev-below-floor text renders when `pill.jev = { verdict:'narrow', p:0.32 }` and `by:'rule'`.
Use `@testing-library/react` (`fireEvent`, `screen.getByRole('dialog')`).

## Definition of done (each chunk)

- `npx jest` green from `dashboard/`, `npm run build` succeeds.
- Reply with: files changed, test counts before/after, and any deviation from this brief.
