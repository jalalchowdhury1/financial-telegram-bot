# Intraday Live IV for Volatility Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During market hours the 🌡️ Volatility Metrics card shows live intraday IV / rank / %ile / VRP (≤5 min old) instead of yesterday's close.

**Architecture:** A live-spot override tier in `/api/vol`: one keyless `cnbcQuotes(['.VIX','.VXN','.VVIX'])` call (5-min revalidate) feeds `buildVolMetrics` a third argument; a live level replaces the last EOD close as "current" ONLY when finite, > 0, and strictly newer by date. Rank/%ile window and RV 21d stay EOD-only. Mirrors the repo's Finnhub-overrides-Polygon pattern. Spec: `docs/superpowers/specs/2026-07-15-vol-intraday-live-design.md`.

**Tech Stack:** Next.js 13.5 App Router route handlers, Jest + @testing-library/react. All work in `dashboard/`. Branch: `vol-intraday-live`.

**Owner's bar:** "I will base my judgement on it" — a garbage quote must NEVER replace good EOD math. Every guard has a test.

---

### Task 1: `cnbcQuotes` returns the full `lastTime`

**Files:**
- Modify: `dashboard/lib/sources.js:176-196` (the `cnbcQuotes` function)
- Test: `dashboard/lib/__tests__/sources.test.js` (inside the existing `describe('cnbcQuotes', …)` block, after the `'handles a single (non-array) QuickQuote object'` test)

- [ ] **Step 1: Write the failing test**

Add inside `describe('cnbcQuotes', () => { … })`:

```js
    test('returns the full last_time as lastTime alongside the sliced asOf', async () => {
        mockFetch({ QuickQuoteResult: { QuickQuote: [
            { symbol: '.VIX', last: '16.39', change: '-0.11', change_pct: '-0.67', last_time: '2026-07-15T13:42:31.000-0400' },
            { symbol: '.VXN', last: '26.28', change: '0.00', change_pct: '0.00', last_time: '2026-07-14' },
        ] } });
        const out = await cnbcQuotes(['.VIX', '.VXN']);
        expect(out['.VIX'].lastTime).toBe('2026-07-15T13:42:31.000-0400');
        expect(out['.VIX'].asOf).toBe('2026-07-15');
        expect(out['.VXN'].lastTime).toBe('2026-07-14'); // CNBC sends date-only off-hours
        expect(out['.VXN'].asOf).toBe('2026-07-14');
    });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npx jest lib/__tests__/sources.test.js -t "lastTime" --silent`
Expected: FAIL — `expect(out['.VIX'].lastTime).toBe(…)` receives `undefined`.

- [ ] **Step 3: Write minimal implementation**

In `dashboard/lib/sources.js`, in `cnbcQuotes`, the `out[it.symbol] = { … }` object currently ends with:

```js
            asOf: typeof it.last_time === 'string' ? it.last_time.slice(0, 10) : null,
```

Add one property after it:

```js
            asOf: typeof it.last_time === 'string' ? it.last_time.slice(0, 10) : null,
            lastTime: typeof it.last_time === 'string' ? it.last_time : null,
```

- [ ] **Step 4: Run the full sources suite to verify pass + no regression**

Run: `cd dashboard && npx jest lib/__tests__/sources.test.js --silent`
Expected: PASS, all tests (additive field — copper/gold callers ignore it).

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/sources.js dashboard/lib/__tests__/sources.test.js
git commit -m "feat(sources): cnbcQuotes returns full lastTime alongside date-only asOf"
```

---

### Task 2: `buildVolMetrics` live-override math

**Files:**
- Modify: `dashboard/lib/vol.js:78-99` (the `buildVolMetrics` function + its JSDoc)
- Test: `dashboard/lib/__tests__/vol.test.js` (new `describe` block at the end)

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/lib/__tests__/vol.test.js`:

```js
describe('buildVolMetrics — live intraday overrides', () => {
    const mkSeries = (values) => values.map((v, i) => ({ date: `2026-01-${String((i % 28) + 1).padStart(2, '0')}`, value: v }));
    // 1y window: 251 EOD days at 16, last EOD close 20 on 2026-07-14 → window min 16, max 20.
    const vixWithLast = () => {
        const s = mkSeries(Array(252).fill(16));
        s[s.length - 1] = { date: '2026-07-14', value: 20 };
        return s;
    };
    const liveVix = { value: 18, date: '2026-07-15', lastTime: '2026-07-15T13:42:31.000-0400' };

    it('replaces the last EOD close when the quote date is strictly newer', () => {
        const out = buildVolMetrics({ VIX: vixWithLast() }, { SPY: Array(40).fill(100) }, { VIX: liveVix });
        const spy = out.tickers.find((t) => t.ticker === 'SPY');
        expect(spy.iv).toBeCloseTo(18);
        expect(spy.live).toBe(true);
        expect(spy.asOf).toBe('2026-07-15');
        expect(spy.ivRank1y).toBeCloseTo(50);   // (18−16)/(20−16) against the EOD window
        expect(spy.rv21).toBeCloseTo(0, 6);     // RV stays EOD-only (flat closes)
        expect(spy.vrp).toBeCloseTo(18);        // live IV − EOD RV
        expect(out.updated_at).toBe('2026-07-15');
        expect(out.live_at).toBe('2026-07-15T13:42:31.000-0400');
    });

    it('scales the live level by the proxy multiplier but ranks on the raw index', () => {
        const vxn = mkSeries(Array(252).fill(20));
        vxn[vxn.length - 1] = { date: '2026-07-14', value: 30 };
        const out = buildVolMetrics({ VXN: vxn }, {}, { VXN: { value: 25, date: '2026-07-15', lastTime: '2026-07-15T10:00:00.000-0400' } });
        const qqq = out.tickers.find((t) => t.ticker === 'QQQ');
        const tqqq = out.tickers.find((t) => t.ticker === 'TQQQ');
        expect(qqq.iv).toBeCloseTo(25);
        expect(tqqq.iv).toBeCloseTo(75); // 3×25
        expect(tqqq.ivRank1y).toBeCloseTo(qqq.ivRank1y); // multiplier cancels
        expect(tqqq.live).toBe(true);
    });

    it('skips the override when the quote is not strictly newer (evenings/weekends)', () => {
        const sameDay = buildVolMetrics({ VIX: vixWithLast() }, {}, { VIX: { ...liveVix, date: '2026-07-14' } });
        const older = buildVolMetrics({ VIX: vixWithLast() }, {}, { VIX: { ...liveVix, date: '2026-07-11' } });
        for (const out of [sameDay, older]) {
            const spy = out.tickers.find((t) => t.ticker === 'SPY');
            expect(spy.iv).toBeCloseTo(20);
            expect(spy.live).toBe(false);
            expect(spy.asOf).toBe('2026-07-14');
            expect(out.live_at).toBeNull();
        }
    });

    it('never lets a garbage quote replace good EOD math', () => {
        const cases = [
            { value: NaN, date: '2026-07-15' },
            { value: -3, date: '2026-07-15' },
            { value: 0, date: '2026-07-15' },
            { value: 18, date: null },
            { value: 18 }, // no date at all
        ];
        for (const quote of cases) {
            const out = buildVolMetrics({ VIX: vixWithLast() }, {}, { VIX: quote });
            const spy = out.tickers.find((t) => t.ticker === 'SPY');
            expect(spy.iv).toBeCloseTo(20);
            expect(spy.live).toBe(false);
        }
    });

    it('a live quote alone (no EOD series) produces nothing — no window, no metrics', () => {
        const out = buildVolMetrics({}, {}, { VIX: liveVix });
        const spy = out.tickers.find((t) => t.ticker === 'SPY');
        expect(spy.iv).toBeNull();
        expect(spy.live).toBe(false);
    });

    it('a date-only lastTime never becomes live_at (avoids UTC-midnight misformatting)', () => {
        const out = buildVolMetrics({ VIX: vixWithLast() }, {}, { VIX: { value: 18, date: '2026-07-15', lastTime: '2026-07-15' } });
        const spy = out.tickers.find((t) => t.ticker === 'SPY');
        expect(spy.live).toBe(true);        // the override still applies
        expect(out.live_at).toBeNull();     // but the footnote falls back to updated_at
        expect(out.updated_at).toBe('2026-07-15');
    });

    it('is fully backward-compatible when liveQuotes is omitted', () => {
        const out = buildVolMetrics({ VIX: vixWithLast() }, { SPY: Array(40).fill(100) });
        const spy = out.tickers.find((t) => t.ticker === 'SPY');
        expect(spy.iv).toBeCloseTo(20);
        expect(spy.live).toBe(false);
        expect(out.live_at).toBeNull();
    });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && npx jest lib/__tests__/vol.test.js -t "live intraday" --silent`
Expected: FAIL — `spy.live` is `undefined`, live values not applied.

- [ ] **Step 3: Implement**

Replace `buildVolMetrics` (and its JSDoc) in `dashboard/lib/vol.js` with:

```js
/**
 * Assemble the payload for /api/vol.
 *
 * `liveQuotes` (optional) carries intraday index levels from CNBC's quote
 * endpoint, keyed by index name: { VIX: { value, date, lastTime }, … }. A live
 * level REPLACES the last EOD close as "current" ONLY when it is finite, > 0,
 * and its date is STRICTLY newer than the last EOD point — evenings, weekends
 * and a lagging quote all fall back to plain EOD behavior. The 1y
 * rank/percentile window stays EOD-only, and RV 21d never sees a partial day.
 * `live_at` is the newest applied quote's full timestamp (only ever a full
 * ISO string with a 'T'; a date-only lastTime is withheld so the UI can't
 * misparse it as UTC midnight).
 *
 * @param {object} indexSeries  {VIX|VXN|VVIX: ascending [{date, value}] | null}
 * @param {object} etfCloses    {SPY|…: ascending array of closes | null}
 * @param {object} [liveQuotes] {VIX|VXN|VVIX: {value, date, lastTime}} (optional)
 * @returns {{updated_at: string|null, live_at: string|null, tickers: Array}}
 */
export function buildVolMetrics(indexSeries = {}, etfCloses = {}, liveQuotes = {}) {
    let updated = null;
    let liveAt = null;
    const tickers = Object.entries(VOL_PROXIES).map(([ticker, { index, mult }]) => {
        const series = indexSeries[index] || null;
        const last = series && series.length ? series[series.length - 1] : null;
        const window = series ? series.slice(-ONE_YEAR).map((p) => p.value) : [];
        const quote = liveQuotes ? liveQuotes[index] : null;
        const live = !!(last && quote && Number.isFinite(quote.value) && quote.value > 0
            && typeof quote.date === 'string' && quote.date > last.date);
        const current = live ? quote.value : (last ? last.value : null);
        const asOf = live ? quote.date : (last ? last.date : null);
        const iv = current != null ? mult * current : null;
        const rank = current != null ? ivRank(window, current) : null;
        const pctile = current != null ? ivPercentile(window, current) : null;
        const rv21 = realizedVol(etfCloses[ticker] || []);
        const vrp = iv != null && rv21 != null ? iv - rv21 : null;
        if (asOf && (!updated || asOf > updated)) updated = asOf;
        if (live && typeof quote.lastTime === 'string' && quote.lastTime.includes('T')
            && (!liveAt || quote.lastTime > liveAt)) liveAt = quote.lastTime;
        return { ticker, proxy: mult === 1 ? index : `${mult}×${index}`, iv, ivRank1y: rank, ivPctile1y: pctile, rv21, vrp, live, asOf };
    });
    return { updated_at: updated, live_at: liveAt, tickers };
}
```

(Behavioral note: when no override applies, `current === last.value`, so every existing
code path is unchanged; the row just gains `live: false`.)

- [ ] **Step 4: Run the full vol suite — new AND pre-existing tests must pass**

Run: `cd dashboard && npx jest lib/__tests__/vol.test.js --silent`
Expected: PASS (all describe blocks — the pre-existing `buildVolMetrics` tests prove backward compatibility).

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/vol.js dashboard/lib/__tests__/vol.test.js
git commit -m "feat(vol): live intraday override in buildVolMetrics — guarded, EOD window untouched"
```

---

### Task 3: Route wiring — fetch live quotes, label sources `+live`

**Files:**
- Modify: `dashboard/app/api/vol/route.js`

No route-level Jest exists for this repo's API routes (convention: pure logic is lib-tested; routes are verified live incl. `?_fail=` on prod). The route change is deliberately thin glue; it is exercised end-to-end in Task 5.

- [ ] **Step 1: Update the import**

In `dashboard/app/api/vol/route.js` change line 20:

```js
import { cnbcHistory, cnbcQuotes, polygonDaily, fredObservations, yahooChart } from '../../../lib/sources';
```

- [ ] **Step 2: Add the live-quote fetcher**

Insert after the `fetchEtfCloses` function (after line 79):

```js
/**
 * Live intraday index levels — ONE keyless CNBC quote call for all three
 * indices, 5-min revalidate (vs 30-min for the daily histories). Gated by
 * vol_cnbc (per-SOURCE semantics, same gate as the CNBC daily bars). Any
 * failure returns {} — buildVolMetrics then serves EOD closes exactly as
 * before this tier existed. Same live-overrides-stale pattern as SPY's
 * Finnhub spot override.
 */
async function fetchLiveQuotes(faults, notes) {
    try {
        const quotes = await gate('vol_cnbc', faults, () => cnbcQuotes(INDICES.map((n) => `.${n}`), { revalidate: 300 }));
        const out = {};
        for (const n of INDICES) {
            const q = quotes[`.${n}`];
            if (q) out[n] = { value: q.price, date: q.asOf, lastTime: q.lastTime };
        }
        return out;
    } catch (e) {
        notes.push(`live quotes: ${String(e?.message).slice(0, 80)}`);
        return {};
    }
}
```

- [ ] **Step 3: Wire it into the producer**

In `GET`, replace the producer body between `const notes = [];` and the `return { …payload… }` with:

```js
        const notes = [];
        const [indexResults, etfResults, liveQuotes] = await Promise.all([
            Promise.all(INDICES.map((n) => fetchIndex(n, fredKey, faults, notes))),
            Promise.all(TICKERS.map((t) => fetchEtfCloses(t, polygonKey, faults, notes))),
            fetchLiveQuotes(faults, notes),
        ]);
        const indexSeries = {};
        INDICES.forEach((n, i) => {
            indexSeries[n] = indexResults[i].series;
        });
        const etfCloses = {};
        const etfSources = [];
        TICKERS.forEach((t, i) => {
            etfCloses[t] = etfResults[i].closes;
            if (etfResults[i].source) etfSources.push(`${t}:${etfResults[i].source}`);
        });

        const payload = buildVolMetrics(indexSeries, etfCloses, liveQuotes);
        // Which indices actually got a live override (buildVolMetrics is the
        // single authority on that decision — derive, don't re-guess).
        const liveIndices = new Set(payload.tickers.filter((t) => t.live).map((t) => VOL_PROXIES[t.ticker].index));
        const indexSources = [];
        INDICES.forEach((n, i) => {
            if (indexResults[i].source) indexSources.push(`${n}:${indexResults[i].source}${liveIndices.has(n) ? '+live' : ''}`);
        });

        const anyData = payload.tickers.some((t) => t.iv != null || t.rv21 != null);
        return {
            ...payload,
            _meta: {
                source: indexSources.concat(etfSources).join(' · ') || 'none',
                hasErrors: !anyData,
                messages: notes,
            },
        };
```

(Note the reorder: `indexSources` is now built AFTER `buildVolMetrics` so the `+live`
suffix can be derived from the payload's own `live` flags. `fallback` in the `serve()`
opts gains `live_at: null` for shape consistency:)

```js
        fallback: { updated_at: null, live_at: null, tickers: [], _meta: { source: 'Unavailable', hasErrors: true, messages: [] } },
```

- [ ] **Step 4: Update the route's header comment**

Add to the doc comment at the top of `route.js`, after the ETF closes lines:

```js
 *   Live intraday overrides: CNBC quote endpoint '.VIX'/'.VXN'/'.VVIX'
 *                (keyless, 5-min revalidate; gated by vol_cnbc). Applied by
 *                buildVolMetrics only when strictly newer than the last EOD
 *                close — see lib/vol.js. Sources show it as e.g. 'VIX:cboe+live'.
```

- [ ] **Step 5: Sanity-run the whole dashboard test suite**

Run: `cd dashboard && npm test -- --silent`
Expected: PASS (route isn't unit-tested, but this catches import/syntax slips via any suite that touches sources/vol).

- [ ] **Step 6: Commit**

```bash
git add dashboard/app/api/vol/route.js
git commit -m "feat(vol): route fetches live CNBC index quotes — 5-min revalidate, vol_cnbc-gated, +live source labels"
```

---

### Task 4: Footnote UI — intraday timestamp + green dot

**Files:**
- Modify: `dashboard/components/VolMetricsTable.js`
- Test: `dashboard/components/__tests__/VolMetricsTable.test.js`

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/components/__tests__/VolMetricsTable.test.js`:

```js
it('shows the intraday footnote with an ET timestamp when rows are live', async () => {
    const livePayload = {
        ...payload,
        updated_at: '2026-07-15',
        live_at: '2026-07-15T13:42:31.000-0400',
        tickers: payload.tickers.map((t) => ({ ...t, live: t.ticker !== 'UVXY', asOf: '2026-07-15' })),
    };
    global.fetch = jest.fn().mockResolvedValue({ json: async () => livePayload });
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('SPY')).toBeInTheDocument());
    expect(screen.getByText(/As of 2026-07-15, 1:42 PM ET · intraday/)).toBeInTheDocument();
});

it('falls back to the plain date when live rows have no usable timestamp', async () => {
    const p = { ...payload, updated_at: '2026-07-15', live_at: null, tickers: payload.tickers.map((t) => ({ ...t, live: true })) };
    global.fetch = jest.fn().mockResolvedValue({ json: async () => p });
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('SPY')).toBeInTheDocument());
    expect(screen.getByText(/As of 2026-07-15 · intraday/)).toBeInTheDocument();
});
```

(The existing `'renders a row per ticker'` test — payload with no `live` fields — keeps
asserting the unchanged EOD footnote `As of 2026-07-03`, guarding backward compatibility.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dashboard && npx jest components/__tests__/VolMetricsTable.test.js --silent`
Expected: the two new tests FAIL (no intraday footnote rendered); the pre-existing ones PASS.

- [ ] **Step 3: Implement**

In `dashboard/components/VolMetricsTable.js`, add after the `vrpColor` definition:

```js
/**
 * '2026-07-15T13:42:31.000-0400' → '2026-07-15, 1:42 PM ET'. Normalizes the
 * narrow no-break space newer ICU builds put before AM/PM. Returns null on
 * anything unparseable so the caller falls back to the plain date.
 */
const formatLiveAt = (iso) => {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return null;
    const date = d.toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
    const time = d.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' }).replace(/[\u202f\u00a0]/g, ' ');
    return `${date}, ${time} ET`;
};

/** Footnote timestamp: live rows get a green dot + ET time + 'intraday'. */
const asOfNote = (data) => {
    const anyLive = Array.isArray(data.tickers) && data.tickers.some((t) => t.live);
    if (anyLive) {
        const when = (data.live_at && formatLiveAt(data.live_at)) || data.updated_at;
        if (when) return <> <span style={{ color: 'var(--green)' }}>●</span> As of {when} · intraday.</>;
    }
    return data.updated_at ? ` As of ${data.updated_at}.` : '';
};
```

Then in the footnote div (line 79-82), replace:

```jsx
                            {data.updated_at ? ` As of ${data.updated_at}.` : ''}
```

with:

```jsx
                            {asOfNote(data)}
```

- [ ] **Step 4: Run the component suite**

Run: `cd dashboard && npx jest components/__tests__/VolMetricsTable.test.js --silent`
Expected: PASS — all tests including the two pre-existing ones.

- [ ] **Step 5: Commit**

```bash
git add dashboard/components/VolMetricsTable.js dashboard/components/__tests__/VolMetricsTable.test.js
git commit -m "feat(vol): intraday footnote — green dot + ET timestamp when live overrides are active"
```

---

### Task 5: End-to-end verification (local)

**Files:** none modified — verification only. Reliability gate: the owner trades on these numbers.

- [ ] **Step 1: Full test suite + build**

Run: `cd dashboard && npm test -- --silent && npm run build`
Expected: all Jest suites PASS; build output lists `/api/vol` as `λ` (server) — **never `○` (static)**. If `○`, the request-touch was lost; stop and fix.

- [ ] **Step 2: Live local smoke test**

Run (needs `FRED_API_KEY`/`POLYGON_KEY` in `dashboard/.env.local` if present; the CNBC tiers are keyless so the live path works without them):

```bash
cd dashboard && (npm run dev &) && sleep 8
curl -s 'http://localhost:3000/api/vol' | python3 -m json.tool | head -60
```

Expected **during US market hours (9:30–16:00 ET)**: every ticker `live: true`, `asOf` = today, `live_at` a full ISO timestamp, `_meta.source` like `VIX:cboe+live · VXN:cboe+live · VVIX:cboe+live · SPY:cnbc · …`.
Expected **off-hours**: `live: false` everywhere, payload byte-for-byte equivalent to pre-change behavior (plus the new `live`/`live_at` fields).

- [ ] **Step 3: Cross-check live values against an independent source (CBOE's own delayed quotes)**

```bash
curl -s 'https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VIX.json' | python3 -c "import json,sys; print('CBOE VIX:', json.load(sys.stdin)['data']['current_price'])"
curl -s 'http://localhost:3000/api/vol' | python3 -c "import json,sys; d=json.load(sys.stdin); t=[x for x in d['tickers'] if x['ticker']=='SPY'][0]; print('our SPY IV:', t['iv'], 'live:', t['live'])"
```

Expected: our SPY IV within ~0.5 vol pts of CBOE's (CBOE is 15-min delayed; small drift is normal). Repeat mentally for VXN via `_VXN.json` if in doubt. **If the numbers diverge materially, STOP — do not merge.**

- [ ] **Step 4: Fault injection — the reliability contract**

```bash
curl -s 'http://localhost:3000/api/vol?_fail=vol_cnbc' | python3 -c "
import json,sys; d=json.load(sys.stdin)
assert all(not t['live'] for t in d['tickers']), 'live must be off under vol_cnbc'
assert '+live' not in d['_meta']['source'], 'no +live label under vol_cnbc'
assert any(t['iv'] is not None for t in d['tickers']), 'EOD values must still serve (CBOE tier)'
print('fault path OK:', d['_meta']['source'])"
```

Expected: `fault path OK: VIX:cboe · …` — EOD data intact with the live tier disabled. Then kill the dev server (`kill %1` or `pkill -f "next dev"`).

---

### Task 6: AGENTS.md, PR, deploy verification

**Files:**
- Modify: `AGENTS.md` (§3, the Volatility-metrics bullet)

- [ ] **Step 1: Document the new tier in AGENTS.md §3**

In the `- **Volatility metrics** (…)` bullet, after the "**ETF closes** (for RV21, 3 tiers)" sub-bullet, add:

```markdown
  - **Live intraday overrides (added 2026-07-15)**: one keyless CNBC quote call
    (`.VIX`/`.VXN`/`.VVIX`, 5-min revalidate, gated by `vol_cnbc`) feeds
    `buildVolMetrics` a live "current" level that replaces the last EOD close ONLY
    when finite, > 0, and strictly date-newer — rank/%ile still use the EOD 1y
    window (UNSCALED), RV 21d never includes a partial day, VRP = live IV − EOD RV.
    Rows gain `live`, payload gains `live_at` (full ISO or null), sources show
    `VIX:cboe+live`, and the card footnote shows a green dot + ET time. Off-hours /
    quote failure ⇒ identical to pre-2026-07-15 EOD behavior (never-throw kept).
```

- [ ] **Step 2: Commit + push + open the PR**

```bash
git add AGENTS.md
git commit -m "docs(AGENTS): document the vol live intraday override tier"
git push -u origin vol-intraday-live
gh pr create --title "feat(vol): live intraday IV for the Volatility Metrics card" --body "Implements docs/superpowers/specs/2026-07-15-vol-intraday-live-design.md — live CNBC .VIX/.VXN/.VVIX quotes override the last EOD close (only when strictly newer + sane), so IV/rank/%ile/VRP are ≤5 min old during market hours. EOD window + RV untouched; vol_cnbc-gated; never-throw preserved.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01HCsn6CAHLRRfotQDY5FqeK"
```

- [ ] **Step 3: Wait for CI green, then merge**

`ci.yml` (pytest + jest + build) is the merge gate. Merge via `gh pr merge --squash --delete-branch` once green (if branch protection demands the owner's click, hand him the PR link).

- [ ] **Step 4: Post-deploy verification on production (during market hours)**

Vercel auto-deploys `main`. Then:

```bash
curl -s 'https://financial-telegram-bot-beryl.vercel.app/api/vol' | python3 -m json.tool | head -40
curl -s 'https://financial-telegram-bot-beryl.vercel.app/api/vol?_fail=vol_cnbc' | python3 -c "
import json,sys; d=json.load(sys.stdin)
assert all(not t['live'] for t in d['tickers']); print('prod fault path OK')"
```

Expected: prod shows `live: true` + today's timestamp + `+live` sources during market hours; the fault path serves EOD. Load the dashboard in a browser and eyeball the green-dot footnote. **Done only when both are confirmed live.**
