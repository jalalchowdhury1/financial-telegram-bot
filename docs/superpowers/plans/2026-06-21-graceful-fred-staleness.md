# Graceful FRED Staleness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the last-known FRED value in orange with a 🕐 clock instead of N/A when a series is merely stale; reserve N/A for true unavailability; alert only when a series is >3 days overdue.

**Architecture:** Centralize the behavior in `lib/freshness.js` (keep stale values, expose `staleDays`, return a `tone`), then style each FRED tile by `tone`, bump three late-month monthly deadlines, and rewrite the health check to alert on overdue (not on normal lag) across both `indicators` and `checklist`.

**Tech Stack:** Next.js (React), Jest + @testing-library/react, Python + pytest.

**Spec:** `docs/superpowers/specs/2026-06-21-graceful-fred-staleness-design.md`

**Working dir for JS test commands:** `cd dashboard`. Python tests run from repo root.

---

### Task 1: `freshness.js` — keep stale values, add `staleDays`, return `tone`

**Files:**
- Modify: `dashboard/lib/freshness.js`
- Test: `dashboard/lib/__tests__/freshness.test.js`

- [ ] **Step 1: Update existing tests to the new semantics + add new tests**

In `dashboard/lib/__tests__/freshness.test.js`, replace the `withFreshness` describe block's "nulls a stale value" test and the `freshnessNote` "stale" test, and add `staleDays`/`tone` coverage. Apply these exact edits:

Replace:
```js
    test('nulls a stale value and flags stale', () => {
        const r = withFreshness(1.23, '2026-05-01', 5, NOW);
        expect(r.value).toBeNull();
        expect(r.stale).toBe(true);
        expect(r.unavailable).toBe(false);
    });
```
with:
```js
    test('KEEPS a stale value (no longer nulled) and flags stale + staleDays', () => {
        const r = withFreshness(1.23, '2026-05-01', 5, NOW); // 29 days old, deadline 5
        expect(r.value).toBe(1.23);          // value is kept, not nulled
        expect(r.stale).toBe(true);
        expect(r.unavailable).toBe(false);
        expect(r.staleDays).toBe(24);        // floor(29 - 5)
    });

    test('fresh value has staleDays 0', () => {
        expect(withFreshness(1.23, '2026-05-29', 5, NOW).staleDays).toBe(0);
    });
```

Replace:
```js
    test('stale value is amber with a "couldn\'t refresh" note', () => {
        const note = freshnessNote({ value: null, asOf: '2026-05-01', stale: true });
        expect(note.amber).toBe(true);
        expect(note.suffix).toContain("couldn't refresh");
    });
```
with:
```js
    test('stale value is orange-toned with a clock note and keeps showing', () => {
        const note = freshnessNote({ value: 1.23, asOf: '2026-05-01', stale: true });
        expect(note.tone).toBe('stale');
        expect(note.amber).toBe(true);          // back-compat getter
        expect(note.suffix).toContain('🕐');
        expect(note.suffix).toContain('stale');
    });

    test('truly unavailable value is unavailable-toned', () => {
        const note = freshnessNote({ value: null, asOf: null, stale: false, unavailable: true });
        expect(note.tone).toBe('unavailable');
        expect(note.suffix).toContain('Unavailable');
    });
```

- [ ] **Step 2: Run the tests to verify the new ones fail**

Run: `cd dashboard && npx jest lib/__tests__/freshness.test.js`
Expected: FAIL — `staleDays` is `undefined`, `tone` is `undefined`, stale value is `null`.

- [ ] **Step 3: Rewrite `withFreshness` and `freshnessNote` in `dashboard/lib/freshness.js`**

Replace the `withFreshness` function body with:
```js
export function withFreshness(value, asOfDate, deadlineDays, now = new Date()) {
    const unavailable = value === undefined || value === null || Number.isNaN(value);
    const stale = isStale(asOfDate, deadlineDays, now) && !unavailable;
    let staleDays = 0;
    if (stale && asOfDate) {
        const then = new Date(asOfDate);
        if (!Number.isNaN(then.getTime())) {
            const ageDays = (now.getTime() - then.getTime()) / MS_PER_DAY;
            staleDays = Math.max(0, Math.floor(ageDays - deadlineDays));
        }
    }
    return {
        value: unavailable ? null : value, // keep stale values; null ONLY when truly missing
        asOf: asOfDate ?? null,
        stale,
        unavailable,
        staleDays,
    };
}
```

Replace the `freshnessNote` function with:
```js
/**
 * Given a metric ({ value, asOf, stale, unavailable }), return the tooltip suffix and a
 * display `tone`: 'fresh' | 'stale' | 'unavailable'. `amber` is kept for back-compat.
 */
export function freshnessNote(metric) {
    if (!metric) return { suffix: '', tone: 'fresh', amber: false };
    const dateStr = formatAsOf(metric.asOf);
    const isUnavailable = metric.unavailable === true || metric.value === null || metric.value === undefined;
    if (metric.stale && !isUnavailable) {
        return {
            suffix: dateStr ? ` • 🕐 As of ${dateStr} (stale)` : ' • 🕐 Stale',
            tone: 'stale',
            amber: true,
        };
    }
    if (isUnavailable) {
        return { suffix: ' • ⚠ Unavailable — source busy, try again shortly', tone: 'unavailable', amber: true };
    }
    return { suffix: dateStr ? ` • As of ${dateStr}` : '', tone: 'fresh', amber: false };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd dashboard && npx jest lib/__tests__/freshness.test.js`
Expected: PASS (all tests, including the unchanged `isStale`/`formatAsOf` ones).

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/freshness.js dashboard/lib/__tests__/freshness.test.js
git commit -m "feat(freshness): keep stale values, expose staleDays + tone"
```

---

### Task 2: `constants.js` — bump late-month monthly deadlines to 95

**Files:**
- Modify: `dashboard/lib/constants.js:43-60` (the `FRED_FRESHNESS` object)
- Test: `dashboard/lib/__tests__/freshness.test.js`

- [ ] **Step 1: Add a failing regression test**

In `dashboard/lib/__tests__/freshness.test.js`, inside the `describe('FRED_FRESHNESS vs real reporting lag', ...)` block, add:
```js
    test('M2/Durable/Savings are NOT stale at their worst-case age (~85d, day before next print)', () => {
        const dayBeforeNextPrint = new Date('2026-06-25T12:00:00Z'); // May prints ~Jun 26-27
        for (const id of ['M2SL', 'DGORDER', 'PSAVERT']) {
            expect(isStale('2026-04-01', FRED_FRESHNESS[id], dayBeforeNextPrint)).toBe(false);
        }
    });
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npx jest lib/__tests__/freshness.test.js -t "worst-case age"`
Expected: FAIL — at deadline 80, Apr-1 on Jun 25 is 85 days → stale `true`.

- [ ] **Step 3: Bump the three deadlines in `dashboard/lib/constants.js`**

Replace:
```js
    UNRATE: 80,
    UMCSENT: 95,
    M2SL: 80,
    RSXFS: 80,
    HOUST: 80,
    INDPRO: 80,
```
with (note: keep whatever the UMCSENT comment block currently is; only the three values change):
```js
    UNRATE: 80,
    UMCSENT: 95,
    // Late-month monthly releases — newest point legitimately ages ~85d before the next
    // print (M2 H.6 ~4th Tue; Durable Goods ~25th; Personal Income ~last business day).
    // 95 covers that worst case + slack; graceful staleness shows orange/🕐 if exceeded.
    M2SL: 95,
    RSXFS: 80,
    HOUST: 80,
    INDPRO: 80,
    DGORDER: 95,
    PSAVERT: 95,
```
Then DELETE the now-duplicate `DGORDER`/`PSAVERT` lines further down in the object (search for `DGORDER:` and `PSAVERT:` and remove the old `80` entries so each key appears once). Verify with: `cd dashboard && node -e "const c=require('./lib/constants.js'); console.log(c.FRED_FRESHNESS.M2SL, c.FRED_FRESHNESS.DGORDER, c.FRED_FRESHNESS.PSAVERT)"` → if ESM `require` fails, instead `grep -c 'DGORDER:' lib/constants.js` must equal `1`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npx jest lib/__tests__/freshness.test.js`
Expected: PASS (the new test + all existing).

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/constants.js dashboard/lib/__tests__/freshness.test.js
git commit -m "feat(constants): widen M2SL/DGORDER/PSAVERT freshness to 95d (FRED late-month lag)"
```

---

### Task 3: `globals.css` — add the stale orange color

**Files:**
- Modify: `dashboard/app/globals.css:24-29` (CSS custom properties)

- [ ] **Step 1: Add the variables**

After the `--yellow-bg` line (`:root` block), add:
```css
  --orange: #f59e0b;
  --orange-bg: rgba(245, 158, 11, 0.12);
```

- [ ] **Step 2: Verify it's present**

Run: `grep -n "\-\-orange" dashboard/app/globals.css`
Expected: two matches (`--orange` and `--orange-bg`).

- [ ] **Step 3: Commit**

```bash
git add dashboard/app/globals.css
git commit -m "feat(css): add --orange for stale-data styling"
```

---

### Task 4: `BullChecklist.js` — 🕐 + orange for stale, N/A only when unavailable

**Files:**
- Modify: `dashboard/components/BullChecklist.js:55-80`
- Test: `dashboard/components/__tests__/BullChecklist.test.js` (create)

- [ ] **Step 1: Write the failing component test**

Create `dashboard/components/__tests__/BullChecklist.test.js`:
```js
import { render, screen } from '@testing-library/react';
import BullChecklist from '../BullChecklist';

const mk = (over) => ({
    value: 1, asOf: '2026-04-01', stale: false, unavailable: false, staleDays: 0,
    bullish: true, status: 'good', label: 'Item', ...over,
});

test('stale item shows a clock icon + value (not N/A) and still counts in the score', () => {
    const fred = { checklist: {
        a: mk({ label: 'Fresh A', value: 5, bullish: true }),
        b: mk({ label: 'Stale B', value: 3, stale: true, staleDays: 2, bullish: true }),
    }};
    render(<BullChecklist fred={fred} loading={false} />);
    expect(screen.getByText('+3.0%')).toBeInTheDocument(); // stale value shown, not N/A
    expect(screen.getByText('🕐')).toBeInTheDocument();      // clock signals stale
    expect(screen.getByText(/2\/2/)).toBeInTheDocument();    // score counts the stale-bullish item
});

test('unavailable item shows N/A with the ⚪ icon', () => {
    const fred = { checklist: {
        a: mk({ label: 'Gone', value: null, unavailable: true, stale: false, bullish: false }),
    }};
    render(<BullChecklist fred={fred} loading={false} />);
    expect(screen.getByText('N/A')).toBeInTheDocument();
    expect(screen.getByText('⚪')).toBeInTheDocument();
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd dashboard && npx jest components/__tests__/BullChecklist.test.js`
Expected: FAIL — current code renders `⚪` for stale (not `🕐`) and there is no clock.

- [ ] **Step 3: Update the icon and value styling**

In `dashboard/components/BullChecklist.js`, replace:
```js
                                const note = freshnessNote(item);
                                const icon = note.amber ? '⚪' : item.bullish ? '✅' : '🔴';
```
with:
```js
                                const note = freshnessNote(item);
                                const icon = note.tone === 'unavailable' ? '⚪'
                                    : note.tone === 'stale' ? '🕐'
                                    : item.bullish ? '✅' : '🔴';
```

Replace the value `<span>`'s style (the one containing `note.amber ? { color: 'var(--yellow)' }`):
```js
                                        <span className={`checklist-value ${item.bullish ? 'stat-positive' : 'stat-negative'}`} style={{ fontSize: '0.95rem', ...(note.amber ? { color: 'var(--yellow)' } : {}) }}>
```
with:
```js
                                        <span className={`checklist-value ${item.bullish ? 'stat-positive' : 'stat-negative'}`} style={{ fontSize: '0.95rem', ...(note.tone === 'stale' ? { color: 'var(--orange)' } : note.tone === 'unavailable' ? { color: 'var(--yellow)' } : {}) }}>
```

(The value text block below it is unchanged — it already renders the number when present and `'N/A'` when not.)

- [ ] **Step 4: Run to verify it passes**

Run: `cd dashboard && npx jest components/__tests__/BullChecklist.test.js`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/components/BullChecklist.js dashboard/components/__tests__/BullChecklist.test.js
git commit -m "feat(checklist): 🕐 + orange for stale items, N/A only when unavailable"
```

---

### Task 5: `EconomicIndicatorGrid.js` — 🕐 + orange for stale tiles

**Files:**
- Modify: `dashboard/components/EconomicIndicatorGrid.js:45-54`

- [ ] **Step 1: Update the value rendering by tone**

In `dashboard/components/EconomicIndicatorGrid.js`, replace:
```js
                            const note = freshnessNote(ind.metric);
                            return (
                                <div className="stat-row" key={ind.label}>
                                    <span className="stat-label">
                                        {ind.icon} <span className="tooltip-trigger" data-tooltip={`${ind.tooltip}${note.suffix}`}>{ind.label}</span>
                                    </span>
                                    <span className="stat-right">
                                        <span className={`stat-value ${statusColor(ind.status)}`} style={note.amber ? { color: 'var(--yellow)' } : undefined}>{ind.value}</span>
                                        <span className="stat-benchmark">{ind.benchmark}</span>
                                    </span>
                                </div>
                            );
```
with:
```js
                            const note = freshnessNote(ind.metric);
                            const staleStyle = note.tone === 'stale' ? { color: 'var(--orange)' }
                                : note.tone === 'unavailable' ? { color: 'var(--yellow)' }
                                : undefined;
                            const shownValue = note.tone === 'stale' ? `🕐 ${ind.value}` : ind.value;
                            return (
                                <div className="stat-row" key={ind.label}>
                                    <span className="stat-label">
                                        {ind.icon} <span className="tooltip-trigger" data-tooltip={`${ind.tooltip}${note.suffix}`}>{ind.label}</span>
                                    </span>
                                    <span className="stat-right">
                                        <span className={`stat-value ${statusColor(ind.status)}`} style={staleStyle}>{shownValue}</span>
                                        <span className="stat-benchmark">{ind.benchmark}</span>
                                    </span>
                                </div>
                            );
```

- [ ] **Step 2: Run the dashboard suite to ensure nothing regressed**

Run: `cd dashboard && npx jest`
Expected: PASS (no test targets this file directly; this guards against syntax errors).

- [ ] **Step 3: Commit**

```bash
git add dashboard/components/EconomicIndicatorGrid.js
git commit -m "feat(indicators): 🕐 + orange for stale economic-indicator tiles"
```

---

### Task 6: `page.js` — Yield Curve & Profit Margin stale states

**Files:**
- Modify: `dashboard/app/page.js:274-283` (Yield Curve value branch) and `dashboard/app/page.js:301-310` (Profit Margin value branch)

- [ ] **Step 1: Update the Yield Curve value branch**

Replace:
```js
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    <div className="hero-price" style={{ fontSize: '2.2rem', color: fred.yieldCurve.current >= 0 ? 'var(--green)' : 'var(--red)' }}>
                                        {fred.yieldCurve.current >= 0 ? '+' : ''}{fred.yieldCurve.current.toFixed(3)}%
                                    </div>
                                </div>
                                <MiniChart history={fred.yieldCurve.history} color="#818cf8" gradientId="yieldGrad" showZero={true} recessions={fred.recessions || []} />
                            </>
                        )}
```
with:
```js
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    <div className="hero-price" style={{ fontSize: '2.2rem', color: fred.yieldCurve.stale ? 'var(--orange)' : fred.yieldCurve.current >= 0 ? 'var(--green)' : 'var(--red)' }}>
                                        {fred.yieldCurve.stale ? '🕐 ' : ''}{fred.yieldCurve.current >= 0 ? '+' : ''}{fred.yieldCurve.current.toFixed(3)}%
                                    </div>
                                    {fred.yieldCurve.stale && (
                                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                            Last data {formatAsOf(fred.yieldCurve.asOf)} (stale)
                                        </div>
                                    )}
                                </div>
                                <MiniChart history={fred.yieldCurve.history} color="#818cf8" gradientId="yieldGrad" showZero={true} recessions={fred.recessions || []} />
                            </>
                        )}
```

- [ ] **Step 2: Update the Profit Margin value branch**

Replace:
```js
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    <div className="hero-price" style={{ fontSize: '2.2rem', color: 'var(--green)' }}>
                                        {fred.profitMargin.current.toFixed(2)}%
                                    </div>
                                </div>
                                <MiniChart history={fred.profitMargin.history} color="#22c55e" gradientId="profitGrad" recessions={fred.recessions || []} />
                            </>
                        )}
```
with:
```js
                        ) : (
                            <>
                                <div className="hero-price-section">
                                    <div className="hero-price" style={{ fontSize: '2.2rem', color: fred.profitMargin.stale ? 'var(--orange)' : 'var(--green)' }}>
                                        {fred.profitMargin.stale ? '🕐 ' : ''}{fred.profitMargin.current.toFixed(2)}%
                                    </div>
                                    {fred.profitMargin.stale && (
                                        <div className="hero-change" style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '4px' }}>
                                            Last data {formatAsOf(fred.profitMargin.asOf)} (stale)
                                        </div>
                                    )}
                                </div>
                                <MiniChart history={fred.profitMargin.history} color="#22c55e" gradientId="profitGrad" recessions={fred.recessions || []} />
                            </>
                        )}
```

- [ ] **Step 3: Verify `formatAsOf` is imported in page.js**

Run: `grep -n "formatAsOf" dashboard/app/page.js | head -1`
Expected: an `import { ... formatAsOf ... }` line already exists (it's used in the N/A branches). If not, add `formatAsOf` to the existing `from '../lib/freshness'` import.

- [ ] **Step 4: Build to confirm JSX is valid**

Run: `cd dashboard && npx jest && npm run build`
Expected: jest PASS; build completes with no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/app/page.js
git commit -m "feat(page): orange + 🕐 stale state for Yield Curve & Profit Margin tiles"
```

---

### Task 7: `health_check.py` — alert on overdue (>3d), sweep checklist too

**Files:**
- Modify: `scripts/health_check.py:73-102` (`check_indicators_na`) and `scripts/health_check.py:314-321` (the `run_all_checks` fred branch)
- Test: `tests/test_health_check.py`

- [ ] **Step 1: Add failing tests for the new overdue rule**

In `tests/test_health_check.py`, after `test_check_indicators_na_missing_object_is_warn`, add:
```js
```
(use Python — append this block:)
```python
def test_check_indicators_na_overdue_more_than_3_days_is_warn():
    indicators = {"m2": {"value": 2.1, "asOf": "2026-04-01", "stale": True,
                         "unavailable": False, "staleDays": 5}}
    f = hc.check_indicators_na(indicators)
    assert f["severity"] == "warn"
    assert "m2" in f["detail"]


def test_check_indicators_na_stale_within_3_days_is_ok():
    indicators = {"m2": {"value": 2.1, "asOf": "2026-04-01", "stale": True,
                         "unavailable": False, "staleDays": 2}}
    assert hc.check_indicators_na(indicators)["severity"] == "ok"


def test_check_indicators_na_sweeps_checklist_keys():
    # checklist-style metric (durable) overdue -> warn
    metrics = {"durable": {"value": 1.0, "asOf": "2026-03-01", "stale": True,
                           "unavailable": False, "staleDays": 10}}
    f = hc.check_indicators_na(metrics)
    assert f["severity"] == "warn"
    assert "durable" in f["detail"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_health_check.py -k "overdue or within_3 or sweeps" -q`
Expected: FAIL — current code treats a present (non-null) stale value as ok and ignores `staleDays`.

- [ ] **Step 3: Rewrite `check_indicators_na`**

Replace the entire `check_indicators_na` function (lines ~73-102) with:
```python
def check_indicators_na(metrics):
    """Flag dashboard metrics that are N/A (fetch failed) or genuinely overdue
    (stale > 3 days past their deadline). Normal reporting lag (a value that is present
    and stale <= 3 days, or not stale at all) is fine and never alarmed. A metric on the
    KNOWN_DISCONTINUED allowlist is expected — reported ok with a note."""
    if not isinstance(metrics, dict):
        return _finding("indicators_na", "warn", "Could not read dashboard indicators",
                        detail="/api/fred returned no metrics object to inspect.",
                        remediation="manual")
    unexpected, expected_na = [], []
    for key, ind in metrics.items():
        if not isinstance(ind, dict):
            continue
        overdue = (ind.get("staleDays") or 0) > 3
        missing = ind.get("unavailable") is True or ind.get("value") is None
        if not (overdue or missing):
            continue
        if key in KNOWN_DISCONTINUED:
            expected_na.append((key, KNOWN_DISCONTINUED[key]))
        elif missing:
            unexpected.append((key, "N/A (fetch failed)"))
        else:
            unexpected.append((key, f"overdue {ind.get('staleDays')}d past deadline (asOf={ind.get('asOf')})"))
    if unexpected:
        return _finding("indicators_na", "warn", "Dashboard indicator(s) N/A or overdue",
                        detail="; ".join(f"{k}: {why}" for k, why in unexpected),
                        remediation="manual",
                        evidence={"unexpected": dict(unexpected), "expected_na": dict(expected_na)})
    note = "; ".join(f"{k} ({why})" for k, why in expected_na) or "all indicators fresh or within lag"
    return _finding("indicators_na", "ok", "Dashboard indicators present (fresh or within lag)",
                    detail=f"Expected-N/A: {note}", evidence={"expected_na": dict(expected_na)})
```

- [ ] **Step 4: Merge `checklist` into the sweep in `run_all_checks`**

Replace:
```python
        if name == "fred" and status == 200:
            try:
                indicators = (json.loads(body) or {}).get("indicators")
            except Exception:
                indicators = None
            findings.append(check_indicators_na(indicators))
```
with:
```python
        if name == "fred" and status == 200:
            try:
                fred = json.loads(body) or {}
                # Sweep BOTH the indicator tiles and the bull-checklist metrics; the
                # checklist (m2/durable/savings/...) was previously not inspected.
                metrics = {**(fred.get("indicators") or {}), **(fred.get("checklist") or {})}
            except Exception:
                metrics = None
            findings.append(check_indicators_na(metrics))
```

- [ ] **Step 5: Run all health-check tests to verify pass (new + existing)**

Run: `python -m pytest tests/test_health_check.py -q`
Expected: PASS — the new overdue tests pass and all existing `check_indicators_na` tests still pass (a `value: None` case is still warned as "N/A (fetch failed)").

- [ ] **Step 6: Commit**

```bash
git add scripts/health_check.py tests/test_health_check.py
git commit -m "feat(health): alert on >3d overdue (not normal lag); sweep checklist too"
```

---

### Task 8: Document the new semantics in AGENTS.md

**Files:**
- Modify: `AGENTS.md` (the FRED freshness bullet near line 241-247)

- [ ] **Step 1: Update the freshness documentation**

Replace the bullet that begins `- FRED dates observations at the *start* of the period ...` (through the UMCSENT exception lines) with:
```markdown
- FRED dates observations at the *start* of the period and publishes weeks late, so fresh
  series legitimately look old. Per-metric freshness deadlines (`FRED_FRESHNESS` in
  `lib/constants.js`, days): daily≈7, weekly≈14, monthly≈80, JOLTS≈110, quarterly≈200.
  **Late-month monthlies are 95** (`UMCSENT`, `M2SL`, `DGORDER`, `PSAVERT`) — their free FRED
  series only print ~the 26th of the following month, so the newest point ages to ~85d before
  the next print; 80 false-alarmed them N/A for ~a week each month. Don't drop these to 80.
- **Stale ≠ N/A (graceful staleness).** A value past its deadline is NO LONGER nulled: it keeps
  showing as the **last-known value in orange with a 🕐 clock** ("as of <date> (stale)").
  `value` goes `null` (→ true N/A, yellow) ONLY when the fetch returns nothing
  (`unavailable:true`). `withFreshness` also returns `staleDays` (whole days past deadline).
  The health check (`scripts/health_check.py:check_indicators_na`) warns only when a metric is
  `unavailable` OR `staleDays > 3` (genuinely overdue) — normal reporting lag never alarms — and
  it sweeps BOTH `indicators` and `checklist`. A genuinely discontinued series keeps showing old
  orange data and warns daily until replaced or added to `KNOWN_DISCONTINUED`.
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs(AGENTS): document graceful staleness + 95d late-month deadlines + 3d alert"
```

---

### Task 9: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full dashboard test suite**

Run: `cd dashboard && npx jest`
Expected: PASS — all suites (freshness, BullChecklist, plus the pre-existing 102 tests).

- [ ] **Step 2: Run the dashboard build**

Run: `cd dashboard && npm run build`
Expected: build completes, no errors.

- [ ] **Step 3: Run the Python test suite**

Run: `python -m pytest -q`
Expected: PASS.

- [ ] **Step 4: Sanity-check the live FRED data the change relies on**

Run: `curl -s "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL" | tail -1`
Expected: the latest line is dated `2026-04-01` (confirms the lag is still upstream; once FRED prints May, the tiles refresh automatically).

---

## Self-Review

**Spec coverage:**
- Goal 1 (never N/A for stale) → Task 1 (keep value) + Tasks 4/5/6 (render it). ✓
- Goal 2 (orange + 🕐) → Tasks 3/4/5/6. ✓
- Goal 3 (alert >3d overdue) → Task 7. ✓
- Goal 4 (N/A only when unavailable) → Task 1 (`value` null only when unavailable) + component N/A branches. ✓
- Goal 5 (all FRED tiles) → Task 4 (checklist), Task 5 (indicator grid), Task 6 (yield curve + profit margin). ✓
- Deadlines (M2SL/DGORDER/PSAVERT→95) → Task 2. ✓
- Docs → Task 8. ✓
- Sheet fallback → explicitly deferred (not in plan). ✓

**Placeholder scan:** none — every code/test step shows full content.

**Type consistency:** `withFreshness` returns `{ value, asOf, stale, unavailable, staleDays }` (Task 1) — consumed as `staleDays` in Task 7 and `tone` (from `freshnessNote`) in Tasks 4/5; `tone` values `'fresh'|'stale'|'unavailable'` used identically across Tasks 4/5/6. `KNOWN_DISCONTINUED` referenced in Task 7 already exists. Consistent.
