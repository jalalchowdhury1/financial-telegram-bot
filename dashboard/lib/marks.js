/**
 * Fresh Print Marks — deciding which numbers changed in a way that is NEWS.
 *
 * A number on the dashboard changes for one of two reasons: a price ticked, or a
 * new print landed. Only the second is worth an owner's attention, so we classify
 * by RELEASE CADENCE rather than by diffing values:
 *
 *   print — the series only moves when an agency publishes, so any change IS the news
 *   move  — the series moves most days, so only an outsized (2σ) move is news
 *   none  — the series moves every day; a mark here would be pure noise
 *
 * The `rate` on each entry is the measured changes-per-day over 167 days of the
 * financial-dashboard-history sheet (2026-03-12 → 2026-08-25). It is recorded as the
 * JUSTIFICATION for the class, not used at runtime: a series frozen by a dead upstream
 * would look "slow" and then fire loudly on recovery, so the class is hardcoded.
 *
 * See docs/superpowers/specs/2026-08-25-fresh-print-marks-design.md.
 */

const EPS = 1e-9;
const SIGMA_MULT = 2;
const MIN_MOVES = 20;      // below this there is no meaningful σ to compare against
const MAX_RUNS = 8;        // sparkline points
const WINDOW_DAYS = 400;   // bound the walk-back; older rows predate two schema changes

/**
 * Every metric carried by Sheet1, keyed by the name the dashboard uses for it.
 * `col` is the 0-based CSV column. Columns are position-mapped and append-only in
 * the writer repo — never renumber these without re-reading that sheet's header.
 */
export const SHEET_METRICS = {
    // ── print: changes only when an agency publishes ────────────────────────
    profitMargin:    { col: 2,  kind: 'print', rate: 0.06, label: 'Profit Margin' },
    sahmRule:        { col: 3,  kind: 'print', rate: 0.07, label: 'Sahm Rule' },
    sentiment:       { col: 4,  kind: 'print', rate: 0.03, label: 'Consumer Sentiment' },
    claims:          { col: 5,  kind: 'print', rate: 0.19, label: 'Initial Claims (4wk)' },
    nfci:            { col: 10, kind: 'print', rate: 0.12, label: 'Financial Conditions' },
    m2:              { col: 11, kind: 'print', rate: 0.08, label: 'M2 Money Supply' },
    retail:          { col: 12, kind: 'print', rate: 0.10, label: 'Retail Sales (3mo)' },
    housing:         { col: 13, kind: 'print', rate: 0.04, label: 'Housing Starts' },
    indpro:          { col: 14, kind: 'print', rate: 0.08, label: 'Industrial Production' },
    jolts:           { col: 15, kind: 'print', rate: 0.04, label: 'Job Openings (JOLTS)' },
    durable:         { col: 16, kind: 'print', rate: 0.11, label: 'Durable Goods' },
    savings:         { col: 17, kind: 'print', rate: 0.04, label: 'Savings Rate' },
    rentIndex:       { col: 18, kind: 'print', rate: 0.03, label: 'US Median Rent' },
    mortgagePayment: { col: 19, kind: 'print', rate: 0.16, label: 'Est. Monthly Mortgage' },
    mortgageRate:    { col: 20, kind: 'print', rate: 0.15, label: '30-Yr Mortgage Rate' },
    atnhpi:          { col: 33, kind: 'print', rate: 0.00, label: 'US House Price Index' },
    aaiiDiff:        { col: 35, kind: 'print', rate: 0.13, label: 'AAII Diff' },

    // ── move: daily, but a 2σ day is still worth a glance ───────────────────
    yieldCurve:      { col: 1,  kind: 'move',  rate: 0.58, label: 'Yield Curve (10Y-2Y)' },
    creditSpread:    { col: 6,  kind: 'move',  rate: 0.35, label: 'BBB Credit Spread' },
    realYields:      { col: 7,  kind: 'move',  rate: 0.60, label: 'Real Yields (10Y TIPS)' },
    peRatio:         { col: 9,  kind: 'move',  rate: 0.67, label: 'Market Valuation (P/E)' },
    copperGold:      { col: 32, kind: 'move',  rate: 0.51, label: 'Copper/Gold Ratio' },

    // ── none: a mark would be noise. Listed so the intent is explicit and testable.
    lei:             { col: 8,  kind: 'none',  rate: 0.03, label: 'Leading Economic Index' },
    tnx:             { col: 21, kind: 'none',  rate: 0.64, label: '10-Year Treasury Yield' },
    t2y:             { col: 22, kind: 'none',  rate: 0.63, label: '2-Year Treasury Yield' },
    dxy:             { col: 23, kind: 'none',  rate: 0.95, label: 'US Dollar Index' },
    cl:              { col: 24, kind: 'none',  rate: 0.54, label: 'Crude Oil WTI' },
    usdcad:          { col: 25, kind: 'none',  rate: 0.21, label: 'USD/CAD' },
    usdinr:          { col: 26, kind: 'none',  rate: 0.89, label: 'USD/INR' },
    usdbdt:          { col: 27, kind: 'none',  rate: 0.95, label: 'USD/BDT' },
    inrbdt:          { col: 28, kind: 'none',  rate: 0.27, label: 'INR/BDT' },
    cadinr:          { col: 29, kind: 'none',  rate: 0.90, label: 'CAD/INR' },
    gold:            { col: 30, kind: 'none',  rate: 0.93, label: 'Gold' },
    btc:             { col: 31, kind: 'none',  rate: 1.00, label: 'Bitcoin' },
    cadbdt:          { col: 34, kind: 'none',  rate: 0.91, label: 'CAD/BDT' },
    vixCurrent:      { col: 36, kind: 'none',  rate: 0.83, label: 'VIX (Current)' },
    vix3m:           { col: 37, kind: 'none',  rate: 0.83, label: 'VIX (3M)' },
    vixFearGreed:    { col: 38, kind: 'none',  rate: 0.78, label: 'VIX Fear/Greed' },
};

/**
 * Metrics NOT in the sheet, whose previous print is derived on the client from the
 * `history[]` arrays /api/fred already returns. LEI is deliberately absent — the FRED
 * series died and its column is written blank.
 */
export const FRED_HISTORY_METRICS = {
    spEps:        { kind: 'print', label: 'S&P 500 EPS' },
    unemployment: { kind: 'print', label: 'Unemployment Rate' },
    bankruptcies: { kind: 'print', label: 'US Bankruptcies' },
    hClaims:      { kind: 'print', label: 'Initial Jobless Claims' },
};

/** @returns {'print'|'move'|'none'} */
export function classify(key) {
    const m = SHEET_METRICS[key] || FRED_HISTORY_METRICS[key];
    return m ? m.kind : 'none';
}

/**
 * Tolerant numeric parse for one sheet cell.
 *
 * An exact 0 is treated as ABSENT, not as a value: rows written before 2026-05-08 use a
 * bare 0 as a missing sentinel, and on that date seven metrics jumped 0 → a real number
 * when the scraper was fixed. Suppressing a genuine 0.00 print is a false negative we
 * accept; lighting a false mark on a plumbing event is not.
 */
export function parseValue(raw) {
    if (raw === undefined || raw === null) return null;
    const s = String(raw).trim();
    if (!s || s.toUpperCase() === 'N/A') return null;
    if (!/^-?[\d,]*\.?\d+$/.test(s.replace(/\s/g, ''))) return null;
    const n = Number(s.replace(/,/g, ''));
    if (!Number.isFinite(n)) return null;
    return Math.abs(n) < EPS ? null : n;
}

/**
 * Today's date in New York as `YYYY-MM-DD`.
 *
 * MUST NOT be `new Date().toISOString()`. This runs on Vercel in UTC while the sheet's
 * Date column is stamped by a scraper on an ET cron — between 8pm and midnight ET the
 * UTC date is already tomorrow, which would silently select the wrong baseline every
 * evening. `en-CA` formats as YYYY-MM-DD, which sorts lexically against sheet dates.
 */
export function todayET(now = new Date()) {
    return new Intl.DateTimeFormat('en-CA', {
        timeZone: 'America/New_York',
        year: 'numeric', month: '2-digit', day: '2-digit',
    }).format(now);
}

/**
 * True when a transition is a units change rather than an economic move.
 *
 * On 2026-03-18 the scraper switched conventions: claims 212000 → 212, housing
 * 1487000 → 1487. Only ratios near a power of 1000 are rejected — a blanket
 * "large ratio" rule would suppress real prints on series that live near zero.
 */
export function isUnitJump(a, b) {
    if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
    const lo = Math.min(Math.abs(a), Math.abs(b));
    const hi = Math.max(Math.abs(a), Math.abs(b));
    if (lo < EPS) return false;
    const r = hi / lo;
    return (r >= 900 && r <= 1100) || (r >= 900000 && r <= 1100000);
}

const daysBetween = (a, b) =>
    Math.round((Date.parse(`${b}T00:00:00Z`) - Date.parse(`${a}T00:00:00Z`)) / 86400000);

/**
 * One column of the CSV as a clean, date-ordered series.
 * Duplicate dates collapse to the LAST row for that date (the scraper runs twice
 * daily and does not dedupe), and absent values are dropped rather than carried.
 */
export function buildSeries(rows, col) {
    const byDate = new Map();
    for (const r of rows) {
        const date = (r[0] || '').trim();
        if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue;
        const v = parseValue(r[col]);
        if (v === null) { byDate.delete(date); continue; }
        byDate.set(date, v);
    }
    return [...byDate.entries()]
        .sort((a, b) => (a[0] < b[0] ? -1 : 1))
        .map(([date, value]) => ({ date, value }));
}

/** Trim a series to the trailing window, so the walk-back can never reach a dead schema. */
function windowed(series, today) {
    const cutoff = new Date(Date.parse(`${today}T00:00:00Z`) - WINDOW_DAYS * 86400000)
        .toISOString().slice(0, 10);
    return series.filter(p => p.date >= cutoff && p.date <= today);
}

/**
 * Everything derivable from HISTORY ALONE for one metric.
 *
 * Deliberately does NOT decide whether a mark fires. The sheet is a snapshot written
 * at 10am/10pm ET while the dashboard renders LIVE values, so history knows only what
 * the number WAS. The comparison happens in `markFor`, where the live value is known —
 * which is also why a print landing at 8:30am is marked immediately instead of waiting
 * for the scraper to catch up.
 *
 * The baseline is the last point strictly BEFORE today, so it is yesterday's number
 * whether or not today's row has been written yet.
 *
 * @returns {{baseline,baselineDate,heldFrom,runs:number[],sigma:number|null}|null}
 */
export function historyFor(series, today) {
    const s = windowed(series || [], today);
    const prior = s.filter(p => p.date < today);
    if (!prior.length) return null;

    const bi = prior.length - 1;
    const base = prior[bi];

    // how long the baseline value had already been standing
    let hi = bi;
    while (hi > 0 && Math.abs(prior[hi - 1].value - base.value) <= EPS) hi--;

    const runs = [];
    for (const p of prior) {
        if (!runs.length || Math.abs(runs[runs.length - 1] - p.value) > EPS) runs.push(p.value);
    }

    // σ of daily moves, for the move tier only; null when there is too little to judge
    let sigma = null;
    const diffs = [];
    for (let i = 1; i < prior.length; i++) {
        if (isUnitJump(prior[i - 1].value, prior[i].value)) continue;
        diffs.push(prior[i].value - prior[i - 1].value);
    }
    if (diffs.length >= MIN_MOVES) {
        const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        const sd = Math.sqrt(diffs.reduce((a, b) => a + (b - mean) ** 2, 0) / diffs.length);
        if (sd > EPS) sigma = sd;
    }

    return {
        baseline: base.value,
        baselineDate: base.date,
        heldFrom: prior[hi].date,
        runs: runs.slice(-MAX_RUNS),
        sigma,
        dailyRuns: prior.slice(-MAX_RUNS).map(p => p.value),
    };
}

/**
 * Decide the mark for one metric by comparing the LIVE value against its history entry.
 * Returns null — no mark — for anything that is not news.
 *
 * @param {string} key      metric key (drives the print/move class)
 * @param {number} live     the value the dashboard is currently rendering
 * @param {object} entry    a `historyFor` result
 * @param {string} today    YYYY-MM-DD in America/New_York
 * @returns {{kind,prev,value,dir,heldFrom,heldDays,runs,sigma?}|null}
 */
export function markFor(key, live, entry, today) {
    const kind = classify(key);
    if (kind === 'none') return null;
    if (!entry || !Number.isFinite(live)) return null;
    const { baseline } = entry;
    if (!Number.isFinite(baseline)) return null;

    // a units change or an N/A recovery is plumbing, not news
    if (isUnitJump(baseline, live)) return null;
    const delta = live - baseline;
    if (Math.abs(delta) <= EPS) return null;

    if (kind === 'move') {
        if (!Number.isFinite(entry.sigma) || !(entry.sigma > EPS)) return null;
        if (Math.abs(delta) <= SIGMA_MULT * entry.sigma) return null;
        return {
            kind: 'move',
            value: live,
            prev: baseline,
            dir: delta > 0 ? 1 : -1,
            sigma: entry.sigma,
            move: delta,
            runs: [...(entry.dailyRuns || []), live].slice(-MAX_RUNS),
        };
    }

    return {
        kind: 'print',
        value: live,
        prev: baseline,
        dir: delta > 0 ? 1 : -1,
        heldFrom: entry.heldFrom,
        heldDays: daysBetween(entry.heldFrom, today),
        runs: [...(entry.runs || []), live].slice(-MAX_RUNS),
    };
}

/**
 * Build the digest served by /api/history: one entry per markable metric, carrying only
 * what history can know. Metrics with no usable history are simply absent.
 */
export function buildDigest(rows, now = new Date()) {
    const today = todayET(now);
    const out = {};
    for (const [key, meta] of Object.entries(SHEET_METRICS)) {
        if (meta.kind === 'none') continue;
        const entry = historyFor(buildSeries(rows, meta.col), today);
        if (!entry) continue;
        out[key] = { kind: meta.kind, label: meta.label, ...entry };
        if (meta.kind === 'print') delete out[key].sigma;      // unused, keep the payload lean
        else delete out[key].runs;
    }
    return { today, metrics: out };
}
