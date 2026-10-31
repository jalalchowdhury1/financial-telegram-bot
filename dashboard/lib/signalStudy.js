/**
 * A signal at or past its line (AAII bears − bulls 20+, VIX ÷ VIX 3M 1+): what SPY did next. Pure,
 * so the v2 drill-in (components/SignalStudy.js) and its test share it.
 *
 * An episode starts at the first reading at/past the line after 3 quiet months (none in the 91 days
 * before), so a long fearful run counts once. Its path is SPY's close (price only, no dividends)
 * from the first trading day on/after the reading to `months` later. The base rate is the same
 * window started on every trading day the signal has data for, so "after a signal" has something honest to beat.
 */
const DAY = 86400000;
const ms = (iso) => Date.parse(`${iso}T00:00:00Z`);
const QUIET = 91 * DAY;
const GRID = 60; // average-path resolution

export const plusMonthsIso = (iso, n) => {
    const [y, m, d] = iso.split('-').map(Number), last = new Date(Date.UTC(y, m - 1 + n + 1, 0)).getUTCDate(); // stops at month end: Aug 31 + 6 = Feb 28
    return new Date(Date.UTC(y, m - 1 + n, Math.min(d, last))).toISOString().slice(0, 10);
};

/** First index with date ≥ iso (pts.length when none). */
function from(pts, iso) {
    let lo = 0, hi = pts.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (pts[m].date < iso) lo = m + 1; else hi = m; }
    return lo;
}

function stats(rets) {
    if (!rets.length) return null;
    const s = [...rets].sort((a, b) => a - b), n = s.length;
    return {
        n,
        avg: rets.reduce((a, b) => a + b, 0) / n,
        median: n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2,
        win: rets.filter((r) => r > 0).length / n,
    };
}

/** The readings that started an episode, from `points` [{date, value}]. */
export function signalEpisodes(points, line) {
    const eps = [];
    let last = -Infinity;
    for (const p of points) {
        if (!(p.value >= line)) continue;
        if (ms(p.date) - last > QUIET) eps.push({ ...p });
        last = ms(p.date);
    }
    return eps;
}

/**
 * Lead combos: two dashboard stats that, past their lines together, led to a good next 6 months. A
 * 2026-10-31 search of every baked stat (level, 3-month and 1-year change by date; ~160k pairs, lines from
 * pre-2010 data, then a 2010+ re-test, a time-shift placebo and the lines around each; notes in the commit)
 * kept these. `key` = the sheet metric and /history file; `days` = a change over that many days (`pct` = in %,
 * else in points), none = the level.
 */
export const COMBOS = [
    { id: 'fo', name: 'Fear + oil crash', tip: 'AAII 20+ while oil is down 15%+ on the year.',
        legs: [{ key: 'aaiiDiff', label: 'AAII', line: 20 }, { key: 'cl', label: 'Oil 1y', days: 365, pct: true, line: -15 }] },
    { id: 'cu', name: 'Jobs fine + CAD slump', tip: 'Jobless claims down 6%+ while USD/CAD is up 3%+, both in 3 months.',
        legs: [{ key: 'claims', label: 'Claims 3m', days: 91, pct: true, line: -6 }, { key: 'usdcad', label: 'USD/CAD 3m', days: 91, pct: true, line: 3 }] },
    { id: 'hv', name: 'Panic + housing boom', tip: 'VIX 25+ while housing starts are up 8%+ in 3 months.',
        legs: [{ key: 'vixCurrent', label: 'VIX', line: 25 }, { key: 'housing', label: 'Housing 3m', days: 91, pct: true, line: 8 }] },
    { id: 'pt', name: 'Selloff + falling rates', tip: 'P/E down 6.5%+ and the 10Y down 0.4+ pts, both in 3 months.',
        legs: [{ key: 'peRatio', label: 'P/E 3m', days: 91, pct: true, line: -6.5 }, { key: 'tnx', label: '10Y 3m', days: 91, line: -0.4 }] },
];
const STALE = 45 * DAY; // a reading still counts this long (monthly stats print about monthly)

/** A leg on its own dates: the reading, or its change vs the reading `days` earlier (by date, never by row). */
function legSeries(pts, { days, pct }) {
    if (!days) return pts.filter((p) => Number.isFinite(p.value));
    const out = [];
    let k = 0;
    for (const p of pts) {
        const ago = ms(p.date) - days * DAY;
        while (k < pts.length - 1 && ms(pts[k + 1].date) <= ago) k++;
        const b = pts[k];
        if (!(ms(b.date) <= ago && ago - ms(b.date) <= STALE && Number.isFinite(p.value) && Number.isFinite(b.value))) continue;
        out.push({ date: p.date, value: pct ? (p.value / b.value - 1) * 100 : p.value - b.value });
    }
    return out;
}

/**
 * One combo as one series for the study: on every date any leg has, `value` = the LESSER of each leg ÷ its
 * line (1+ = both past their lines; clamped −1..2 for the charts) and `legs` = each leg's reading.
 * `series` = each leg's points, in the combo's order.
 */
export function comboSeries(combo, ...series) {
    const L = combo.legs.map((l, i) => legSeries(series[i] || [], l));
    const dates = [...new Set(L.flatMap((s) => s.map((p) => p.date)))].sort();
    const at = L.map(() => 0), out = [];
    for (const d of dates) {
        const legs = L.map((s, i) => {
            while (at[i] < s.length - 1 && s[at[i] + 1].date <= d) at[i]++;
            const p = s[at[i]];
            return p && p.date <= d && ms(d) - ms(p.date) <= STALE ? p.value : null;
        });
        if (legs.includes(null)) continue;
        out.push({ date: d, value: Math.max(-1, Math.min(2, ...legs.map((v, i) => v / combo.legs[i].line))), legs });
    }
    return out;
}

/**
 * `{ episodes, stats, base, avgPath }` for one horizon. Each episode: `{date, value, ret, low, live,
 * path:[{x 0..1, r}]}`; `ret`/`low` are SPY's change at the end of the window (so far, when live)
 * and its worst point inside it. Live episodes stay out of `stats` and `avgPath`.
 */
export function signalStudy(points, spy, months, line) {
    if (!points?.length || !spy?.length) return null;
    const lastSpy = spy[spy.length - 1].date;
    const episodes = signalEpisodes(points, line).map((e) => {
        const i0 = from(spy, e.date);
        if (e.date < spy[0].date || i0 >= spy.length) return null; // SPY starts 1993
        const end = plusMonthsIso(e.date, months), t0 = ms(spy[i0].date), T = ms(end) - t0;
        const p0 = spy[i0].price, path = [];
        let low = 0;
        for (let i = i0; i < spy.length && spy[i].date <= end; i++) {
            const r = spy[i].price / p0 - 1;
            low = Math.min(low, r);
            path.push({ x: (ms(spy[i].date) - t0) / T, r });
        }
        const step = Math.ceil(path.length / 90); // ≤ ~90 points a line
        return {
            ...e, live: lastSpy < end, ret: path[path.length - 1].r, low,
            path: path.filter((_, k) => k % step === 0 || k === path.length - 1),
        };
    }).filter(Boolean);

    const done = episodes.filter((e) => !e.live);
    const base = [];
    for (let i = from(spy, points[0].date), j = i; i < spy.length; i++) { // the signal's own years
        const end = plusMonthsIso(spy[i].date, months);
        if (end > lastSpy) break;
        while (j < spy.length - 1 && spy[j + 1].date <= end) j++;
        base.push(spy[j].price / spy[i].price - 1);
    }
    const avgPath = done.length ? Array.from({ length: GRID + 1 }, (_, g) => {
        const x = g / GRID;
        const rs = done.map(({ path }) => {
            const k = path.findIndex((p) => p.x >= x);
            if (k <= 0) return k === 0 ? path[0].r : path[path.length - 1].r;
            const a = path[k - 1], b = path[k];
            return a.r + ((b.r - a.r) * (x - a.x)) / (b.x - a.x || 1);
        });
        return { x, r: rs.reduce((s, r) => s + r, 0) / rs.length };
    }) : [];
    return { episodes, stats: stats(done.map((e) => e.ret)), base: stats(base), avgPath };
}
