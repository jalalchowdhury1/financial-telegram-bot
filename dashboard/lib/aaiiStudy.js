/**
 * AAII bears − bulls at 20+: what SPY did next. Pure, so the v2 drill-in and its test share it.
 *
 * An episode starts at the first 20+ survey after 3 quiet months (no 20+ reading in the 91 days
 * before), so a long fearful run counts once. Its path is SPY's close (price only, no dividends)
 * from the first trading day on/after the survey to `months` later. The base rate is the same
 * window started on every trading day, so "after a 20+ reading" has something honest to beat.
 */
const DAY = 86400000;
const ms = (iso) => Date.parse(`${iso}T00:00:00Z`);
const QUIET = 91 * DAY;
const GRID = 60; // average-path resolution

export const plusMonthsIso = (iso, n) => { const d = new Date(`${iso}T12:00:00Z`); d.setUTCMonth(d.getUTCMonth() + n); return d.toISOString().slice(0, 10); };

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

/** The readings that started a 20+ episode, from `aaii` [{date, value}]. */
export function aaiiEpisodes(aaii, line = 20) {
    const eps = [];
    let last = -Infinity;
    for (const p of aaii) {
        if (!(p.value >= line)) continue;
        if (ms(p.date) - last > QUIET) eps.push({ date: p.date, value: p.value });
        last = ms(p.date);
    }
    return eps;
}

/**
 * `{ episodes, stats, base, avgPath }` for one horizon. Each episode: `{date, value, ret, low, live,
 * path:[{x 0..1, r}]}`; `ret`/`low` are SPY's change at the end of the window (so far, when live)
 * and its worst point inside it. Live episodes stay out of `stats` and `avgPath`.
 */
export function aaiiStudy(aaii, spy, months, line = 20) {
    if (!aaii?.length || !spy?.length) return null;
    const lastSpy = spy[spy.length - 1].date;
    const episodes = aaiiEpisodes(aaii, line).map((e) => {
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
    for (let i = 0, j = 0; i < spy.length; i++) {
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
