/**
 * 🧬 Factor row — which investing STYLE is the market paying for right now?
 *
 * Each factor is an iShares style ETF measured AGAINST the S&P 500 (SPY): the
 * number is how much more (or less) $1 in the factor ETF is worth than $1 in SPY
 * over the window, and the sparkline is that same ratio over time (a StockCharts-
 * style ratio chart, normalised to 0 at the window start). Price only — no
 * keyless source serves dividends from a datacenter (Yahoo adjclose is 429 from
 * Vercel), so this is labelled "price ratio" in the UI. Dividend gaps between
 * these ETFs are ~0.1 pts/month, ~1 pt/yr.
 *
 * This file is PURE (math + the per-ticker cascade with injected fetchers) so it
 * is fully unit-testable; app/api/factors/route.js wires the real sources.
 *
 * Data per ticker = LONG history (≈10y, weekly or daily) spliced under RECENT
 * daily bars (≈2y). CNBC's weekly bars are dated by the week's SUNDAY but carry
 * the FRIDAY close (verified 105/105 against daily closes), and the current
 * week's bar is an in-progress snapshot — so weekly dates are shifted +5 days
 * and only the part OLDER than the daily series is ever used.
 */

export const BENCH = 'SPY';

export const FACTORS = [
    { key: 'value', label: 'Value', short: 'Value', ticker: 'VLUE', what: 'cheap stocks — low price compared with earnings and assets' },
    { key: 'momentum', label: 'Momentum', short: 'Mom.', ticker: 'MTUM', what: 'stocks that have been rising tend to keep rising' },
    { key: 'quality', label: 'Quality', short: 'Quality', ticker: 'QUAL', what: 'profitable companies with steady earnings and little debt' },
    { key: 'size', label: 'Small caps', short: 'Size', ticker: 'IWM', what: 'small companies (Russell 2000) versus the giants' },
    { key: 'lowvol', label: 'Low vol', short: 'Low vol', ticker: 'USMV', what: 'the calmest, least jumpy stocks' },
];

export const TICKERS = [BENCH, ...FACTORS.map((f) => f.ticker)];

export const WINDOWS = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', '10Y'];
const MONTHS = { '1M': 1, '3M': 3, '6M': 6, '1Y': 12, '3Y': 36, '5Y': 60, '10Y': 120 };

/** A daily series older than this (calendar days) is stale: Fri close read on a Tue after a 3-day weekend = 4. */
export const FRESH_DAYS = 5;
/** A window may start this many days after its nominal start (weekly bars, first-bar offsets). */
const START_TOLERANCE_DAYS = 10;
/** Sparkline resolution: windows with more points than MAX_POINTS are resampled to SPARK_SLOTS by TIME. */
const MAX_POINTS = 64;
const SPARK_SLOTS = 60;
/** No ETF here has moved >18% in a week in 10 years; a bigger bar-to-bar jump is a split/bad print. */
const MAX_BAR_MOVE = 0.35;

const DAY_MS = 864e5;
const toMs = (iso) => Date.parse(`${iso}T00:00:00Z`);
const toIso = (ms) => new Date(ms).toISOString().slice(0, 10);
const round2 = (x) => Math.round(x * 100) / 100;

export function daysBetween(aIso, bIso) {
    return Math.round((toMs(bIso) - toMs(aIso)) / DAY_MS);
}

/** Today's date in New York (the market's calendar), as YYYY-MM-DD. */
export function todayET(now = new Date()) {
    return new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(now);
}

/**
 * Nominal start date of a window ending at `asOf`. YTD starts at the prior year's
 * last day (so the first bar used is the last close of that year). Month
 * arithmetic clamps the day (Mar 31 − 1M = Feb 28/29).
 */
export function windowStart(asOf, key) {
    const [y, m, d] = asOf.split('-').map(Number);
    if (key === 'YTD') return `${y - 1}-12-31`;
    const months = MONTHS[key];
    if (!months) return null;
    const total = y * 12 + (m - 1) - months;
    const ty = Math.floor(total / 12);
    const tm = total % 12; // 0-based
    const lastDay = new Date(Date.UTC(ty, tm + 1, 0)).getUTCDate();
    return toIso(Date.UTC(ty, tm, Math.min(d, lastDay)));
}

/**
 * CNBC weekly bars → Friday-dated bars. Drops every bar whose Friday is within the
 * last 7 days: the newest weekly bar is a moving snapshot, and it can stay WRONG
 * after the week ends (IWM's Sep-25-2026 bar read 285.58 on the Saturday, while the
 * Friday close was 281.97). Normally only the long, older part of this series is
 * used; if it ever has to stand in for the daily series, a week of lag flagged
 * stale beats a fresh-looking wrong number.
 */
export function weeklyToFriday(history, today) {
    if (!Array.isArray(history)) return [];
    const cutoff = today ? toIso(toMs(today) - 7 * DAY_MS) : null;
    const out = [];
    for (const p of history) {
        const fri = toIso(toMs(p.date) + 5 * DAY_MS);
        if (cutoff && fri > cutoff) continue;
        out.push({ date: fri, price: p.price });
    }
    return out;
}

/** Collapse a daily series to the last trading day of each ISO week (used for the bake). */
export function thinToWeekly(history) {
    if (!Array.isArray(history)) return [];
    const out = [];
    let lastWeek = null;
    for (const p of history) {
        const ms = toMs(p.date);
        const dow = (new Date(ms).getUTCDay() + 6) % 7; // Mon=0
        const week = toIso(ms - dow * DAY_MS);
        if (week === lastWeek) out[out.length - 1] = p;
        else { out.push(p); lastWeek = week; }
    }
    return out;
}

/**
 * A usable price series: ascending unique dates, finite positive prices, enough
 * points, and no bar-to-bar move a real ETF cannot make (split / bad print).
 */
export function validSeries(history, minPoints = 20) {
    if (!Array.isArray(history) || history.length < minPoints) return false;
    for (let i = 0; i < history.length; i++) {
        const p = history[i];
        if (!p || typeof p.date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(p.date)) return false;
        if (!Number.isFinite(p.price) || p.price <= 0) return false;
        if (i > 0) {
            if (p.date <= history[i - 1].date) return false;
            if (Math.abs(p.price / history[i - 1].price - 1) > MAX_BAR_MOVE) return false;
        }
    }
    return true;
}

/** Splice: long history strictly OLDER than the recent series, then the recent series. */
export function mergeSeries(long, recent) {
    if (!Array.isArray(recent) || !recent.length) return Array.isArray(long) ? long.slice() : [];
    if (!Array.isArray(long) || !long.length) return recent.slice();
    const cut = recent[0].date;
    return long.filter((p) => p.date < cut).concat(recent);
}

/** Align factor + benchmark on common dates → ascending [{date, f, b}]. */
export function align(fHist, bHist) {
    if (!Array.isArray(fHist) || !Array.isArray(bHist)) return [];
    const b = new Map();
    for (const p of bHist) if (Number.isFinite(p.price) && p.price > 0) b.set(p.date, p.price);
    const out = [];
    for (const p of fHist) {
        const bp = b.get(p.date);
        if (bp && Number.isFinite(p.price) && p.price > 0) out.push({ date: p.date, f: p.price, b: bp });
    }
    out.sort((x, y) => (x.date < y.date ? -1 : 1));
    return out;
}

/**
 * Relative performance of factor vs benchmark over one window.
 * @returns {{rel:number, f:number, b:number, from:string, to:string, spark:number[]}|null}
 *   rel/f/b in % (2dp). rel = (f_end/b_end)/(f_start/b_start) − 1 — "$1 in the
 *   factor vs $1 in SPY". spark = that ratio over time, in %, 0 at the start.
 */
export function computeWindow(series, key) {
    if (!Array.isArray(series) || series.length < 3) return null;
    const last = series[series.length - 1];
    const start = windowStart(last.date, key);
    if (!start) return null;
    let i0 = -1;
    for (let i = series.length - 1; i >= 0; i--) {
        if (series[i].date <= start) { i0 = i; break; }
    }
    if (i0 === -1) {
        // History begins after the nominal start: allow a small offset, else the window is not covered.
        if (daysBetween(start, series[0].date) > START_TOLERANCE_DAYS) return null;
        i0 = 0;
    }
    const pts = series.slice(i0);
    if (pts.length < 3) return null;
    const r0 = pts[0].f / pts[0].b;
    const relAt = (p) => ((p.f / p.b) / r0 - 1) * 100;

    let spark;
    if (pts.length <= MAX_POINTS) {
        spark = pts.map((p) => round2(relAt(p)));
    } else {
        // Resample by TIME, not index: a 10Y window is weekly for 8 years then daily
        // for 2 — by index the last 2 years would take over half the chart.
        const t0 = toMs(pts[0].date);
        const tN = toMs(last.date);
        spark = [];
        let j = 0;
        for (let k = 0; k < SPARK_SLOTS; k++) {
            const t = t0 + ((tN - t0) * k) / (SPARK_SLOTS - 1);
            while (j + 1 < pts.length && toMs(pts[j + 1].date) <= t) j++;
            spark.push(round2(relAt(pts[j])));
        }
        spark[spark.length - 1] = round2(relAt(last));
    }
    return {
        rel: round2(relAt(last)),
        f: round2((last.f / pts[0].f - 1) * 100),
        b: round2((last.b / pts[0].b - 1) * 100),
        from: pts[0].date,
        to: last.date,
        spark,
    };
}

/**
 * Resolve one ticker through the cascade. Every fetcher is injected (tests +
 * route). RECENT tiers must be fresh to win; a stale one is kept only as a
 * candidate in case nothing fresher answers. LONG tiers only need validity —
 * only their part older than the recent series is used. The baked tier never
 * fails, so a ticker always resolves to SOMETHING (flagged stale if old).
 *
 * @param {string} ticker
 * @param {object} o
 * @param {Array<{name:string, fn:(t:string)=>Promise<Array>}>} o.recent  daily tiers, in order
 * @param {Array<{name:string, fn:(t:string)=>Promise<Array>}>} o.long    long-history tiers, in order
 * @param {(t:string)=>Array|null} [o.baked]   sync, never-throws last tier (weekly)
 * @param {string} o.today                    YYYY-MM-DD (New York)
 * @param {number} [o.deadline]               epoch ms; later tiers are skipped past it
 * @returns {Promise<{ticker, history, asOf, stale, recentSource, longSource, tried:string[]}>}
 */
export async function resolveTicker(ticker, { recent = [], long = [], baked = null, today, deadline = Infinity }) {
    const tried = [];
    const late = () => Date.now() > deadline;
    const attempt = async (tier, minPoints) => {
        if (late()) { tried.push(`${tier.name}:deadline`); return null; }
        try {
            const h = await tier.fn(ticker);
            if (!validSeries(h, minPoints)) { tried.push(`${tier.name}:invalid`); return null; }
            return h;
        } catch (e) {
            tried.push(`${tier.name}:${/fault/i.test(e?.message || '') ? 'fault' : 'err'}`);
            return null;
        }
    };

    let rec = null;
    let recentSource = null;
    for (const tier of recent) {
        const h = await attempt(tier, 60);
        if (!h) continue;
        const age = daysBetween(h[h.length - 1].date, today);
        if (!rec || h[h.length - 1].date > rec[rec.length - 1].date) { rec = h; recentSource = tier.name; }
        if (age <= FRESH_DAYS) break;
        tried.push(`${tier.name}:stale`);
    }

    // Long history is only needed when the recent series does not already reach back ~10y.
    let lng = null;
    let longSource = null;
    const needLong = !rec || daysBetween(rec[0].date, today) < 10 * 365;
    if (needLong) {
        for (const tier of long) {
            const h = await attempt(tier, 100);
            if (h) { lng = h; longSource = tier.name; break; }
        }
        if (!lng && baked) {
            try {
                const h = baked(ticker);
                if (validSeries(h, 20)) { lng = h; longSource = 'baked'; }
            } catch { /* never throws */ }
        }
    }

    let history = mergeSeries(lng, rec);
    // Each side passed validSeries on its own, but the SPLICE can still hide a split
    // (a bake from before a split under split-adjusted live bars would put 3Y/5Y/10Y
    // off by the split ratio). A jump beyond MAX_BAR_MOVE at the seam → drop the long part.
    if (lng && rec && history.length > rec.length) {
        const seam = history.length - rec.length;
        if (Math.abs(history[seam].price / history[seam - 1].price - 1) > MAX_BAR_MOVE) {
            tried.push(`${longSource}:splice-jump`);
            history = rec.slice();
            lng = null;
            longSource = null;
        }
    }
    if (!history.length) return { ticker, history: [], asOf: null, stale: true, recentSource, longSource, tried };
    const asOf = history[history.length - 1].date;
    return {
        ticker,
        history,
        asOf,
        stale: daysBetween(asOf, today) > FRESH_DAYS,
        recentSource: recentSource || (longSource ? `${longSource}(only)` : null),
        longSource,
        tried,
    };
}

/**
 * Build the route payload from resolved tickers. Throws only when the benchmark
 * itself has no data (then nothing is computable and serve() falls back).
 *
 * @param {Record<string, ReturnType<typeof resolveTicker> extends Promise<infer R> ? R : never>} resolved
 * @param {{primary?:{recent:string,long:string}, bakedAt?:string}} [info]
 */
export function buildPayload(resolved, { primary = { recent: 'cnbc', long: 'cnbc-weekly' }, bakedAt = null } = {}) {
    const bench = resolved[BENCH];
    if (!bench || !bench.history || bench.history.length < 3) throw new Error(`benchmark ${BENCH} unavailable`);

    const messages = [];
    const factors = [];
    let fallbackUsed = bench.recentSource !== primary.recent || (bench.longSource && bench.longSource !== primary.long);
    let allBaked = bench.recentSource === 'baked(only)';

    for (const f of FACTORS) {
        const r = resolved[f.ticker];
        const series = align(r?.history, bench.history);
        if (series.length < 3) {
            messages.push(`${f.ticker}: unavailable (${(r?.tried || []).join(', ') || 'no data'})`);
            continue;
        }
        if (r.recentSource !== primary.recent || (r.longSource && r.longSource !== primary.long)) fallbackUsed = true;
        if (r.recentSource !== 'baked(only)') allBaked = false;
        const windows = {};
        for (const w of WINDOWS) windows[w] = computeWindow(series, w);
        factors.push({
            key: f.key, label: f.label, short: f.short, ticker: f.ticker, what: f.what,
            asOf: series[series.length - 1].date,
            stale: !!(r.stale || bench.stale),
            windows,
        });
    }

    const stale = !!bench.stale || factors.some((x) => x.stale);
    const sourceOf = (r) => `${r.ticker}:${r.recentSource || 'none'}${r.longSource ? `+${r.longSource}` : ''}`;
    for (const t of TICKERS) {
        const r = resolved[t];
        if (r?.tried?.length) messages.push(`${t} tried: ${r.tried.join(', ')}`);
    }
    return {
        asOf: bench.asOf,
        bench: BENCH,
        basis: 'price',
        windows: WINDOWS,
        factors,
        _meta: {
            source: TICKERS.map((t) => resolved[t]).filter(Boolean).map(sourceOf).join(' · '),
            hasErrors: factors.length < FACTORS.length || stale,
            stale,
            fallback: !!fallbackUsed,
            allBaked: !!allBaked && factors.length > 0,
            bakedAt,
            messages,
        },
    };
}

/** Servable: at least 3 of 5 factors, and not purely the baked floor. */
export function isGoodPayload(p) {
    return !!p && Array.isArray(p.factors) && p.factors.length >= 3 && !p._meta?.allBaked;
}

/**
 * What serve() treats as a live win. A STALE live payload (e.g. SPY fell to the bake
 * while the factors are live, so every window ends at the bake date) must not beat a
 * fresher /tmp or KV copy; serve() still returns it when no cache exists.
 */
export function isFreshGoodPayload(p) {
    return isGoodPayload(p) && !p._meta?.stale;
}

/** Worth storing as last-known-good: servable AND fresh (a stale payload must not refresh the cache's savedAt). */
export function isStorablePayload(p) {
    return isGoodPayload(p) && !p._meta?.stale;
}
