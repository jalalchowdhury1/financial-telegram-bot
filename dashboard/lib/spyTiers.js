/**
 * spyTiers.js — the DIRECT-source tiers beneath the Lambda for /api/spy and
 * /api/spy-daily-move. Pure of routing: each function takes the env keys, the
 * fault set and a `now`, and returns a payload or throws. The routes wrap them in
 * serve() (never-throws + last-known-good).
 *
 *   /api/spy:            Polygon bars → Nasdaq bars → Yahoo, each with a live spot
 *                        overlay (Finnhub → CNBC). A tier whose newest bar is NOT the
 *                        latest session and that has no fresh spot is held back as a
 *                        flagged-stale candidate, served only if every tier fails.
 *   /api/spy-daily-move: Finnhub → CNBC → Polygon → Yahoo. Polygon/Yahoo are SKIPPED
 *                        when their newest bar is not the latest session — Polygon's
 *                        free tier has no bar for today, so it used to report
 *                        YESTERDAY's move as today's (-0.42% vs the real +0.60%, Oct 2026).
 *
 * Fault names (`?_fail=`): polygon, finnhub, cnbc, nasdaq, yahoo.
 * Nasdaq closes are unadjusted (price return), like the Lambda's sheet tier; SPY has
 * no splits, so they line up with Polygon's split-adjusted bars to the cent.
 */

import { yahooChart, polygonDaily, finnhubQuote, cnbcQuotes, nasdaqHistory, dailyChange } from './sources';
import { calculateRSI } from './finance';
import { gate } from './faults';
import { etParts, sessionOf } from './marketClock';

const r2 = (x) => (x == null ? null : Math.round(x * 100) / 100);
const sma = (arr, i, p) => (i >= p - 1 ? arr.slice(i - p + 1, i + 1).reduce((s, v) => s + v, 0) / p : null);
const addDays = (date, n) => new Date(Date.parse(`${date}T12:00:00Z`) + n * 864e5).toISOString().slice(0, 10);
const RETURN_3Y_BARS = 756;
// A spot more than 15% off the newest bar is a vendor glitch, not a SPY move.
const plausible = (spot, ref) => Number.isFinite(spot) && spot > 0 && (!ref || Math.abs(spot / ref - 1) < 0.15);
const fmtPct = (p) => `${p >= 0 ? '+' : ''}${p.toFixed(2)}%`;
const NASDAQ_OPTS = { years: 4, revalidate: 3600, timeout: 6000 }; // one URL per day → shared Data Cache hit

/**
 * The newest NYSE session whose regular open has passed (ET): today from 9:30 on a
 * trading day, otherwise the previous trading day. A daily bar dated earlier than
 * this is a stale day.
 */
export function latestSessionDate(now = Date.now()) {
    const { date, min } = etParts(now);
    const today = sessionOf(date);
    if (today && min >= today.open) return date;
    let d = date;
    for (let i = 1; i <= 10; i++) { d = addDays(date, -i); if (sessionOf(d)) break; }
    return d;
}

/**
 * A flagged-stale build (`_meta.asOf` = its last close, YYYY-MM-DD) beats a last-good
 * copy saved on the same or an EARLIER New York day: e.g. the build holds yesterday's
 * close while the KV copy is 5 days old (or was saved intraday on that same day, before
 * the close). A copy saved on a LATER day is newer and keeps winning. The build stays
 * flagged stale either way (serve() opts.preferNewer).
 */
export function spyPreferNewer(payload, savedAt) {
    const asOf = payload?._meta?.stale ? payload._meta.asOf : null;
    const t = Date.parse(savedAt);
    if (!payload || payload.current == null || !asOf || !Number.isFinite(t)) return false;
    return asOf >= etParts(t).date;
}

/** 3Y price return from oldest->newest bars, or null when they span < 3 years (never a shorter window labelled 3Y). */
export function return3yFrom(prices, current) {
    const n = prices.length;
    const px3y = n >= RETURN_3Y_BARS ? prices[n - RETURN_3Y_BARS] : null;
    return px3y ? ((current - px3y) / px3y) * 100 : null;
}

/** Build the exact /api/spy shape from an oldest->newest [{date,price}] history.
 *  Requires enough history (>=220 trading days) so MA200/52w/RSI are real —
 *  otherwise throws, so the card never receives a broken partial object.
 *  `extra.return3y` fills the 3Y only when these bars are too short for it;
 *  `extra.meta` merges into _meta (stale flags). */
export function buildSpy(history, current, prevClose, source, extra = {}) {
    const prices = history.map((h) => h.price);
    const n = prices.length;
    if (n < 220) throw new Error(`${source}: insufficient history (${n} rows)`);
    const ma200v = sma(prices, n - 1, 200);
    const dc = dailyChange(current, prevClose);
    const last252 = prices.slice(-252);
    const wkHigh = Math.max(...last252);
    // A 3Y return needs 3Y of bars. Polygon's free tier serves ~2y, and clamping the
    // index to 0 used to show that 2-YEAR return as "3Y" (2026-09-26). null = N/A.
    const return3y = return3yFrom(prices, current) ?? extra.return3y ?? null;
    const rsi = calculateRSI(history, 9);

    const chartHistory = [];
    for (let i = Math.max(0, n - 302); i < n; i++) {
        chartHistory.push({ date: history[i].date, price: r2(prices[i]), ma50: r2(sma(prices, i, 50)), ma200: r2(sma(prices, i, 200)) });
    }

    return {
        current: r2(current),
        dailyChange: { value: dc.value, pct: dc.pct },
        ma200: { value: r2(ma200v), pct: ma200v ? ((current - ma200v) / ma200v) * 100 : 0 },
        week52High: { value: r2(wkHigh), pct: wkHigh ? ((current - wkHigh) / wkHigh) * 100 : 0 },
        rsi,
        return3y,
        chartHistory,
        // Reaching this return means a full, validated build (n>=220, all stats
        // computed) — that is healthy data, not an error. A degraded *source*
        // is conveyed by the source label; stale builds carry extra.meta.stale.
        _meta: { source, hasErrors: false, messages: [`Served from ${source}`], ...(extra.meta || {}) },
    };
}

/** CNBC SPY quote, rejected unless it belongs to the latest session. -> {current, prevClose, changePct, asOf} */
export async function cnbcSpy(expected, { revalidate = 120 } = {}) {
    const q = (await cnbcQuotes(['SPY'], { revalidate, timeout: 5000 })).SPY;
    if (!q) throw new Error('CNBC: no SPY quote');
    if (!q.asOf || q.asOf < expected) throw new Error(`CNBC quote dated ${q.asOf || '?'}, latest session ${expected}`);
    if (!Number.isFinite(q.change)) throw new Error('CNBC: no change field');
    return { current: q.price, prevClose: q.price - q.change, changePct: q.changePct, asOf: q.asOf };
}

/** Live spot for the overlay: Finnhub, then CNBC. Fetched once per request (memoized on ctx). */
function spotOnce(ctx) {
    if (!ctx.spotP) {
        ctx.spotP = (async () => {
            const { finnhubKey, faults, messages, expected } = ctx;
            if (finnhubKey) {
                try { return { ...(await gate('finnhub', faults, () => finnhubQuote('SPY', finnhubKey))), label: 'Finnhub' }; }
                catch (e) { messages.push(`Finnhub spot failed: ${e.message}`); }
            }
            try { return { ...(await gate('cnbc', faults, () => cnbcSpy(expected))), label: 'CNBC' }; }
            catch (e) { messages.push(`CNBC spot failed: ${e.message}`); }
            return null;
        })();
    }
    return ctx.spotP;
}

/**
 * Bars -> payload. With a plausible live spot: bars + spot. Without one, the bars'
 * own last close is live ONLY if dated the latest session; otherwise the build is
 * returned as `stale` (relabelled + _meta.stale) for use as a last resort.
 */
async function fromBars(label, history, ctx, extra = {}) {
    const n = history.length;
    const last = history[n - 1];
    const spot = await spotOnce(ctx);
    if (spot && plausible(spot.current, last.price)) {
        return { payload: buildSpy(history, spot.current, spot.prevClose, `${label} + ${spot.label} (fallback)`, extra) };
    }
    if (spot) ctx.messages.push(`${spot.label} spot ${spot.current} implausible vs ${label} close ${last.price}; ignored`);
    const prev = history[n - 2]?.price ?? last.price;
    if (last.date >= ctx.expected) return { payload: buildSpy(history, last.price, prev, `${label} (fallback)`, extra) };
    ctx.messages.push(`${label} newest bar ${last.date} is not the latest session ${ctx.expected}`);
    return {
        stale: buildSpy(history, last.price, prev, `${label} (fallback, last close ${last.date})`, {
            ...extra,
            meta: { stale: true, hasErrors: true, asOf: last.date },
        }),
    };
}

/** 3Y return for a short (Polygon ~2y) series, from Nasdaq bars that line up with it. null if they don't. */
async function nasdaq3y(current, lastBar, ctx) {
    try {
        const h = await gate('nasdaq', ctx.faults, () => nasdaqHistory('SPY', NASDAQ_OPTS));
        const upto = h.filter((b) => b.date <= lastBar.date);
        const tail = upto[upto.length - 1];
        if (!tail || tail.date !== lastBar.date || Math.abs(tail.price / lastBar.price - 1) > 0.005) {
            throw new Error(`bars do not line up (${tail?.date} ${tail?.price} vs ${lastBar.date} ${lastBar.price})`);
        }
        const r = return3yFrom(upto.map((b) => b.price), current);
        if (r == null) throw new Error(`only ${upto.length} bars`);
        return r;
    } catch (e) { ctx.messages.push(`3Y from Nasdaq unavailable: ${e.message}`); return null; }
}

/** Yahoo prevClose from bars: chartPreviousClose is the close BEFORE THE RANGE (5 years ago on range=5y). */
function yahooPrev(y) {
    const h = y.history, n = h.length;
    const mt = y.meta?.regularMarketTime;
    const quoteDay = mt ? etParts(mt * 1000).date : h[n - 1].date;
    return quoteDay === h[n - 1].date ? (h[n - 2]?.price ?? h[n - 1].price) : h[n - 1].price;
}

/**
 * /api/spy direct tiers: Polygon(+spot) → Nasdaq(+spot) → Yahoo → stale candidate → throw.
 * `messages` collects the reasons each tier was passed over.
 */
export async function fallbackSpy(messages, faults = new Set(), { env = process.env, now = Date.now() } = {}) {
    const ctx = { faults, messages, expected: latestSessionDate(now), finnhubKey: env.FINNHUB_KEY || '' };
    let stale = null;
    const keep = (r) => { if (r.payload) return r.payload; stale = stale || r.stale; return null; };

    // 1) Polygon (server-friendly; free tier ≈ 2y, so the 3Y comes from Nasdaq bars).
    if (env.POLYGON_KEY) {
        try {
            const p = await gate('polygon', faults, () => polygonDaily('SPY', env.POLYGON_KEY, { years: 5, revalidate: 1800 }));
            const lastBar = p.history[p.history.length - 1];
            const extra = {};
            if (p.history.length < RETURN_3Y_BARS) {
                const spot = await spotOnce(ctx);
                const cur = spot && plausible(spot.current, lastBar.price) ? spot.current : lastBar.price;
                extra.return3y = await nasdaq3y(cur, lastBar, ctx);
            }
            const out = keep(await fromBars('Polygon', p.history, ctx, extra));
            if (out) return out;
        } catch (e) { messages.push(`Polygon fallback failed: ${e.message}`); }
    } else {
        messages.push('POLYGON_KEY not configured');
    }

    // 2) Nasdaq (KEYLESS, reachable from Vercel; ~4y of daily closes → full 3Y).
    try {
        const h = await gate('nasdaq', faults, () => nasdaqHistory('SPY', NASDAQ_OPTS));
        const out = keep(await fromBars('Nasdaq', h, ctx));
        if (out) return out;
    } catch (e) { messages.push(`Nasdaq fallback failed: ${e.message}`); }

    // 3) Yahoo 5y — best effort (429s from Vercel/cloud IPs; fine locally).
    try {
        const y = await gate('yahoo', faults, () => yahooChart('SPY', { range: '5y', interval: '1d', revalidate: 300 }));
        const last = y.history[y.history.length - 1];
        const quoteDay = y.meta?.regularMarketTime ? etParts(y.meta.regularMarketTime * 1000).date : last.date;
        if (quoteDay >= ctx.expected) return buildSpy(y.history, y.current, yahooPrev(y), 'Yahoo Finance (fallback)');
        messages.push(`Yahoo quote dated ${quoteDay} is not the latest session ${ctx.expected}`);
        stale = stale || buildSpy(y.history, y.current, yahooPrev(y), `Yahoo Finance (fallback, last close ${quoteDay})`, { meta: { stale: true, hasErrors: true, asOf: quoteDay } });
    } catch (e) { messages.push(`Yahoo fallback failed: ${e.message}`); }

    // 4) Every tier failed to give the latest session: an honest, flagged older close
    //    beats a blank card. Never stored as last-known-good (see the route's shouldStore).
    if (stale) return stale;
    throw new Error(`all SPY tiers failed: ${messages.slice(-3).join(' | ')}`);
}

/**
 * /api/spy-daily-move direct tiers: Finnhub → CNBC → Polygon → Yahoo. A bar-based tier
 * whose newest bar is not the latest session is skipped (it would show an old day's move).
 */
export async function fallbackMove(messages, faults = new Set(), { env = process.env, now = Date.now() } = {}) {
    const expected = latestSessionDate(now);
    if (env.FINNHUB_KEY) {
        try {
            const q = await gate('finnhub', faults, () => finnhubQuote('SPY', env.FINNHUB_KEY));
            return { value: fmtPct(dailyChange(q.current, q.prevClose).pct), source: 'Finnhub (fallback)' };
        } catch (e) { messages.push(`Finnhub failed: ${e.message}`); }
    }
    try {
        const q = await gate('cnbc', faults, () => cnbcSpy(expected));
        const pct = Number.isFinite(q.changePct) ? q.changePct : dailyChange(q.current, q.prevClose).pct;
        return { value: fmtPct(pct), source: 'CNBC (fallback)', asOf: q.asOf };
    } catch (e) { messages.push(`CNBC failed: ${e.message}`); }
    if (env.POLYGON_KEY) {
        try {
            const p = await gate('polygon', faults, () => polygonDaily('SPY', env.POLYGON_KEY, { years: 1, revalidate: 600 }));
            const last = p.history[p.history.length - 1];
            if (last.date < expected) throw new Error(`newest bar ${last.date} is not the latest session ${expected}; skipped`);
            return { value: fmtPct(dailyChange(p.current, p.prevClose).pct), source: 'Polygon (fallback)', asOf: last.date };
        } catch (e) { messages.push(`Polygon failed: ${e.message}`); }
    }
    try {
        const y = await gate('yahoo', faults, () => yahooChart('SPY', { range: '5d', interval: '1d', revalidate: 300 }));
        const last = y.history[y.history.length - 1];
        const quoteDay = y.meta?.regularMarketTime ? etParts(y.meta.regularMarketTime * 1000).date : last.date;
        if (quoteDay < expected) throw new Error(`quote dated ${quoteDay} is not the latest session ${expected}; skipped`);
        return { value: fmtPct(dailyChange(y.current, yahooPrev(y)).pct), source: 'Yahoo Finance (fallback)', asOf: quoteDay };
    } catch (e) { messages.push(`Yahoo failed: ${e.message}`); }
    throw new Error(`all SPY move tiers failed: ${messages.slice(-2).join(' | ')}`);
}
