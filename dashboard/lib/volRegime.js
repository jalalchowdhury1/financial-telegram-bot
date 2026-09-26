/**
 * volRegime.js — the lower half of the 🌡️ Volatility card (added 2026-09-26). The owner
 * reads only the SPY + QQQ rows of the table; this half turns the same vol data into
 * three numbers he can act on:
 *
 *   1. VIX CURVE — 9-day, 1-month (VIX), 3-month, 6-month implied vol, and a
 *      Calm / Watch / Stress call from VIX ÷ VIX3M. Near-term fear above 3-month fear
 *      ("backwardation") is the shape of selloffs; the calm shape slopes upward.
 *   2. TQQQ DECAY — what 3× daily leverage costs at today's QQQ volatility: if QQQ ends a
 *      year flat, TQQQ ends ≈ 1 − e^(−3σ²) lower (before fees). (L²−L)/2 = 3 for L = 3.
 *   3. EXPECTED MOVE — SPY / QQQ ±1σ over the next 5 trading days, from VIX / VXN:
 *      IV × √(5/252). About 2 weeks in 3 stay inside it.
 *
 * Pure math + the curve's durable last-good (Upstash KV). app/api/vol does the fetching;
 * every curve point has its own 4–5 source cascade there.
 */
import { defaultKv } from './factorStore';
import { sigmaOf, SIGMA_WINDOW } from './whatMoved';

export const CURVE_TENORS = [
    { tenor: '9D', index: 'VIX9D' },
    { tenor: '1M', index: 'VIX' },
    { tenor: '3M', index: 'VIX3M' },
    { tenor: '6M', index: 'VIX6M' },
];

// VIX ÷ VIX3M cut-offs. Measured on CBOE closes 2009-09 → 2026-09 (4,281 days): below
// 0.90 on 59% of days (calm), 0.90–1.00 on 34% (watch), ≥ 1.00 on 7.6% (stress: Feb
// 2018, Mar 2020, …; the top readings were 1.32–1.34). Median 0.88.
export const CALM_BELOW = 0.9;
export const STRESS_AT = 1.0;

const ISO = /^\d{4}-\d{2}-\d{2}$/;
const round = (v, d) => (Number.isFinite(v) ? Math.round(v * 10 ** d) / 10 ** d : null);

export function curveState(ratio) {
    if (!Number.isFinite(ratio) || ratio <= 0) return null;
    if (ratio >= STRESS_AT) return 'stress';
    if (ratio >= CALM_BELOW) return 'watch';
    return 'calm';
}

const goodValue = (v) => Number.isFinite(v) && v > 0;
const liveNewer = (q, eod) => !!(eod && q && goodValue(q.value) && ISO.test(q.date) && q.date > eod.date);

/**
 * @param {object} eod   {VIX9D|VIX|VIX3M|VIX6M: {value, date, source} | null} — latest close per index
 * @param {object} live  {same keys: {value, date, lastTime}} — CNBC quotes (optional)
 *
 * Live levels apply only when BOTH VIX and VIX3M have a quote newer than their close, so
 * the ratio never divides an intraday VIX by yesterday's VIX3M; then each point uses its
 * own quote when newer. An index with no close at all but a quote uses the quote — the
 * last source tier ('cnbc-quote').
 */
export function buildTermStructure(eod = {}, live = {}) {
    const goLive = liveNewer(live?.VIX, eod?.VIX) && liveNewer(live?.VIX3M, eod?.VIX3M);
    const points = [];
    for (const { tenor, index } of CURVE_TENORS) {
        const e = eod?.[index];
        const q = live?.[index];
        let p = null;
        if (e && goodValue(e.value) && ISO.test(e.date || '')) {
            p = goLive && liveNewer(q, e)
                ? { value: q.value, asOf: q.date, source: `${e.source}+live`, live: true }
                : { value: e.value, asOf: e.date, source: e.source, live: false };
        } else if (q && goodValue(q.value) && ISO.test(q.date || '')) {
            p = { value: q.value, asOf: q.date, source: 'cnbc-quote', live: false };
        }
        if (p) points.push({ tenor, index, ...p, value: round(p.value, 2) });
    }
    const at = (index) => points.find((p) => p.index === index) || null;
    const vix = at('VIX');
    const v3 = at('VIX3M');
    const nine = at('VIX9D');
    const ratio = vix && v3 ? round(vix.value / v3.value, 3) : null;
    const dates = points.map((p) => p.asOf).sort();
    return {
        points,
        ratio,
        state: curveState(ratio),
        // 9-day above 1-month: an event inside the next ~2 weeks is priced (Fed, CPI…).
        // Common (≈26% of days since 2011) — a note, not an alarm.
        frontInverted: nine && vix ? nine.value > vix.value : null,
        asOf: dates[0] || null, // the OLDEST point — never overstate freshness
        live: goLive,
        complete: points.length === CURVE_TENORS.length,
        stale: false,
    };
}

/** True when the curve is anything but four fresh CBOE points (edge cache + health check read this). */
export function curveDegraded(curve) {
    if (!curve || !curve.state || !curve.complete || curve.stale) return true;
    return curve.points.some((p) => !String(p.source || '').startsWith('cboe'));
}

/** % a leveraged fund loses over a year when its index ends flat, at annual vol `volPct`. Before fees. */
export function leveragedDecayPct(volPct, leverage = 3) {
    if (!goodValue(volPct) || !goodValue(leverage)) return null;
    const s = volPct / 100;
    return (1 - Math.exp(-((leverage * leverage - leverage) / 2) * s * s)) * 100;
}

/** ±1σ move in % over `days` trading days, from annualized implied vol in vol points. */
export function expectedMovePct(ivPct, days = 5) {
    if (!goodValue(ivPct) || !goodValue(days)) return null;
    return ivPct * Math.sqrt(days / 252);
}

/**
 * VIX since its last close, for the "What moved" strip (lib/whatMoved.js): today's level —
 * the live intraday quote when it is newer than the last close — against the last daily
 * close BEFORE it. Same cascaded series as the table (CBOE → CNBC → FRED → Yahoo), so it
 * inherits every backup. σ = the series' own daily % moves (last 60). null when there is
 * no earlier close to compare with.
 */
export function vixDay(series, live) {
    const s = (Array.isArray(series) ? series : []).filter((p) => p && goodValue(p.value) && ISO.test(p.date || ''));
    const last = s[s.length - 1];
    if (!last) return null;
    const cur = liveNewer(live, last)
        ? { value: live.value, asOf: live.date, live: true }
        : { value: last.value, asOf: last.date, live: false };
    const prior = s.filter((p) => p.date < cur.asOf);
    const prev = prior[prior.length - 1];
    if (!prev) return null;
    return {
        value: round(cur.value, 2),
        asOf: cur.asOf,
        live: cur.live,
        prev: round(prev.value, 2),
        prevDate: prev.date,
        sigma: sigmaOf(s.slice(-(SIGMA_WINDOW + 1)).map((p) => p.value), 'pct'),
    };
}

/**
 * The payload's `regime` block, from the table rows (SPY → VIX, QQQ → VXN / RV21) and the curve.
 * Never throws; a missing input nulls only its own number.
 */
export function buildRegime(tickers, curve) {
    const row = (t) => (Array.isArray(tickers) ? tickers.find((x) => x && x.ticker === t) : null) || {};
    const spy = row('SPY');
    const qqq = row('QQQ');
    return {
        curve: curve || null,
        decay: {
            leverage: 3,
            realizedVol: round(qqq.rv21, 2),
            impliedVol: round(qqq.iv, 2),
            realized: round(leveragedDecayPct(qqq.rv21), 2), // at QQQ's last-21-day volatility
            implied: round(leveragedDecayPct(qqq.iv), 2),     // at VXN (what options expect)
        },
        moves: { days: 5, SPY: round(expectedMovePct(spy.iv), 2), QQQ: round(expectedMovePct(qqq.iv), 2) },
    };
}

// ── Durable last-good for the curve (Upstash KV) ─────────────────────────────────────
// Tiers when the fresh curve can't make a call (no VIX or no VIX3M after every source):
// /tmp last-good payload's curve (app/api/vol) → this KV copy → "unavailable".
export const CURVE_KV_KEY = 'ftb:vol:curve:lg';
export const CURVE_MAX_AGE_MS = 5 * 864e5; // a 3-day weekend + a holiday, with slack
let lastSavedAsOf = null; // one write per close date per warm instance

/** Save a complete, fresh, all-CBOE curve. Never throws. @returns {Promise<boolean>} */
export async function saveCurveKV(curve, { kv = defaultKv, now = Date.now(), force = false } = {}) {
    try {
        if (curveDegraded(curve)) return false;
        if (!force && lastSavedAsOf === curve.asOf) return false;
        const ok = await kv.set(CURVE_KV_KEY, { curve, savedAt: new Date(now).toISOString() });
        if (ok) lastSavedAsOf = curve.asOf;
        return !!ok;
    } catch {
        return false;
    }
}

/** The KV copy relabelled as stale, or null when missing / too old / unusable. Never throws. */
export async function loadCurveKV({ kv = defaultKv, now = Date.now() } = {}) {
    try {
        const raw = await kv.get(CURVE_KV_KEY);
        const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
        return staleCurve(parsed?.curve, parsed?.savedAt, now, 'KV');
    } catch {
        return null;
    }
}

/** Relabel a saved curve as a backup copy; null unless it can still make a call and is recent. */
export function staleCurve(curve, savedAt, now = Date.now(), tier = 'last-good') {
    if (!curve || !curve.state || !Array.isArray(curve.points)) return null;
    const t = Date.parse(savedAt || '');
    if (!Number.isFinite(t) || now - t > CURVE_MAX_AGE_MS) return null;
    return { ...curve, live: false, stale: true, backup: `${tier} ${String(savedAt).slice(0, 16)}Z` };
}
