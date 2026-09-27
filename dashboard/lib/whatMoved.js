/**
 * whatMoved.js — the "What moved" strip: the 5 biggest moves since the last close.
 *
 * Ranked FAIRLY. Raw size means nothing across units (BTC −3% is a quiet day, 10Y +12bp
 * is not), so each move is divided by that series' own typical daily move — σ of its
 * last 60 daily changes. The chip shows the move in its own units (%, bp, points); the
 * rank is by "× a normal day", and 2× or more is flagged ⚡.
 *
 * Where each "since last close" comes from:
 *   SPY, 10Y, Oil, Gold, BTC — the live payload's own `dailyChange`, σ from its own
 *                              `history` (the same numbers their cards show)
 *   VIX                      — /api/vol `vixDay`: today's level vs the last close before it,
 *                              from the same CBOE → CNBC → FRED → Yahoo series as the vol card
 *   F&G                      — CNN's `previousClose` (CNN or RapidAPI only: the route's VIX
 *                              proxy would repeat the VIX chip, its stale cache is an old move)
 * σ backups: VIX and F&G fall back to the history sheet's σ (/api/history `moves`), and
 * F&G to FG_SIGMA_BAKED until its sheet column has 20+ changes. The sheet is used for σ
 * ONLY, never as a baseline: its Date column is the runner's UTC date, so a date's last
 * row is the 10am ET intraday snapshot, not a close (caught 2026-09-26: it made Friday's
 * VIX move read −6.6% instead of −5.1%). No usable source for the Dollar or USD/BDT
 * (no history on the live payload), so they are not ranked.
 * A series with no σ, a stale date, or a move that rounds to zero is left out — never
 * guessed. Nothing to show = the strip renders nothing.
 *
 * "Since the last close" is pinned to the MARKET DATE — the newest of VIX's as-of date and
 * SPY's last chart date. A slower feed (FRED's DGS10 runs a business day behind) is shown
 * one business day late at most, with its weekday on the chip ("10Y +7bp Thu"); further
 * behind, it is dropped, so an old move never sits next to today's as if it were today's.
 */
import { buildSeries, todayET } from './marks';

export const TOP_N = 5;
export const SIGMA_WINDOW = 60;
export const MIN_DIFFS = 20;
export const BIG_Z = 2;
/** A quote or baseline older than this is not "since yesterday" any more. */
export const MAX_AGE_DAYS = 5;
/** Business days a mover may trail the market date (and then carries its weekday). */
export const MAX_LAG_BIZ_DAYS = 1;
/** F&G sources that are CNN's own index. The route's other tiers are proxies or stale. */
export const LIVE_FG_SOURCES = ['CNN', 'RapidAPI'];

/**
 * σ of CNN Fear & Greed daily point changes, MEASURED 2026-09-26 from CNN graphdata:
 * 268 trading days, 2025-09-02 → 2026-09-25 → 4.78 (last 60: 4.46). Used only until the
 * history sheet's own F&G column (added 2026-09-26) holds MIN_DIFFS changes.
 */
export const FG_SIGMA_BAKED = 4.78;

/** History-sheet columns (0-based, Sheet1 — same map as lib/marks.js SHEET_METRICS), σ only. */
export const SHEET_MOVERS = {
    vix: { col: 36, unit: 'pct' },
    fg: { col: 66, unit: 'pts' },
};

/** `jump` is the card's `data-jump` label — tapping a chip scrolls there. */
export const MOVERS = [
    { key: 'spy', label: 'SPY', unit: 'pct', jump: 'SPY overview' },
    { key: 'vix', label: 'VIX', unit: 'pct', jump: 'Volatility' },
    { key: 'fg', label: 'F&G', unit: 'pts', jump: 'Fear & Greed' },
    { key: 'tnx', label: '10Y', unit: 'bp', jump: 'Markets' },
    { key: 'cl', label: 'Oil', unit: 'pct', jump: 'Markets' },
    { key: 'gc', label: 'Gold', unit: 'pct', jump: 'Markets' },
    { key: 'btc', label: 'BTC', unit: 'pct', jump: 'Markets' },
];

const num = (v) => (typeof v === 'number' ? v : typeof v === 'string' && v.trim() ? Number(v) : NaN);

/** The change from a to b in a unit: % of a, basis points, or plain points. */
export function change(a, b, unit) {
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    if (unit === 'pct') return a === 0 ? null : ((b - a) / Math.abs(a)) * 100;
    if (unit === 'bp') return (b - a) * 100;
    return b - a;
}

/**
 * σ of the last SIGMA_WINDOW daily changes. Zero changes are skipped: the sheet repeats
 * Friday's value on weekend rows, and counting those would shrink σ and make every
 * weekday move look bigger than it is (errs conservative: σ can only come out larger).
 */
export function sigmaOf(values, unit) {
    const v = (values || []).map(num).filter(Number.isFinite);
    const diffs = [];
    for (let i = 1; i < v.length; i++) {
        const d = change(v[i - 1], v[i], unit);
        if (d != null && Number.isFinite(d) && Math.abs(d) > 1e-12) diffs.push(d);
    }
    const w = diffs.slice(-SIGMA_WINDOW);
    if (w.length < MIN_DIFFS) return null;
    const mean = w.reduce((a, b) => a + b, 0) / w.length;
    const sd = Math.sqrt(w.reduce((a, b) => a + (b - mean) ** 2, 0) / w.length);
    return sd > 1e-9 ? sd : null;
}

/** Server side (/api/history): backup σ for the sheet-backed movers. */
export function buildMoveDigest(rows, now = new Date()) {
    const today = todayET(now);
    const out = {};
    for (const [key, { col, unit }] of Object.entries(SHEET_MOVERS)) {
        const s = buildSeries(rows || [], col).filter((p) => p.date <= today);
        const sigma = sigmaOf(s.slice(-(SIGMA_WINDOW + 1)).map((p) => p.value), unit);
        if (sigma) out[key] = { sigma };
    }
    return out;
}

const daysBetween = (a, b) => Math.round((Date.parse(`${b}T00:00:00Z`) - Date.parse(`${a}T00:00:00Z`)) / 86400000);
const isDate = (s) => typeof s === 'string' && /^\d{4}-\d{2}-\d{2}/.test(s);
/** A dated quote older than MAX_AGE_DAYS is not "since yesterday". Undated = trust the feed. */
const stale = (asOf, today) => isDate(asOf) && daysBetween(asOf.slice(0, 10), today) > MAX_AGE_DAYS;

/** Mon–Fri days in (a, b] — holidays count, so a holiday gap errs toward dropping. */
export function bizDaysBetween(a, b) {
    let n = 0;
    const end = Date.parse(`${b}T00:00:00Z`);
    for (let t = Date.parse(`${a}T00:00:00Z`) + 86400000; t <= end && n < 30; t += 86400000) {
        const w = new Date(t).getUTCDay();
        if (w > 0 && w < 6) n++;
    }
    return n;
}
const weekday = (d) => new Date(`${d}T12:00:00Z`).toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' });

/** The newest of VIX's as-of date and SPY's last chart date, or null. */
export function marketDateOf({ spy, vol } = {}) {
    const hist = spy && !spy.error && Array.isArray(spy.chartHistory) ? spy.chartHistory : [];
    const ds = [vol?.vixDay?.asOf, hist[hist.length - 1]?.date].filter(isDate).map((d) => d.slice(0, 10)).sort();
    return ds.length ? ds[ds.length - 1] : null;
}

/** "today" when the market date is today in New York, else its weekday ("Fri"); null = unknown. */
export function movedWhen(feeds, now = new Date()) {
    const d = marketDateOf(feeds);
    if (!d) return null;
    return d === todayET(now) ? 'today' : weekday(d);
}

const prices = (hist) => (Array.isArray(hist) ? hist.map((p) => num(p?.price)) : []);

/** SPY / 10Y / Oil / Gold / BTC: the payload's own dailyChange + history. */
function fromPayload(item, unit, today) {
    if (!item || !Array.isArray(item.history) || item.history.length < 2) return null;
    const value = num(item.current);
    const dc = item.dailyChange;
    if (!Number.isFinite(value) || !dc || !Number.isFinite(num(dc.value))) return null;
    const date = item.lastDate || item.history[item.history.length - 1]?.date;
    if (stale(date, today)) return null;
    const prev = value - num(dc.value);
    const delta = unit === 'pct' && Number.isFinite(num(dc.pct)) ? num(dc.pct) : change(prev, value, unit);
    return { value, prev, delta, sigma: sigmaOf(prices(item.history), unit), date: isDate(date) ? date.slice(0, 10) : null };
}

function read(key, unit, { spy, fg, extra, vol, history }, today) {
    const moves = history?.moves || {};
    switch (key) {
        case 'spy':
            if (!spy || spy.error) return null;
            return fromPayload({ current: spy.current, dailyChange: spy.dailyChange, history: spy.chartHistory }, unit, today);
        case 'tnx': return fromPayload(extra?.rates?.tnx, unit, today);
        case 'cl': case 'gc': case 'btc': return fromPayload(extra?.commodities?.[key], unit, today);
        case 'fg': {
            if (!fg || fg.error || !LIVE_FG_SOURCES.includes(fg._meta?.source)) return null;
            const value = num(fg.score), prev = num(fg.previousClose);
            if (!Number.isFinite(value) || !Number.isFinite(prev)) return null;
            return { value, prev, delta: value - prev, sigma: moves.fg?.sigma || FG_SIGMA_BAKED };
        }
        case 'vix': {
            const d = vol?.vixDay;
            if (!d || stale(d.asOf, today)) return null;
            const value = num(d.value), prev = num(d.prev);
            return { value, prev, delta: change(prev, value, unit), sigma: num(d.sigma) || moves.vix?.sigma, date: isDate(d.asOf) ? d.asOf.slice(0, 10) : null };
        }
        default: return null;
    }
}

/** The move as the chip prints it: "+0.5%", "−7bp", "+3". */
export function fmtMove(delta, unit) {
    const a = Math.abs(delta);
    const body = unit === 'bp' ? `${Math.round(a)}bp` : unit === 'pts' ? `${Math.round(a)}` : `${a < 0.1 ? a.toFixed(2) : a.toFixed(1)}%`;
    return `${delta > 0 ? '+' : '−'}${body}`;
}

/** True when the move would print as zero ("+0bp", "+0.00%") — nothing to show. */
export function roundsToZero(delta, unit) {
    const a = Math.abs(delta);
    return unit === 'pct' ? a < 0.005 : Math.round(a) === 0;
}

export const fmtLevel = (v) => v.toLocaleString('en-US', { maximumFractionDigits: Math.abs(v) >= 1000 ? 0 : 2 });

/**
 * @returns {Array<{key,label,unit,jump,value,prev,delta,z,big,text,day}>} biggest first, ≤ TOP_N.
 *   `day` = the move's weekday when it trails the market date, else null.
 */
export function collectMoves(feeds = {}, now = new Date()) {
    const today = todayET(now);
    let market = null;
    try { market = marketDateOf(feeds); } catch { market = null; }
    const out = [];
    for (const m of MOVERS) {
        let r = null;
        try { r = read(m.key, m.unit, feeds, today); } catch { r = null; }
        if (!r || !Number.isFinite(r.delta) || !Number.isFinite(r.sigma) || r.sigma <= 0) continue;
        if (roundsToZero(r.delta, m.unit)) continue;
        // SPY's `current` is a live spot even when its chart ends a day earlier — never lagged.
        const lag = m.key !== 'spy' && r.date && market && r.date < market ? bizDaysBetween(r.date, market) : 0;
        if (lag > MAX_LAG_BIZ_DAYS) continue;
        const z = Math.abs(r.delta) / r.sigma;
        out.push({ ...m, value: r.value, prev: r.prev, delta: r.delta, z, big: z >= BIG_Z, text: fmtMove(r.delta, m.unit), day: lag ? weekday(r.date) : null });
    }
    return out.sort((a, b) => b.z - a.z).slice(0, TOP_N);
}
