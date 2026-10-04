/**
 * Global Markets: what each row's change covers (ExtraMarketsGrid.js).
 *
 * One unlabelled "+X.XX%" column used to mix four windows: TNX/T2Y move per session,
 * MORT30 and the mortgage payment per week, ZRI rent per month, ATNHPI per quarter.
 * Here the change comes from the row's own last two history points and is tagged
 * from the gap between their dates:
 *   1-4 days -> that session's weekday ("Thu"), 6-8 -> "1w", 28-31 -> "1mo",
 *   89-92 -> "3mo", anything else -> no tag (the number stays, the window is unknown).
 * Rate rows print basis points like What moved ("−5bp"); prices print % ("+0.23%").
 *
 * A trailing bar on a non-NYSE day (weekend / holiday) that only repeats the previous
 * close is dropped, so a copied Friday close never shows as "+0.00%". A non-session
 * bar whose value MOVED (CL's real Sunday futures bar) is kept. BTC trades every day,
 * so it is never dropped; weekly / monthly series are never trimmed either.
 *
 * If the shown value is not the last history point, the history window does not
 * describe it: the source's own dailyChange is used, untagged. No history, or no
 * usable number -> null (the row shows nothing, never a fake 0).
 */
import { sessionOf, weekdayOf } from './marketClock';

const DAY_MS = 864e5;
const ON_HISTORY_TOLERANCE = 1e-4;   // relative: absorbs a source rounding current vs its bar
const ALWAYS_TRADES = new Set(['BTC']);

const toNum = (v) => (v == null || v === '' ? NaN : Number(v));
const dayGap = (a, b) => Math.round((Date.parse(`${b}T12:00:00Z`) - Date.parse(`${a}T12:00:00Z`)) / DAY_MS);

function tagFor(prevDate, lastDate) {
    const g = dayGap(prevDate, lastDate);
    if (g >= 1 && g <= 4) return weekdayOf(lastDate);
    if (g >= 6 && g <= 8) return '1w';
    if (g >= 28 && g <= 31) return '1mo';
    if (g >= 89 && g <= 92) return '3mo';
    return null;
}

/** "+25bp" / "−5bp" / "0bp" or "+0.23%" / "−1.01%" / "0.00%" (true minus, no sign on zero). */
function format(delta, rate) {
    if (rate) {
        const bp = Math.round(delta * 100);
        if (bp === 0) return { text: '0bp', dir: 0 };
        return { text: `${bp > 0 ? '+' : '−'}${Math.abs(bp)}bp`, dir: Math.sign(bp) };
    }
    const shown = Math.abs(delta).toFixed(2);
    if (Number(shown) === 0) return { text: '0.00%', dir: 0 };
    return { text: `${delta > 0 ? '+' : '−'}${shown}%`, dir: Math.sign(delta) };
}

/**
 * @param d  one /api/market-extra metric: {current, dailyChange?, history?: [{date, price}]}
 * @param opts.rate    true for yield/rate rows (TNX, T2Y, MORT30): change in bp
 * @param opts.ticker  row ticker (BTC is never trimmed)
 * @returns {{text:string, tag:string|null, dir:-1|0|1}|null}
 */
export function marketWindow(d, { rate = false, ticker } = {}) {
    const pts = (Array.isArray(d?.history) ? d.history : [])
        .map((p) => ({ date: p?.date, v: toNum(p?.price ?? p?.value) }))
        .filter((p) => typeof p.date === 'string' && Number.isFinite(p.v));
    if (pts.length < 2) return null;

    const n = pts.length;
    const tail = pts[n - 1];
    // Only a daily series is trimmed (tail within a session gap of the bar before it), so a
    // weekly / monthly print that happens to fall on a holiday or a Saturday is never dropped.
    const gap = dayGap(pts[n - 2].date, tail.date);
    const trim = !ALWAYS_TRADES.has(ticker) && n >= 3 && gap >= 1 && gap <= 4
        && sessionOf(tail.date) == null && tail.v === pts[n - 2].v;
    const last = trim ? pts[n - 2] : tail;
    const prev = trim ? pts[n - 3] : pts[n - 2];

    const cur = toNum(d.current);
    if (Number.isFinite(cur) && Math.abs(cur - last.v) > Math.abs(last.v) * ON_HISTORY_TOLERANCE) {
        // The shown value is off the history: fall back to the source's own change, no window.
        const dc = d.dailyChange || {};
        const delta = rate ? toNum(dc.value) : toNum(dc.pct);
        if (!Number.isFinite(delta)) return null;
        return { ...format(delta, rate), tag: null };   // pct is already in percent
    }

    let delta;
    if (rate) delta = last.v - prev.v;
    else {
        if (prev.v === 0) return null;
        delta = 100 * (last.v / prev.v - 1);
    }
    return { ...format(delta, rate), tag: tagFor(prev.date, last.date) };
}

/** What a row with no history says instead of a change. */
export function spotLabel(ticker) {
    if (typeof ticker !== 'string') return null;
    if (ticker.includes('/') || ticker === 'DXY') return 'daily rate';   // ER-API style, once a day
    if (ticker === 'GOLD' || ticker === 'BTC') return 'live';            // gold-api / Coinbase spot fallback
    return null;
}
