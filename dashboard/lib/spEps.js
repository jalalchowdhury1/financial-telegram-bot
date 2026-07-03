/**
 * S&P 500 EPS (trailing 12 months, as-reported) — parsers + source cascade.
 *
 * Kept separate from the FRED route (like copperGold.js) so the cascade logic
 * and HTML/CSV parsing are pure and unit-testable. The route supplies the
 * network-bound source descriptors; everything here is deterministic.
 *
 * A "source descriptor" is: { name, freshnessDays, fetch: async () =>
 *   { current:number, currentDate:'YYYY-MM-DD', historyAsc:[{date,value}] } }
 * where value is index-level TTM EPS in dollars (~240 in 2025). Each source may
 * be disabled for testing via the `eps_<name>` fault (e.g. ?_fail=eps_multpl).
 *
 * Unlike copper/gold (whose resolveLeg REJECTS stale sources outright), a stale
 * EPS level is still served — marked stale:true → orange 🕐 in the UI — because
 * TTM as-reported earnings inherently lag ~2-3 quarters and an old real number
 * beats an N/A. Fresh sources are still preferred over stale ones.
 */
import { isStale } from './freshness';

/** History start — matches the Profit Margin card's span (quarterly since 1947). */
export const SP_EPS_START = '1947-01-01';

const MONTHS = { Jan: '01', Feb: '02', Mar: '03', Apr: '04', May: '05', Jun: '06', Jul: '07', Aug: '08', Sep: '09', Oct: '10', Nov: '11', Dec: '12' };

/**
 * Parse multpl's "S&P 500 Earnings by Month" table HTML into ascending
 * [{date:'YYYY-MM-DD', value}]. Rows look like:
 *   <td>Sep 30, 2025</td> <td> &#x2002; 241.50 </td>
 */
export function parseMultplEps(html) {
    const out = [];
    if (typeof html !== 'string') return out;
    // The value must be followed by a tag boundary (`<`), so a response body
    // truncated mid-number can't smuggle in a phantom partial value.
    const re = /<td>\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),\s+(\d{4})\s*<\/td>\s*<td>\s*(?:&#x2002;|&nbsp;)?\s*([\d,]+(?:\.\d+)?)\s*</g;
    let m;
    while ((m = re.exec(html)) !== null) {
        const value = parseFloat(m[4].replace(/,/g, ''));
        if (!Number.isFinite(value)) continue;
        out.push({ date: `${m[3]}-${MONTHS[m[1]]}-${m[2].padStart(2, '0')}`, value });
    }
    out.sort((a, b) => (a.date < b.date ? -1 : 1));
    return out;
}

/**
 * Parse the datasets/s-and-p-500 GitHub mirror of Shiller's monthly data
 * (Date,SP500,...,Real Earnings,...) into ascending [{date, value}] of TTM
 * as-reported EPS. The REAL (inflation-adjusted) column is used so the series
 * matches multpl's table, which is also in constant current dollars. Recent
 * months carry a 0.0 placeholder until the actuals land — those are dropped,
 * so the newest row is the newest genuine value.
 */
export function parseShillerCsv(text) {
    const out = [];
    if (typeof text !== 'string' || !text.trim()) return out;
    const lines = text.trim().split('\n');
    const header = lines[0].split(',').map((h) => h.trim());
    const dateIdx = header.indexOf('Date');
    const epsIdx = header.indexOf('Real Earnings');
    if (dateIdx === -1 || epsIdx === -1) return out;
    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',');
        const date = cols[dateIdx]?.trim();
        const value = parseFloat(cols[epsIdx]);
        if (!/^\d{4}-\d{2}-\d{2}$/.test(date || '') || !Number.isFinite(value) || value <= 0) continue;
        out.push({ date, value });
    }
    out.sort((a, b) => (a.date < b.date ? -1 : 1));
    return out;
}

/**
 * Trim a monthly EPS history to 1947 on — the same span as the Profit Margin
 * card. The full monthly cadence is kept (~945 points) so MiniChart's explicit
 * monthly mode (cadence="monthly") can offer 1Y/3Y/5Y tabs with real detail;
 * a 1Y window is 12 points instead of 4.
 */
export function toMonthlyHistory(historyAsc, start = SP_EPS_START) {
    if (!Array.isArray(historyAsc)) return [];
    return historyAsc.filter((h) => h.date >= start);
}

/**
 * Try each source in order. Skip fault-injected ones (`eps_<name>`); a source
 * with a fresh-enough newest point wins the level immediately; a stale one is
 * kept as a graceful-staleness fallback (served with stale:true only if no
 * fresh source answers). The FIRST source that exposes history supplies the
 * chart, even when a later/spot-only source supplies the level. Stops fetching
 * as soon as both a fresh level and a history are in hand. Never throws.
 */
export async function resolveSpEps(sources, faults, now = new Date()) {
    const FUTURE_TOLERANCE_MS = 2 * 86400000; // >2 days ahead of now = a source-side date bug
    const tried = [];
    let fresh = null;      // first fresh level
    let staleBest = null;  // first stale level (fallback)
    let history = [];
    let historySource = null;
    for (const s of sources) {
        if (fresh && history.length) break;
        if (faults && faults.has(`eps_${s.name}`)) { tried.push(`${s.name}:off`); continue; }
        let r;
        try { r = await s.fetch(); } catch (e) { tried.push(`${s.name}:err`); continue; }
        if (!r) { tried.push(`${s.name}:empty`); continue; }
        // A date meaningfully in the future means the source mangled its dates —
        // don't let it pass as "fresh" forever (isStale can only catch OLD dates).
        const ts = r.currentDate ? new Date(r.currentDate).getTime() : NaN;
        if (Number.isFinite(ts) && ts - now.getTime() > FUTURE_TOLERANCE_MS) {
            tried.push(`${s.name}:future(${r.currentDate})`);
            continue;
        }
        // Harvest history even when the level is unusable — a chart from a source
        // with a broken spot value still beats no chart.
        if (!history.length && Array.isArray(r.historyAsc) && r.historyAsc.length) {
            history = r.historyAsc;
            historySource = s.name;
        }
        if (!Number.isFinite(r.current) || !r.currentDate) { tried.push(`${s.name}:empty`); continue; }
        const stale = isStale(r.currentDate, s.freshnessDays, now);
        tried.push(`${s.name}:${stale ? `stale(${r.currentDate})` : 'ok'}`);
        if (!fresh && !stale) fresh = { current: r.current, asOf: r.currentDate, source: s.name };
        else if (!staleBest && stale) staleBest = { current: r.current, asOf: r.currentDate, source: s.name };
    }
    const level = fresh || staleBest;
    return {
        current: level ? level.current : null,
        asOf: level ? level.asOf : null,
        stale: !!(level && !fresh),
        unavailable: !level,
        source: level ? level.source : null,
        historySource,
        history,
        tried,
    };
}
