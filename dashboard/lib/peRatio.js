/**
 * S&P 500 P/E — the three-layer cascade behind the "Market Valuation" tile.
 *
 *   1. multpl  (`pe_multpl`) — scrape of "Current S&P 500 PE Ratio" (TTM, as-reported)
 *   2. yahoo   (`pe_yahoo`)  — SPY key-statistics "PE Ratio (TTM)" × 1.07. A PHANTOM
 *                              from Vercel (JS-walled); kept as a best-effort attempt.
 *   3. computed (`pe_computed`) — S&P 500 close ÷ trailing-12M EPS (the same sum multpl
 *                              does), price and EPS from the route's own sources.
 *                              `peSource: 'computed'`; the tile says so.
 *   (A FRED `PE10` "CAPE" tier sat here until 2026-10-09 — FRED has no such series,
 *   it 400'd on every call. Removed; `peIsCape` stays false.)
 *
 * Every layer sits behind gate(): `?_fail=pe_multpl,pe_yahoo` proves the computed
 * tier, `?_fail=pe_multpl,pe_yahoo,pe_computed` proves the tile goes N/A rather than
 * inventing a number. Pure given injected fetchers;
 * never throws.
 */
import { gate } from './faults';

// "Current S&P 500 PE Ratio: 26.66" — the number must follow the label.
export function parseMultplPe(html) {
    const m = typeof html === 'string' ? html.match(/Current S&P 500 PE Ratio[^\d]*(\d+\.\d+)/) : null;
    const v = m ? parseFloat(m[1]) : NaN;
    return Number.isFinite(v) && v > 0 ? v : null;
}

// Yahoo's SPY P/E runs below the index's; ×1.07 is the long-standing bridge.
export function parseYahooPe(html) {
    const m = typeof html === 'string' ? html.match(/PE Ratio \(TTM\)[\s\S]*?(\d+\.\d+)/i) : null;
    const v = m ? parseFloat(m[1]) * 1.07 : NaN;
    return Number.isFinite(v) && v > 0 ? v : null;
}

/**
 * @param {{ multplHtml: () => Promise<string>, yahooHtml: () => Promise<string>,
 *           computed: () => Promise<{ spx:number, spxDate:string, eps:number, epsDate:string }> }} fetchers
 * @param {Set<string>} faults
 * @returns {Promise<{ peRatio: number|null, peSource: 'multpl'|'yahoo'|'computed'|null,
 *                     peIsCape: boolean, peAsOf: string|null, messages: string[] }>}
 */
export async function resolvePeRatio(fetchers, faults, { maskKey = (s) => s } = {}) {
    const messages = [];
    const err = (e) => maskKey(e?.message || String(e));

    try {
        const v = await gate('pe_multpl', faults, async () => parseMultplPe(await fetchers.multplHtml()));
        if (v) return { peRatio: v, peSource: 'multpl', peIsCape: false, peAsOf: null, messages };
        messages.push('P/E multpl failed: no match');
    } catch (e) { messages.push(`P/E multpl failed: ${err(e)}`); }

    try {
        const v = await gate('pe_yahoo', faults, async () => parseYahooPe(await fetchers.yahooHtml()));
        if (v) return { peRatio: v, peSource: 'yahoo', peIsCape: false, peAsOf: null, messages };
        messages.push('P/E Yahoo failed: no match');
    } catch (e) { messages.push(`P/E Yahoo failed: ${err(e)}`); }

    try {
        const c = await gate('pe_computed', faults, () => fetchers.computed());
        const v = c && c.eps > 0 ? c.spx / c.eps : NaN;
        // sanity band: the TTM P/E has lived between ~5 and ~125 (2009); outside 5–80 means a bad leg
        if (Number.isFinite(v) && v >= 5 && v <= 80) {
            messages.push(`P/E computed: S&P ${c.spx} (${c.spxDate}) ÷ EPS ${c.eps} (${c.epsDate})`);
            return { peRatio: Math.round(v * 100) / 100, peSource: 'computed', peIsCape: false, peAsOf: c.spxDate || null, messages };
        }
        messages.push(`P/E computed failed: implausible ${v}`);
    } catch (e) { messages.push(`P/E computed failed: ${err(e)}`); }

    messages.push('P/E unavailable — all layers failed');
    return { peRatio: null, peSource: null, peIsCape: false, peAsOf: null, messages };
}
