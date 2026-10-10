/**
 * S&P 500 P/E — the three-layer cascade behind the "Market Valuation" tile.
 *
 *   1. multpl  (`pe_multpl`) — scrape of "Current S&P 500 PE Ratio" (TTM, as-reported)
 *   2. yahoo   (`pe_yahoo`)  — SPY key-statistics "PE Ratio (TTM)" × 1.07. A PHANTOM
 *                              from Vercel (JS-walled); kept as a best-effort attempt.
 *   3. fred    (`pe_fred`)   — FRED `PE10` = Shiller CAPE. A DIFFERENT METRIC: a 10-yr
 *                              inflation-adjusted smoothed ratio (runs ~40 when TTM is
 *                              ~30). Served only labelled as CAPE (`peSource: 'cape'`,
 *                              `peIsCape: true`), and the tile then reads "CAPE ~40",
 *                              never "P/E ~40".
 *
 * Every layer sits behind gate(), so `?_fail=pe_multpl,pe_yahoo` proves the CAPE
 * fallback and its labelling on prod, and `?_fail=pe_multpl,pe_yahoo,pe_fred` proves
 * the tile goes N/A rather than inventing a number. Pure given injected fetchers;
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
 *           capeObs: () => Promise<Array<{date:string,value:number}>> }} fetchers
 * @param {Set<string>} faults
 * @returns {Promise<{ peRatio: number|null, peSource: 'multpl'|'yahoo'|'cape'|null,
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
        const obs = await gate('pe_fred', faults, () => fetchers.capeObs());
        const v = Array.isArray(obs) && obs.length && Number.isFinite(obs[0]?.value) ? obs[0].value : null;
        if (v) {
            messages.push('P/E is Shiller CAPE (10-yr smoothed), NOT trailing-twelve-month — multpl scrape failed');
            // CAPE is monthly; its own observation date is the honest as-of.
            return { peRatio: v, peSource: 'cape', peIsCape: true, peAsOf: obs[0].date || null, messages };
        }
        messages.push('P/E CAPE failed: no observations');
    } catch (e) { messages.push(`P/E CAPE failed: ${err(e)}`); }

    messages.push('P/E unavailable — all layers failed');
    return { peRatio: null, peSource: null, peIsCape: false, peAsOf: null, messages };
}
