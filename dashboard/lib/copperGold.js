/**
 * Copper/Gold ratio — leg cascade + ratio/trend computation.
 *
 * Kept separate from the FRED route so the cascade logic (try sources in order,
 * skip fault-injected ones, reject empty/stale results) and the ratio/delta math
 * are pure and unit-testable. The route supplies the network-bound source
 * descriptors; everything here is deterministic given those.
 *
 * A "source descriptor" is: { name, freshnessDays, fetch: async () =>
 *   { current:number, currentDate:'YYYY-MM-DD', historyAsc:[{date,price}] } }
 * where price is in the canonical unit (copper $/lb, gold $/oz). Each source may
 * be disabled for testing via the `cg_<name>` fault (e.g. ?_fail=cg_cnbc).
 */
import { isStale } from './freshness';
import { copperGoldRatio, priceAtAgo, ratioChange } from './finance';

export const CHANGE_WINDOWS = [30, 90]; // ≈ 1 month, ≈ 3 months

/**
 * Try each source in order. Skip fault-injected ones; reject results that are
 * empty (no finite price/date) or whose newest point is staler than the source
 * allows — then fall through to the next source. Returns the first healthy leg
 * (with a `tried` trail), or a null leg if every source failed.
 */
export async function resolveLeg(sources, faults, now = new Date()) {
    const tried = [];
    for (const s of sources) {
        if (faults && faults.has(`cg_${s.name}`)) { tried.push(`${s.name}:off`); continue; }
        try {
            const r = await s.fetch();
            if (!r || !Number.isFinite(r.current) || !r.currentDate) { tried.push(`${s.name}:empty`); continue; }
            if (isStale(r.currentDate, s.freshnessDays, now)) { tried.push(`${s.name}:stale(${r.currentDate})`); continue; }
            tried.push(`${s.name}:ok`);
            return {
                current: r.current,
                currentDate: r.currentDate,
                historyAsc: Array.isArray(r.historyAsc) ? r.historyAsc : [],
                source: s.name,
                tried,
            };
        } catch (e) { tried.push(`${s.name}:err`); }
    }
    return { current: null, currentDate: null, historyAsc: [], source: null, tried };
}

/**
 * Build the dashboard `indicators.copperGold` object from two resolved legs.
 * The ratio is null (→ N/A) unless BOTH legs resolved. The trend over each window
 * is computed from each leg's own history; if either leg lacks enough history to
 * span a window, that window's change is null (we still show the ratio).
 */
export function buildCopperGold(copper, gold, windows = CHANGE_WINDOWS) {
    const ratio = copper.current != null && gold.current != null ? copperGoldRatio(copper.current, gold.current) : null;

    const [w1, w3] = windows;
    const changeForWindow = (w) => {
        if (ratio == null) return null;
        const cAgo = priceAtAgo(copper.historyAsc, copper.currentDate, w);
        const gAgo = priceAtAgo(gold.historyAsc, gold.currentDate, w);
        const ratioAgo = cAgo != null && gAgo != null ? copperGoldRatio(cAgo, gAgo) : null;
        return ratioChange(ratio, ratioAgo);
    };
    const d30 = changeForWindow(w1);
    const d90 = changeForWindow(w3);

    // asOf = the older of the two legs (honest about the laggier leg, e.g. monthly copper).
    const asOf = copper.currentDate && gold.currentDate
        ? (copper.currentDate < gold.currentDate ? copper.currentDate : gold.currentDate)
        : (copper.currentDate || gold.currentDate || null);

    const dir = d30 ?? d90; // prefer the 1-month trend for the rising/falling flag
    const status = ratio == null ? 'unknown' : (dir == null ? 'neutral' : dir.value > 0 ? 'rising' : dir.value < 0 ? 'falling' : 'neutral');

    return {
        value: ratio,
        asOf,
        stale: false, // a served value is always fresh — stale sources are rejected in resolveLeg
        unavailable: ratio == null,
        status,
        change: d30 ? d30.value : null,
        changePct: d30 ? d30.pct : null,
        change3mo: d90 ? d90.value : null,
        changePct3mo: d90 ? d90.pct : null,
        copper: copper.current,
        gold: gold.current,
        copperSource: copper.source,
        goldSource: gold.source,
        source: copper.source || gold.source ? `copper:${copper.source ?? 'n/a'} · gold:${gold.source ?? 'n/a'}` : null,
        tried: { copper: copper.tried, gold: gold.tried },
    };
}
