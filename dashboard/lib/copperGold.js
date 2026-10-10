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

// Human labels for the copper leg. LME cash (Westmetall) is NOT COMEX HG: the two can
// differ by several percent (e.g. a US tariff premium on COMEX), so the card must say
// which market the copper price came from.
export const COPPER_LEG_LABELS = {
    cnbc: 'COMEX HG (CNBC)',
    westmetall: 'LME cash (Westmetall)',
    fred: 'IMF monthly (FRED PCOPPUSDM)',
    goldapi: 'spot (gold-api)',
};

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
        copperLabel: copper.source ? (COPPER_LEG_LABELS[copper.source] || copper.source) : null,
        goldSource: gold.source,
        source: copper.source || gold.source ? `copper:${copper.source ?? 'n/a'} · gold:${gold.source ?? 'n/a'}` : null,
        tried: { copper: copper.tried, gold: gold.tried },
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Westmetall LME copper (cash-settlement) — the daily copper tier WITH history.
// ─────────────────────────────────────────────────────────────────────────────

export const LB_PER_TONNE = 2204.6226;
const WM_MONTHS = {
    january: '01', february: '02', march: '03', april: '04', may: '05', june: '06',
    july: '07', august: '08', september: '09', october: '10', november: '11', december: '12',
};
const stripTags = (h) => String(h).replace(/<[^>]*>/g, ' ').replace(/&nbsp;/g, ' ').replace(/\s+/g, ' ').trim();

/**
 * Westmetall's table HTML → ascending [{date:'YYYY-MM-DD', price}] in $/lb.
 *
 * Rows: `<td>09. October 2026</td><td>14,689.00</td><td>14,570.00</td><td>233,025</td>`
 * under a header `date | LME Copper Cash-Settlement | LME Copper 3-month | LME Copper
 * stock` that repeats every month. COLUMN-ANCHORED: the cash column is located by
 * its header text, never by position — reading the STOCK column (tonnes, ~240,000)
 * by mistake would be a 16x wrong copper price. No header → [] (no data, not wrong
 * data). USD per metric tonne ÷ 2204.6226 = USD per lb, matching COMEX/CNBC units.
 */
export function parseWestmetallTable(html) {
    if (typeof html !== 'string' || !html) return [];
    const rows = html.match(/<tr[^>]*>[\s\S]*?<\/tr>/gi) || [];
    let dateIdx = -1, cashIdx = -1;
    const byDate = new Map();
    for (const row of rows) {
        const ths = row.match(/<th[^>]*>[\s\S]*?<\/th>/gi);
        if (ths) {
            const h = ths.map((c) => stripTags(c).toLowerCase());
            dateIdx = h.findIndex((x) => x === 'date');
            cashIdx = h.findIndex((x) => /cash-settlement/.test(x));
            continue;
        }
        if (dateIdx < 0 || cashIdx < 0) continue;
        const cells = (row.match(/<td[^>]*>[\s\S]*?<\/td>/gi) || []).map(stripTags);
        const m = /^(\d{1,2})\.\s*([a-z]+)\s+(\d{4})$/i.exec(cells[dateIdx] || '');
        const mm = m && WM_MONTHS[m[2].toLowerCase()];
        const perTonne = parseFloat(String(cells[cashIdx] || '').replace(/,/g, ''));
        if (!mm || !Number.isFinite(perTonne) || perTonne <= 0) continue;
        byDate.set(`${m[3]}-${mm}-${m[1].padStart(2, '0')}`, perTonne / LB_PER_TONNE);
    }
    return [...byDate.entries()]
        .map(([date, price]) => ({ date, price }))
        .sort((a, b) => (a.date < b.date ? -1 : 1));
}

/**
 * Build the Westmetall copper leg `{ current, currentDate, historyAsc }` ($/lb).
 * `fetchPage(year?)` returns one year's table HTML (no year = current year). When the
 * current-year page spans < 100 days (January-March), the prior year is fetched too —
 * best-effort: its failure only shortens the history, never fails the leg. Throws
 * when the current page yields no rows (resolveLeg records `westmetall:err`).
 */
export async function westmetallCopperLeg(fetchPage, now = new Date()) {
    const hist = parseWestmetallTable(await fetchPage());
    let historyAsc = hist;
    const oldest = hist[0]?.date;
    if (!oldest || (now.getTime() - new Date(oldest).getTime()) / 864e5 < 100) {
        let prev = [];
        try { prev = parseWestmetallTable(await fetchPage(now.getUTCFullYear() - 1)); }
        catch (e) { console.warn(`[Westmetall prior year] ${e?.message}`); }
        const seen = new Set(hist.map((p) => p.date));
        historyAsc = [...prev.filter((p) => !seen.has(p.date)), ...hist];
    }
    const last = historyAsc[historyAsc.length - 1];
    if (!last) throw new Error('Westmetall LME_Cu_cash: no rows parsed');
    return { current: last.price, currentDate: last.date, historyAsc };
}
