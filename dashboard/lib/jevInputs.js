/**
 * Pure helpers for repairing pill inputs that have incomplete data from the
 * sibling /api/fred route.
 *
 * Every function here is unit-testable with no network or side effects.
 *
 * Exports:
 *   FRESH                    — freshness deadlines per input (days)
 *   claims4wkFromHistory     — 4-week average of weekly ICSA in thousands
 *   sahmFromHistory          — Sahm rule from monthly UNRATE history
 *   parseTreasurySpreadCsv   — Treasury daily-yield CSV → [{date, value: long − short}]
 *   resolvePillInput         — generic cascade over sources → { value, asOf, source, tried }
 */

import { isStale } from './freshness';
import { loadLastGood, saveLastGood } from './store';
import { splitCsvLine } from './horsemen';

// Days before a source's NEWEST point counts as stale. Dated by OBSERVATION, not release:
// UNRATE for August is dated 08-01 and published ~Sep 4, and the next print lands ~Oct 3 —
// so the newest point is legitimately up to ~65 days old. 45 rejected every Sahm tier for
// the last three weeks of each month (prod, 2026-09-19: `sheet:stale(2026-08-01)`).
export const FRESH = { T10Y3M: 7, NFCI: 14, ICSA: 14, UNRATE: 75 };

const finite = (n) => typeof n === 'number' && Number.isFinite(n);

/**
 * 4-week average of weekly ICSA in THOUSANDS, from an ascending history.
 * Each value in the history is a raw number (e.g. 231000 for 231k).
 * Returns the mean ÷ 1000, not rounded. null if < 4 points.
 */
export function claims4wkFromHistory(history) {
    if (!Array.isArray(history) || history.length < 4) return null;
    const last4 = history.slice(-4);
    const total = last4.reduce((sum, p) => sum + (p.value ?? 0), 0);
    return total / 4000; // ÷1000 for thousands, ÷4 for the average
}

/**
 * Sahm: mean of the latest 3 months minus the min of the latest 12 months,
 * from an ascending monthly UNRATE history (each point has a date and value
 * as a decimal fraction, e.g. 4.1 for 4.1%).
 * Returns the raw difference (not a percentage subtraction). null if < 12 points.
 */
export function sahmFromHistory(history) {
    if (!Array.isArray(history) || history.length < 12) return null;
    const last12 = history.slice(-12);
    const last3 = last12.slice(-3);
    const mean3 = last3.reduce((s, p) => s + (p.value ?? 0), 0) / 3;
    const min12 = Math.min(...last12.map((p) => p.value ?? Infinity));
    return mean3 - min12;
}

/** MM/DD/YYYY → YYYY-MM-DD, or null. */
function usDateToIso(s) {
    const m = String(s || '').trim().match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
    if (!m) return null;
    const [, mm, dd, yyyy] = m;
    return `${yyyy}-${mm.padStart(2, '0')}-${dd.padStart(2, '0')}`;
}

function ascendUnique(rows) {
    const byDate = new Map();
    for (const r of rows) byDate.set(r.date, r.value);
    return [...byDate.entries()].map(([date, value]) => ({ date, value })).sort((a, b) => (a.date < b.date ? -1 : 1));
}

/**
 * US Treasury daily par yield curve CSV (home.treasury.gov, keyless, newest
 * first) → ascending [{date, value}] where value = long tenor − short tenor,
 * rounded to FRED's 2dp. Columns are read BY HEADER NAME (e.g. '3 mo', '10 yr',
 * case-insensitive) — Treasury inserts tenors over time, so never by position.
 * Verified 2026-09-18: 10 Yr 5.01 − 3 Mo 4.14 = 0.87 = FRED T10Y3M 0.87.
 */
export function parseTreasurySpreadCsv(csv, shortName, longName) {
    if (!csv || typeof csv !== 'string') return [];
    const lines = csv.trim().split(/\r?\n/).filter((l) => l.trim());
    if (lines.length < 2) return [];
    const header = splitCsvLine(lines[0]).map((h) => h.trim().toLowerCase());
    const dateIdx = header.findIndex((h) => h === 'date');
    const iS = header.indexOf(String(shortName).toLowerCase());
    const iL = header.indexOf(String(longName).toLowerCase());
    if (dateIdx < 0 || iS < 0 || iL < 0) throw new Error(`Treasury CSV: ${shortName} / ${longName} column not found`);
    const out = [];
    for (const line of lines.slice(1)) {
        const f = splitCsvLine(line);
        const date = usDateToIso(f[dateIdx]);
        if (!date) continue;
        const s = parseFloat(f[iS]), l = parseFloat(f[iL]);
        if (!finite(s) || !finite(l)) continue;
        out.push({ date, value: Math.round((l - s) * 100) / 100 });
    }
    return ascendUnique(out);
}

/**
 * Generic cascade for one pill input.
 *
 * sources: [{ name, freshnessDays, fetch, derive? } | { name, freshnessDays, read }]
 *   fetch() must return an ascending [{date, value}] array or throw. Stale
 *   sources (newest point older than freshnessDays) are skipped.
 *   derive(history) (optional) turns the history into the value we keep —
 *   e.g. a 4-week average — instead of the newest point; null → source skipped.
 *   read() is the single-value form: returns { value, asOf } (a snapshot such
 *   as the Google-Sheet last-known-good tab) — same freshness rule on asOf.
 * faults: Set (see lib/faults.js) — `hm_<name>` skips that source; any fault
 *   present disables last-good WRITES; `lastgood` disables the READ.
 * now: Date — injectable "current" time for staleness tests.
 * lastGoodKey: string — store key, e.g. 'jev-t10y3m'.
 * maxStaleMs: number — max age in ms for a last-good entry (default 7 days).
 * store: optional { load(key, maxAgeMs), save(key, data) } (may be async) —
 *   defaults to the /tmp store in lib/store.js. lib/jevStore.js adds KV.
 *
 * Returns { value: number|null, asOf: string|null, source: string|null, tried: string[] }.
 * Never throws. `tried` carries the error text of each failed source, so
 * `_meta.inputTried` on the pills payload shows WHY a tier fell through.
 */
export async function resolvePillInput({ sources, faults = new Set(), now = new Date(), lastGoodKey, maxStaleMs = 7 * 864e5, store = null }) {
    const tried = [];
    const load = store && typeof store.load === 'function' ? (k, m) => store.load(k, m) : (k, m) => loadLastGood(k, m);
    const save = store && typeof store.save === 'function' ? (k, d) => store.save(k, d) : (k, d) => saveLastGood(k, d);
    const errText = (e) => String(e?.message || e || 'error').replace(/\s+/g, ' ').slice(0, 80);

    for (const s of sources) {
        if (faults.has(`hm_${s.name}`)) {
            tried.push(`${s.name}:off`);
            continue;
        }
        try {
            let value, asOf;
            if (typeof s.read === 'function') {
                const r = await s.read();
                if (!r || !finite(r.value) || !r.asOf) {
                    tried.push(`${s.name}:empty`);
                    continue;
                }
                if (isStale(r.asOf, s.freshnessDays, now)) {
                    tried.push(`${s.name}:stale(${r.asOf})`);
                    continue;
                }
                value = r.value;
                asOf = r.asOf;
            } else {
                const history = await s.fetch();
                if (!Array.isArray(history) || history.length < 2) {
                    tried.push(`${s.name}:empty`);
                    continue;
                }
                const last = history[history.length - 1];
                if (!last || !finite(last.value) || !last.date) {
                    tried.push(`${s.name}:empty`);
                    continue;
                }
                if (isStale(last.date, s.freshnessDays, now)) {
                    tried.push(`${s.name}:stale(${last.date})`);
                    continue;
                }
                value = typeof s.derive === 'function' ? s.derive(history) : last.value;
                if (!finite(value)) {
                    tried.push(`${s.name}:underived`);
                    continue;
                }
                asOf = last.date;
            }
            tried.push(`${s.name}:ok`);
            if (lastGoodKey && faults.size === 0) {
                try { await save(lastGoodKey, { value, asOf, source: s.name }); } catch { /* best effort */ }
            }
            return { value, asOf, source: s.name, tried };
        } catch (e) {
            tried.push(`${s.name}:err(${errText(e)})`);
        }
    }

    // All sources failed — try last-known-good
    if (lastGoodKey && !(faults.has('lastgood'))) {
        try {
            const lg = await load(lastGoodKey, maxStaleMs);
            if (lg && lg.data && lg.data.value != null) {
                tried.push(`lastgood:ok(${lg.savedAt})`);
                return { value: lg.data.value, asOf: lg.data.asOf ?? null, tried, source: 'lastgood' };
            }
        } catch { /* fall through */ }
    }

    tried.push('lastgood:none');
    return { value: null, asOf: null, source: null, tried };
}
