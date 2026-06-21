/**
 * Last-resort fallback: reconstruct an /api/fred-shaped payload from the
 * `dashboard_lkg` helper tab of the financial-dashboard-history Google Sheet.
 *
 * Only used when live FRED AND the /tmp last-known-good are both unavailable
 * (a total outage on a cold serverless instance — see store.js `serve()`).
 * The helper tab is written each scraper run as self-describing `key,value`
 * rows (no fragile column-position coupling), values + asOf only (no chart
 * history), with N/A metrics omitted entirely.
 *
 * Everything here is defensive: a malformed sheet returns null, never throws.
 */

import { EXTERNAL_URLS } from './constants';

const INDICATOR_KEYS = ['sahmRule', 'sentiment', 'claims', 'creditSpread', 'realYields', 'copperGold'];
const CHECKLIST_KEYS = ['nfci', 'm2', 'retail', 'housing', 'indpro', 'jolts', 'durable', 'savings'];
const STALE_DAYS = 4; // > 3 so the health check flags the genuine outage

/** Parse a single CSV line into fields, honoring gviz's double-quote wrapping. */
function parseCsvLine(line) {
    const out = [];
    let cur = '', inQ = false;
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (inQ) {
            if (c === '"') {
                if (line[i + 1] === '"') { cur += '"'; i++; } else { inQ = false; }
            } else { cur += c; }
        } else if (c === '"') {
            inQ = true;
        } else if (c === ',') {
            out.push(cur); cur = '';
        } else {
            cur += c;
        }
    }
    out.push(cur);
    return out;
}

/**
 * Parse the helper-tab CSV (two columns: key,value) into a flat string map.
 * Assumes one record per physical line (the writer never emits values with embedded
 * newlines) — `parseCsvLine` handles in-line quotes/escaping but not multi-line cells.
 */
export function parseLkgCsv(text) {
    const map = {};
    if (!text || typeof text !== 'string') return map;
    for (const line of text.split(/\r?\n/)) {
        if (!line.trim()) continue;
        const fields = parseCsvLine(line);
        const key = (fields[0] || '').trim();
        if (!key || key === 'key') continue; // skip blanks + header
        map[key] = (fields[1] ?? '').trim();
    }
    return map;
}

/** Tolerant numeric parse: returns null (not 0) for missing/"N/A"/junk. */
function num(v) {
    if (v === undefined || v === null) return null;
    const s = String(v).trim();
    if (!s || s.toUpperCase() === 'N/A') return null;
    const cleaned = s.replace(/[^0-9.\-]/g, '');
    if (cleaned === '' || cleaned === '-' || cleaned === '.') return null;
    const n = Number(cleaned);
    return Number.isFinite(n) ? n : null;
}

const orNull = (v) => (v === undefined || v === '' ? null : v);

/**
 * Rebuild an /api/fred-shaped payload from the parsed helper-tab map. History/
 * recessions are empty (charts blank in this rare path). Every metric is flagged
 * stale; _meta marks the payload stale + hasErrors so the health check still alerts.
 * Returns null if no usable metric is present.
 */
export function reconstructFred(map, now = new Date()) {
    if (!map || typeof map !== 'object') return null;
    const updatedAt = orNull(map['updated_at']);

    const indicators = {};
    for (const k of INDICATOR_KEYS) {
        const value = num(map[`indicators.${k}.value`]);
        if (value === null) continue;
        indicators[k] = {
            value,
            asOf: orNull(map[`indicators.${k}.asOf`]),
            status: orNull(map[`indicators.${k}.status`]) ?? undefined,
            stale: true, unavailable: false, staleDays: STALE_DAYS,
        };
    }

    const checklist = {};
    for (const k of CHECKLIST_KEYS) {
        const value = num(map[`checklist.${k}.value`]);
        if (value === null) continue;
        checklist[k] = {
            value,
            asOf: orNull(map[`checklist.${k}.asOf`]),
            status: orNull(map[`checklist.${k}.status`]) ?? undefined,
            bullish: String(map[`checklist.${k}.bullish`]).toLowerCase() === 'true',
            label: orNull(map[`checklist.${k}.label`]) ?? undefined,
            stale: true, unavailable: false, staleDays: STALE_DAYS,
        };
    }

    const ycCurrent = num(map['yieldCurve.current']);
    const pmCurrent = num(map['profitMargin.current']);
    const peRatio = num(map['peRatio']);

    const metricCount = Object.keys(indicators).length + Object.keys(checklist).length
        + (ycCurrent !== null ? 1 : 0) + (pmCurrent !== null ? 1 : 0);
    if (metricCount === 0) return null;

    return {
        yieldCurve: ycCurrent !== null
            ? { current: ycCurrent, asOf: orNull(map['yieldCurve.asOf']), stale: true, history: [] }
            : { current: null, asOf: null, stale: false, history: [] },
        profitMargin: pmCurrent !== null
            ? { current: pmCurrent, asOf: orNull(map['profitMargin.asOf']), stale: true, history: [] }
            : { current: null, asOf: null, stale: false, history: [] },
        peRatio,
        peRatioAsOf: updatedAt || now.toISOString(),
        recessions: [],
        indicators,
        checklist,
        _meta: {
            source: `Google Sheet (last-known-good ${updatedAt || 'unknown'})`,
            stale: true,
            hasErrors: true,
            fetchedAt: now.toISOString(),
            messages: ['live FRED + /tmp cache unavailable; served Google-Sheet last-known-good'],
            loadedCount: 0,
            lastGoodAt: updatedAt,
        },
    };
}

/** Fetch + reconstruct the sheet last-known-good. Never throws; returns null on any failure. */
export async function fetchSheetLkg(now = new Date()) {
    try {
        const url = EXTERNAL_URLS.SHEET_LKG;
        if (!url) return null;
        const opts = { cache: 'no-store' };
        try { opts.signal = AbortSignal.timeout(6000); } catch { /* older runtime: no timeout */ }
        const res = await fetch(url, opts);
        if (!res || !res.ok) return null;
        const text = await res.text();
        return reconstructFred(parseLkgCsv(text), now);
    } catch {
        return null;
    }
}
