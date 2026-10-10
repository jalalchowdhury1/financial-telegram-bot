/**
 * Per-field source cascade for /api/sheets (NotSoBoring, FrontRunner, VIX pill).
 *
 * Every field resolves on its own, so one broken sheet no longer drags the others
 * down, and every field carries its own provenance in `_meta.fields`:
 *
 *   NotSoBoring / FrontRunner:  sheet (primary URL) → sheet (alt URL)
 *                               → /tmp last-good → KV last-good → 'N/A'
 *   VIX current / 3M:           sheet → alt → CBOE daily CSV (completed sessions only:
 *                               live only when its close is the latest completed session
 *                               AND the market is not in regular hours; else stale)
 *                               → FRED VIXCLS / VXVCLS (lags a day ⇒ stale)
 *                               → /tmp last-good → KV last-good → 'N/A'
 *   VIX fear/greed tag:         lib/vixFearGreed.js (CBOE → FRED) → /tmp → KV → 'N/A'
 *                               (the retired sheet C2 is NEVER read — it froze at
 *                               GREED13 while the truth was GREED04)
 *
 * Last-good copies are PER FIELD, `{value, savedAt, source}`, in /tmp
 * (`sheets-fields`) and KV (`ftb:lg:sheets-fields`). Only live values (sheet or
 * CBOE) are ever saved; a served copy is relabelled `KV last-good (<savedAt>) ← <orig>`
 * and listed in `_meta.staleFields`. Fault test mode never writes either store.
 *
 * Fault switches (`?_fail=`): sheets_main, sheets_alt, sheets_cboe, sheets_fred,
 * sheets_cache (/tmp tier), sheets_kvlg (KV tier); the generic `lastgood` disables
 * both last-good tiers and `kvlg` the KV tier, same as serve().
 *
 * Nothing here throws: every tier is wrapped and degrades to the next one.
 */
import crypto from 'crypto';
import { GOOGLE_SHEETS } from './constants';
import { gate } from './faults';
import { parseCboeCsv } from './vol';
import { isStale } from './freshness';
import { parseEnvelope } from './kv';
import { dailyCloseStatus } from './marketClock';

export const TMP_KEY = 'sheets-fields';
export const KV_KEY = 'ftb:lg:sheets-fields';
const MARK_KEY = 'sheets-fields-kvmark';
// Same rationale as lib/store.js: a durable last-good copy needs no minute-level freshness
// (intraday VIX changes every minute, which used to mean ~1 SET/min per warm instance).
export const KV_REWRITE_MS = 60 * 60e3;
export const KV_MIN_GAP_MS = 60 * 60e3;

// Max age of a served last-good copy. NotSoBoring/FrontRunner are daily signals
// (a week is the store.js default); VIX values cover a weekend + holiday, no more.
export const FIELD_MAX_AGE_MS = {
    NotSoBoring: 7 * 864e5,
    FrontRunner: 7 * 864e5,
    vixCurrent: 4 * 864e5,
    vixThreeMonth: 4 * 864e5,
    vixFearGreed: 4 * 864e5,
};
export const FIELDS = Object.keys(FIELD_MAX_AGE_MS);
// The FrontRunner cell arrives as "BIL (T-Bill ETF)1": an n8n artifact digit glued after the
// closing paren (the Telegram bot's clean_val strips it too). Only digits right after ")" go.
export function frontRunnerText(cell) {
    return usableText((cell || '').split('\n')[0].trim().replace(/\)\d+$/, ')'));
}

const SHEET_FIELDS = ['NotSoBoring', 'FrontRunner', 'vixCurrent', 'vixThreeMonth'];
// The newest CBOE/FRED print must be at most this old to count at all (weekend + holiday).
const VIX_FRESH_DAYS = 4;

export const CBOE_URLS = {
    vixCurrent: 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
    vixThreeMonth: 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv',
};
const FRED_SERIES = { vixCurrent: 'VIXCLS', vixThreeMonth: 'VXVCLS' };

export const TIER = {
    main: 'Google Sheets (Live)',
    alt: 'Google Sheets (Alt URL)',
    cboe: 'CBOE',
    fred: 'FRED Proxy (VIX)',
    tmp: 'Cached',
    kv: 'KV last-good',
};

export function parseCSV(text) {
    return String(text || '').split('\n').map(row => {
        const result = [];
        let current = '';
        let inQuotes = false;
        for (const char of row) {
            if (char === '"') inQuotes = !inQuotes;
            else if (char === ',' && !inQuotes) { result.push(current); current = ''; }
            else current += char;
        }
        result.push(current);
        return result;
    });
}

// A cell is usable when it is non-empty and not a sheet error/placeholder (#N/A, #REF!,
// "Loading..."). VIX cells must also be numbers — a word there is never a VIX level.
const usableText = (v) => {
    const s = String(v ?? '').trim();
    return s && s.toUpperCase() !== 'N/A' && !s.startsWith('#') && !/^loading/i.test(s) ? s : null;
};
const usableNum = (v) => {
    const s = usableText(v);
    return s && Number.isFinite(parseFloat(s)) && /^-?\d/.test(s) ? s : null;
};

export const SHEETS = [
    { name: 'NotSoBoring', url: GOOGLE_SHEETS.NOT_SO_BORING, parse: (rows) => ({ NotSoBoring: usableText(rows[2]?.[1]) }) },
    { name: 'FrontRunner', url: GOOGLE_SHEETS.FRONT_RUNNER, parse: (rows) => ({ FrontRunner: frontRunnerText(rows[1]?.[0]) }) },
    // C2 (the retired fear/greed tag) is deliberately NOT parsed any more.
    { name: 'VIX', url: GOOGLE_SHEETS.VIX, parse: (rows) => ({ vixCurrent: usableNum(rows[1]?.[0]), vixThreeMonth: usableNum(rows[1]?.[1]) }) },
];

// Alternative export URL format for Google Sheets.
export function altUrl(url) {
    return url.includes('gid=') ? url.replace('export?format=csv', 'export?format=csv&output=csv') : url;
}

const live = (value, source, tier) => ({ value, source, tier, live: true, stale: false });

/** Newest point of a CBOE CSV, rejected when frozen (> VIX_FRESH_DAYS old). */
async function cboeLatest(field, fetchText, now) {
    const series = parseCboeCsv(await fetchText(CBOE_URLS[field], { revalidate: 0 }));
    const last = series[series.length - 1];
    if (!last) throw new Error('empty CSV');
    if (isStale(last.date, VIX_FRESH_DAYS, now)) throw new Error(`frozen (newest ${last.date})`);
    return last;
}

/** Newest valid FRED observation, rejected when > VIX_FRESH_DAYS old. */
async function fredLatest(field, apiKey, fetchJson, now) {
    if (!apiKey) throw new Error('FRED_API_KEY not configured');
    const data = await fetchJson(
        `https://api.stlouisfed.org/fred/series/observations?series_id=${FRED_SERIES[field]}&api_key=${apiKey}&file_type=json&sort_order=desc&limit=10`,
        { revalidate: 0 },
    );
    const obs = (data?.observations || []).find((o) => o && o.value !== '.' && Number.isFinite(parseFloat(o.value)));
    if (!obs) throw new Error('no valid observation');
    if (isStale(obs.date, VIX_FRESH_DAYS, now)) throw new Error(`frozen (newest ${obs.date})`);
    return { date: obs.date, value: parseFloat(obs.value) };
}

/**
 * Live tiers only. Returns `{fields, messages}`; a field no live tier could fill is null.
 */
export async function resolveLiveFields({ faults, fetchText, fetchJson, fredApiKey, now = new Date() } = {}) {
    const fields = Object.fromEntries(SHEET_FIELDS.map((f) => [f, null]));
    const messages = [];

    // Sheets: each sheet independently, primary URL then the alt URL.
    await Promise.all(SHEETS.map(async (sheet) => {
        for (const [tier, gateName, url] of [['main', 'sheets_main', sheet.url], ['alt', 'sheets_alt', altUrl(sheet.url)]]) {
            try {
                const parsed = await gate(gateName, faults, async () => sheet.parse(parseCSV(await fetchText(url))));
                const got = Object.entries(parsed).filter(([, v]) => v != null);
                for (const [f, v] of got) if (!fields[f]) fields[f] = live(v, TIER[tier], tier);
                if (got.length === Object.keys(parsed).length) return;
                messages.push(`${sheet.name} sheet (${tier}) had no usable value for ${Object.keys(parsed).filter((f) => parsed[f] == null).join(', ')}`);
            } catch (e) {
                messages.push(`${sheet.name} sheet (${tier}) failed: ${String(e?.message).slice(0, 120)}`);
            }
        }
    }));

    // VIX levels the sheet couldn't give: CBOE (same-day) → FRED (lags a trading day).
    for (const f of ['vixCurrent', 'vixThreeMonth']) {
        if (fields[f]) continue;
        const csv = CBOE_URLS[f].split('/').pop();
        try {
            const last = await gate('sheets_cboe', faults, () => cboeLatest(f, fetchText, now));
            const label = f === 'vixCurrent' ? 'current' : '3M';
            // The CSV holds COMPLETED sessions only. Intraday its newest row is yesterday's
            // close, not today's level: flagged stale (staleFields → pill + bot show 🕐) and
            // never persisted, exactly like the FRED tier. When it IS current it is saved
            // with savedAt = that session's close, so the copy can't outlive its real age.
            const st = dailyCloseStatus(last.date, new Date(now).getTime());
            if (st.current) {
                fields[f] = { ...live(last.value.toFixed(2), `CBOE ${csv} (close ${last.date})`, 'cboe'), ...(st.closeMs ? { asOfMs: st.closeMs } : {}) };
                messages.push(`VIX ${label} from CBOE ${csv} (${last.date})`);
            } else {
                const why = st.inSession ? 'market open: not today\'s level' : `latest completed session is ${st.expected}`;
                fields[f] = { value: last.value.toFixed(2), source: `CBOE ${csv} (close ${last.date}; ${why})`, live: false, stale: true, tier: 'cboe' };
                messages.push(`VIX ${label} from CBOE ${csv} close ${last.date} (STALE: ${why})`);
            }
            continue;
        } catch (e) { messages.push(`CBOE ${csv} failed: ${String(e?.message).slice(0, 120)}`); }
        try {
            const obs = await gate('sheets_fred', faults, () => fredLatest(f, fredApiKey, fetchJson, now));
            // A lagged official close is honest data but NOT today's value: flag it stale
            // and never save it as last-good.
            fields[f] = { value: obs.value.toFixed(2), source: `FRED ${FRED_SERIES[f]} (close ${obs.date}; lags a trading day)`, live: false, stale: true, tier: 'fred' };
            messages.push(`VIX ${f === 'vixCurrent' ? 'current' : '3M'} from FRED ${FRED_SERIES[f]} ${obs.date} (may lag a trading day)`);
        } catch (e) { messages.push(`FRED ${FRED_SERIES[f]} failed: ${String(e?.message).slice(0, 120)}`); }
    }
    return { fields, messages };
}

const tierOf = (field) => (field ? field.tier : null);

function fromRecord(record, f, label, now) {
    const r = record?.[f];
    if (!r || r.value == null || !r.savedAt) return null;
    if (now - Date.parse(r.savedAt) > FIELD_MAX_AGE_MS[f]) return null;
    return {
        value: r.value,
        source: `${label} (${r.savedAt}) ← ${r.source || 'unknown'}`,
        live: false,
        stale: true,
        savedAt: r.savedAt,
        tier: label === TIER.kv ? 'kv' : 'tmp',
    };
}

/**
 * Fill every still-null field from the per-field last-good copies: /tmp first, then KV.
 * Mutates and returns `fields`; adds a message per filled field. Never throws.
 */
export async function fillFromLastGood(fields, { faults, store, kv, now = Date.now() } = {}) {
    const messages = [];
    const has = (n) => !!(faults && faults.has(n));
    const missing = () => FIELDS.filter((f) => !fields[f]);
    if (!missing().length) return { fields, messages, kvRecord: undefined };

    if (!has('lastgood') && !has('sheets_cache')) {
        let rec = null;
        try { rec = store.load(TMP_KEY)?.data || null; } catch { /* ignore */ }
        for (const f of missing()) {
            const got = fromRecord(rec, f, '/tmp last-good', now);
            if (got) { fields[f] = got; messages.push(`${f}: served /tmp last-good from ${got.savedAt} (STALE)`); }
        }
    }
    let kvRecord;
    if (missing().length && !has('lastgood') && !has('kvlg') && !has('sheets_kvlg')) {
        try { kvRecord = parseEnvelope(await kv.get(KV_KEY))?.data || null; } catch { kvRecord = null; }
        for (const f of missing()) {
            const got = fromRecord(kvRecord, f, TIER.kv, now);
            if (got) { fields[f] = got; messages.push(`${f}: served KV last-good from ${got.savedAt} (STALE)`); }
        }
    }
    return { fields, messages, kvRecord };
}

/**
 * Save every LIVE field to /tmp (always) and KV (throttled: at most one SET per
 * KV_MIN_GAP_MS / KV_REWRITE_MS — 60 min — per instance). Merges with the existing copies so a field
 * that wasn't live this time keeps its older entry. Skipped entirely in fault test mode.
 * `kvRecord` = the KV record if fillFromLastGood already read it (saves a GET).
 */
export async function persistLiveFields(fields, { faults, store, kv, now = Date.now(), kvRecord } = {}) {
    try {
        if (faults && faults.size > 0) return false;
        const at = new Date(now).toISOString();
        const fresh = {};
        // A field may carry its real as-of time (CBOE: the session close); never stamp it newer.
        const savedAtOf = (x) => (Number.isFinite(x.asOfMs) && x.asOfMs < now ? new Date(x.asOfMs).toISOString() : at);
        for (const f of FIELDS) if (fields[f]?.live) fresh[f] = { value: fields[f].value, savedAt: savedAtOf(fields[f]), source: fields[f].source };
        if (!Object.keys(fresh).length) return false;

        let tmpRec = null;
        try { tmpRec = store.load(TMP_KEY)?.data || null; } catch { /* ignore */ }
        try { store.save(TMP_KEY, { ...(tmpRec || {}), ...fresh }); } catch { /* ignore */ }

        const hash = crypto.createHash('sha1')
            .update(JSON.stringify(Object.entries(fresh).map(([f, r]) => [f, r.value])))
            .digest('hex');
        let mark = null;
        try { mark = store.load(MARK_KEY)?.data || null; } catch { /* ignore */ }
        const age = mark ? now - Date.parse(mark.at) : Infinity;
        if (mark && (mark.hash === hash ? age < KV_REWRITE_MS : age < KV_MIN_GAP_MS)) return false;

        let base = kvRecord;
        if (base === undefined && Object.keys(fresh).length < FIELDS.length) {
            try { base = parseEnvelope(await kv.get(KV_KEY))?.data || null; } catch { base = null; }
        }
        const ok = await kv.set(KV_KEY, { data: { ...(base || {}), ...fresh }, savedAt: at });
        if (ok) { try { store.save(MARK_KEY, { hash, at }); } catch { /* ignore */ } }
        return !!ok;
    } catch {
        return false;
    }
}

/**
 * Top-level `_meta` summary. `source` keeps the old strings the status bar keys on:
 * all four pills from the primary sheet → 'Google Sheets (Live)'; nothing at all →
 * 'Static Defaults'; any stale field → prefixed 'Stale: '.
 */
export function summarize(fields) {
    const staleFields = FIELDS.filter((f) => fields[f]?.stale);
    const tiers = [...new Set(SHEET_FIELDS.map((f) => tierOf(fields[f])).filter(Boolean))];
    const allNA = SHEET_FIELDS.every((f) => !fields[f]);
    let source;
    if (allNA) source = 'Static Defaults';
    else {
        source = tiers.map((t) => TIER[t]).join(' + ');
        if (SHEET_FIELDS.some((f) => !fields[f])) source += ' (some N/A)';
        if (staleFields.length) source = `Stale: ${source}`;
    }
    const healthy = SHEET_FIELDS.every((f) => tierOf(fields[f]) === 'main');
    const meta = {};
    for (const f of FIELDS) {
        const x = fields[f];
        meta[f] = x ? { source: x.source, stale: !!x.stale, ...(x.savedAt ? { savedAt: x.savedAt } : {}) } : { source: 'Unavailable', stale: false };
    }
    return { source, hasErrors: !healthy || staleFields.length > 0, stale: staleFields.length > 0, staleFields, fields: meta };
}

/** The route's public shape (unchanged keys): N/A for anything unresolved. */
export function toResults(fields) {
    const v = (f) => fields[f]?.value ?? 'N/A';
    return {
        NotSoBoring: v('NotSoBoring'),
        FrontRunner: v('FrontRunner'),
        VIX: { current: v('vixCurrent'), threeMonth: v('vixThreeMonth'), fearGreed: v('vixFearGreed') },
    };
}
