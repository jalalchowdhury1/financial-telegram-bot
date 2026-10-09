import fs from 'fs';
import { GOOGLE_SHEETS } from '../../../lib/constants';
import { fetchText, fetchJson } from '../../../lib/fetcher';
import { faultsFrom } from '../../../lib/faults';
import { cacheHeaders } from '../../../lib/cdn';
import { resolveVixFearGreedTag } from '../../../lib/vixFearGreed';
import { loadLastGood, saveLastGood } from '../../../lib/store';
import { resolveAaii } from '../../../lib/aaii';
import { defaultKv } from '../../../lib/factorStore';
import bakedAaii from '../../../lib/data/aaiiNewest.json';

export const dynamic = 'force-dynamic';
const CACHE_FILE = '/tmp/financial-dashboard-sheets-cache.json';

// Static defaults — last reasonable values as ultimate fallback
const STATIC_DEFAULTS = {
    NotSoBoring: 'N/A',
    FrontRunner: 'N/A',
    AAIIDiff: 'N/A',
    VIX: { current: 'N/A', threeMonth: 'N/A', fearGreed: 'N/A' }
};

function parseCSV(text) {
    return text.split('\n').map(row => {
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

const SHEETS = [
    { name: 'NotSoBoring', url: GOOGLE_SHEETS.NOT_SO_BORING, parse: (rows) => rows[2]?.[1]?.trim() || 'N/A' },
    { name: 'FrontRunner', url: GOOGLE_SHEETS.FRONT_RUNNER, parse: (rows) => (rows[1]?.[0]?.trim() || 'N/A').split('\n')[0].trim() },
    // AAIIDiff is NOT read from a sheet any more: it comes straight from AAII
    // (lib/aaii.js, the same resolver as /api/aaii) and is laid over whichever
    // layer below won — see GET(). The old AAII sheet's writer key leaked (2026-09-27).
    { name: 'VIX', url: GOOGLE_SHEETS.VIX, parse: (rows) => ({ current: rows[1]?.[0]?.trim() || 'N/A', threeMonth: rows[1]?.[1]?.trim() || 'N/A', fearGreed: rows[1]?.[2]?.trim() || 'N/A' }) }
];

// Alternative export URL formats for Google Sheets
function altUrl(url) {
    // Try /export?format=csv variant if the URL uses /export?format=csv&gid=...
    return url.includes('gid=') ? url.replace('export?format=csv', 'export?format=csv&output=csv') : url;
}

async function fetchSheets(sheets) {
    const resolved = await Promise.all(sheets.map(async (sheet) => {
        const text = await fetchText(sheet.url);
        return { name: sheet.name, data: sheet.parse(parseCSV(text)) };
    }));
    const results = {};
    for (const r of resolved) results[r.name] = r.data;
    return results;
}

function saveCache(data) {
    try { fs.writeFileSync(CACHE_FILE, JSON.stringify({ ...data, _cachedAt: new Date().toISOString() })); } catch {}
}

function loadCache() {
    try {
        if (fs.existsSync(CACHE_FILE)) return JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
    } catch {}
    return null;
}

/**
 * The original 5-layer Google Sheets cascade, unchanged in behavior — just
 * factored out of GET() so it returns `{results, source, hasErrors, messages}`
 * instead of returning a Response directly. This gives GET() exactly ONE
 * place to layer the FRED-computed VIX fear/greed override on top of
 * whichever layer won (see resolveVixFearGreedTag in lib/vixFearGreed.js),
 * regardless of which of these 5 layers supplied the rest of the payload.
 */
async function resolveSheetsCascade() {
    const messages = [];

    // Layer 1: Live Google Sheets (primary URLs)
    try {
        const results = await fetchSheets(SHEETS);
        saveCache(results);
        return { results, source: 'Google Sheets (Live)', hasErrors: false, messages: ['Live data loaded'] };
    } catch (e) { messages.push(`Layer 1 (Live Sheets) failed: ${e.message}`); }

    // Layer 2: /tmp cache from previous successful load
    const cached = loadCache();
    if (cached) {
        const isRecent = cached._cachedAt && (Date.now() - new Date(cached._cachedAt).getTime()) < 24 * 60 * 60 * 1000;
        if (isRecent) {
            messages.push(`Serving cache from ${cached._cachedAt}`);
            return { results: cached, source: 'Google Sheets (Cached)', hasErrors: true, messages };
        }
        messages.push('Cache exists but is stale (>24h), trying other sources');
    } else {
        messages.push('Layer 2 (cache) empty');
    }

    // Layer 3: Alternative Google Sheets export URL format
    try {
        const altSheets = SHEETS.map(s => ({ ...s, url: altUrl(s.url) }));
        const results = await fetchSheets(altSheets);
        saveCache(results);
        return { results, source: 'Google Sheets (Alt URL)', hasErrors: true, messages };
    } catch (e) { messages.push(`Layer 3 (Alt URL) failed: ${e.message}`); }

    // Layer 4: FRED proxy for VIX (AAII is resolved separately in GET())
    try {
        const fredKey = process.env.FRED_API_KEY;
        let vixCurrent = 'N/A';

        if (fredKey) {
            // VIX from FRED VIXCLS
            try {
                const vixData = await fetchJson(
                    `https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key=${fredKey}&file_type=json&sort_order=desc&limit=5`,
                    { revalidate: 0 }
                );
                const vixObs = vixData.observations.filter(o => o.value !== '.');
                if (vixObs.length > 0) vixCurrent = parseFloat(vixObs[0].value).toFixed(2);
            } catch {}
        }

        const results = {
            ...(cached || STATIC_DEFAULTS),
            VIX: { current: vixCurrent, threeMonth: 'N/A', fearGreed: 'N/A' },
        };
        return { results, source: 'FRED Proxy (VIX)', hasErrors: true, messages };
    } catch (e) { messages.push(`Layer 4 (FRED proxy) failed: ${e.message}`); }

    // Layer 5: Stale cache (even if >24h) or hardcoded defaults
    if (cached) {
        messages.push(`Serving stale cache from ${cached._cachedAt}`);
        return { results: cached, source: 'Stale Cache (all sources failed)', hasErrors: true, messages };
    }

    messages.push('All 5 layers failed — returning static defaults');
    return { results: STATIC_DEFAULTS, source: 'Static Defaults', hasErrors: true, messages };
}

export async function GET(request) {
    const faults = faultsFrom(request);
    // AAII straight from AAII (lib/aaii.js), in parallel with the sheet cascade. Never
    // throws out of here: any failure = 'N/A' + hasErrors, never an old sheet value.
    const aaiiPromise = resolveAaii({ fetchText, store: { load: loadLastGood, save: saveLastGood }, kv: defaultKv, baked: bakedAaii, faults })
        .catch((e) => ({ payload: null, messages: [`aaii resolver threw: ${String(e?.message).slice(0, 120)}`] }));
    const [cascade, aaii] = await Promise.all([resolveSheetsCascade(), aaiiPromise]);
    const { results: sheetResults, source, messages } = cascade;
    let { hasErrors } = cascade;
    const a = aaii.payload;
    const results = {
        ...sheetResults,
        AAIIDiff: a ? a.diff : 'N/A',
        AAII: a
            ? { bull: a.bull, neutral: a.neutral, bear: a.bear, as_of: a.as_of, source: a.source, stale: a.stale, lastGood: !!aaii.lastGood }
            : null,
    };
    if (!a) {
        hasErrors = true;
        messages.push(`AAII unavailable: ${aaii.messages.join(' | ')}`);
    } else {
        messages.push(`AAII ${a.as_of} via ${a.source}${aaii.cachedAt ? ` (cached ${aaii.cachedAt})` : ''}`);
        if (a.stale) { hasErrors = true; messages.push(`AAII is STALE: survey ${a.as_of} is more than 9 days old`); }
        if (aaii.lastGood) { hasErrors = true; messages.push('AAII from the last good copy: every live tier failed'); }
    }

    // VIX fear/greed tag (the pill's "GREED13"-style value): FRED-computed is
    // now the primary source (folds in the vix-fear-greed repo's formula —
    // see lib/vixFearGreed.js), the sheet's C2 value already sitting in
    // `results.VIX.fearGreed` (from whichever cascade layer above won) is the
    // transition fallback for as long as that repo still writes it, and
    // 'N/A' is the last resort. Which source actually won is always recorded
    // in `messages` so a silent fall-back to the sheet is never invisible.
    const sheetFearGreed = results?.VIX?.fearGreed ?? 'N/A';
    const { tag: vixFearGreed, message: fearGreedMessage, fallback: tagFallback = false } = await resolveVixFearGreedTag({
        fredApiKey: process.env.FRED_API_KEY,
        fetchJson,
        fetchText,
        sheetValue: sheetFearGreed,
        faults
    });
    messages.push(fearGreedMessage);

    const finalResults = results?.VIX
        ? { ...results, VIX: { ...results.VIX, fearGreed: vixFearGreed } }
        : results;

    const body = { ...finalResults, _meta: { source, hasErrors, messages } };
    // Edge-cached only when healthy and not a fault test (lib/cdn.js). A VIX tag that
    // fell back to the sheet's own value is not healthy, though hasErrors stays false.
    return Response.json(body, { headers: cacheHeaders('sheets', { payload: body, testMode: faults.size > 0, degraded: tagFallback }) });
}
