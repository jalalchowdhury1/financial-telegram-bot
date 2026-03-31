import fs from 'fs';
import { GOOGLE_SHEETS } from '../../../lib/constants';
import { fetchText, fetchJson } from '../../../lib/fetcher';

export const dynamic = 'force-dynamic';
const CACHE_FILE = '/tmp/financial-dashboard-sheets-cache.json';

// Static defaults — last reasonable values as ultimate fallback
const STATIC_DEFAULTS = {
    NotSoBoring: 'N/A',
    FrontRunner: 'N/A',
    AAIIDiff: '0.00%',
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
    { name: 'AAIIDiff', url: GOOGLE_SHEETS.AAII, parse: (rows) => rows[1]?.[4]?.trim() || 'N/A' },
    { name: 'VIX', url: GOOGLE_SHEETS.VIX, parse: (rows) => ({ current: rows[1]?.[0]?.trim() || 'N/A', threeMonth: rows[1]?.[1]?.trim() || 'N/A', fearGreed: rows[1]?.[2]?.trim() || 'N/A' }) },
    { name: 'SPYDallyMove', url: GOOGLE_SHEETS.SPY_DAILY_MOVE, parse: (rows) => rows[11]?.[1]?.trim() || 'N/A' }
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
    try { fs.writeFileSync(CACHE_FILE, JSON.stringify({ ...data, _cachedAt: new Date().toISOString() })); } catch { }
}

function loadCache() {
    try {
        if (fs.existsSync(CACHE_FILE)) return JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
    } catch { }
    return null;
}

export async function GET() {
    const messages = [];

    // Layer 1: Live Google Sheets (primary URLs)
    try {
        const results = await fetchSheets(SHEETS);
        saveCache(results);
        return Response.json({
            ...results,
            _meta: { source: 'Google Sheets (Live)', hasErrors: false, messages: ['Live data loaded'] }
        });
    } catch (e) { messages.push(`Layer 1 (Live Sheets) failed: ${e.message}`); }

    // Layer 2: /tmp cache from previous successful load
    const cached = loadCache();
    if (cached) {
        const isRecent = cached._cachedAt && (Date.now() - new Date(cached._cachedAt).getTime()) < 24 * 60 * 60 * 1000;
        if (isRecent) {
            messages.push(`Serving cache from ${cached._cachedAt}`);
            return Response.json({
                ...cached,
                _meta: { source: 'Google Sheets (Cached)', hasErrors: true, messages }
            });
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
        return Response.json({
            ...results,
            _meta: { source: 'Google Sheets (Alt URL)', hasErrors: true, messages }
        });
    } catch (e) { messages.push(`Layer 3 (Alt URL) failed: ${e.message}`); }

    // Layer 4: FRED/Yahoo proxies for VIX + AAII sentiment estimate
    try {
        const fredKey = process.env.FRED_API_KEY;
        let vixCurrent = 'N/A';
        let aaiDiff = 'N/A';

        if (fredKey) {
            // VIX from FRED VIXCLS
            try {
                const vixData = await fetchJson(
                    `https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key=${fredKey}&file_type=json&sort_order=desc&limit=5`,
                    { revalidate: 0 }
                );
                const vixObs = vixData.observations.filter(o => o.value !== '.');
                if (vixObs.length > 0) vixCurrent = parseFloat(vixObs[0].value).toFixed(2);
            } catch { }

            // AAII diff proxy from FRED UMCSENT (consumer sentiment)
            // Not a perfect match, but correlated — use as estimate
            try {
                const sentData = await fetchJson(
                    `https://api.stlouisfed.org/fred/series/observations?series_id=UMCSENT&api_key=${fredKey}&file_type=json&sort_order=desc&limit=2`,
                    { revalidate: 0 }
                );
                const sentObs = sentData.observations.filter(o => o.value !== '.');
                if (sentObs.length >= 2) {
                    const diff = parseFloat(sentObs[0].value) - parseFloat(sentObs[1].value);
                    aaiDiff = `${diff >= 0 ? '+' : ''}${diff.toFixed(2)}% (UMCSENT proxy)`;
                }
            } catch { }
        }

        const results = {
            ...(cached || STATIC_DEFAULTS),
            VIX: { current: vixCurrent, threeMonth: 'N/A', fearGreed: 'N/A' },
            AAIIDiff: aaiDiff
        };
        return Response.json({
            ...results,
            _meta: { source: 'FRED Proxy (VIX + sentiment)', hasErrors: true, messages }
        });
    } catch (e) { messages.push(`Layer 4 (FRED proxy) failed: ${e.message}`); }

    // Layer 5: Stale cache (even if >24h) or hardcoded defaults
    if (cached) {
        messages.push(`Serving stale cache from ${cached._cachedAt}`);
        return Response.json({
            ...cached,
            _meta: { source: 'Stale Cache (all sources failed)', hasErrors: true, messages }
        });
    }

    messages.push('All 5 layers failed — returning static defaults');
    return Response.json({
        ...STATIC_DEFAULTS,
        _meta: { source: 'Static Defaults', hasErrors: true, messages }
    });
}
