import fs from 'fs';
import { GOOGLE_SHEETS } from '../../../lib/constants';
import { fetchText } from '../../../lib/fetcher';

export const dynamic = 'force-dynamic';
const CACHE_FILE = '/tmp/financial-dashboard-sheets-cache.json';

export async function GET() {
    try {
        const sheets = [
            {
                name: 'NotSoBoring',
                url: GOOGLE_SHEETS.NOT_SO_BORING,
                parse: (rows) => rows[2]?.[1]?.trim() || 'N/A'
            },
            {
                name: 'FrontRunner',
                url: GOOGLE_SHEETS.FRONT_RUNNER,
                parse: (rows) => (rows[1]?.[0]?.trim() || 'N/A').split('\n')[0].trim()
            },
            {
                name: 'AAIIDiff',
                url: GOOGLE_SHEETS.AAII,
                parse: (rows) => rows[1]?.[4]?.trim() || 'N/A'
            },
            {
                name: 'VIX',
                url: GOOGLE_SHEETS.VIX,
                parse: (rows) => ({
                    current: rows[1]?.[0]?.trim() || 'N/A',
                    threeMonth: rows[1]?.[1]?.trim() || 'N/A',
                    fearGreed: rows[1]?.[2]?.trim() || 'N/A'
                })
            }
        ];

        let results = {};
        let source = 'Google Sheets (Live)';
        let usedCache = false;
        let fetchFailed = false;

        try {
            const fetchPromises = sheets.map(async (sheet) => {
                const text = await fetchText(sheet.url);
                const rows = text.split('\n').map(row => {
                    const result = [];
                    let current = '';
                    let inQuotes = false;
                    for (const char of row) {
                        if (char === '"') { inQuotes = !inQuotes; }
                        else if (char === ',' && !inQuotes) { result.push(current); current = ''; }
                        else { current += char; }
                    }
                    result.push(current);
                    return result;
                });
                return { name: sheet.name, data: sheet.parse(rows) };
            });

            const resolved = await Promise.all(fetchPromises);
            clearTimeout(timeoutId);

            for (const r of resolved) {
                results[r.name] = r.data;
            }

            // Save valid fetch to cache
            fs.writeFileSync(CACHE_FILE, JSON.stringify(results));
        } catch (networkError) {
            clearTimeout(timeoutId);
            fetchFailed = true;
            console.warn('Google Sheets fetch failed/timed out. Falling back to cache...', networkError.message);

            try {
                if (fs.existsSync(CACHE_FILE)) {
                    const cachedData = fs.readFileSync(CACHE_FILE, 'utf8');
                    results = JSON.parse(cachedData);
                    source = 'Google Sheets (Cached Backup)';
                    usedCache = true;
                } else {
                    throw new Error('No cache file available');
                }
            } catch (cacheError) {
                return Response.json({
                    error: 'Live fetch failed and cache unavailable.',
                    _meta: { source: 'Failed', hasErrors: true, messages: ['Live fetch failed and `/tmp` cache unavailable on Vercel.'] }
                }, { status: 500 });
            }
        }

        return Response.json({
            ...results,
            _meta: {
                source: source,
                hasErrors: fetchFailed,
                messages: [usedCache ? `Loaded from local cache due to API failure` : `Loaded live data directly from Google`]
            }
        });
    } catch (error) {
        return Response.json({
            error: error.message,
            _meta: { source: 'Failed', hasErrors: true, messages: [error.message] }
        }, { status: 500 });
    }
}
