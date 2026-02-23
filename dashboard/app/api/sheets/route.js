import fs from 'fs';

export const dynamic = 'force-dynamic';
const CACHE_FILE = '/tmp/financial-dashboard-sheets-cache.json';

export async function GET() {
    try {
        const sheets = [
            {
                name: 'NotSoBoring',
                url: 'https://docs.google.com/spreadsheets/d/10Y8Jus8_fMwH9H69vWh7thSzl2hH34Ri3BRbDw_GEgw/export?format=csv&gid=0',
                parse: (rows) => rows[2]?.[1]?.trim() || 'N/A'
            },
            {
                name: 'FrontRunner',
                url: 'https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=1668420064',
                parse: (rows) => (rows[1]?.[0]?.trim() || 'N/A').split('\n')[0].trim()
            },
            {
                name: 'AAIIDiff',
                url: 'https://docs.google.com/spreadsheets/d/1zQQ2am1yhzTwY7nx8xPak4Q0WoNMwxWj7Ekr-fDEIF4/export?format=csv&gid=0',
                parse: (rows) => rows[1]?.[4]?.trim() || 'N/A'
            },
            {
                name: 'VIX',
                url: 'https://docs.google.com/spreadsheets/d/1vdlPNlT6gRpzMHuQUT7olqUNb455CQM3ab4wPuCE5R0/export?format=csv&gid=790638481',
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

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5.0s timeout for Vercel

        try {
            const fetchPromises = sheets.map(async (sheet) => {
                const res = await fetch(sheet.url, { next: { revalidate: 0 }, signal: controller.signal });
                if (!res.ok) throw new Error(`Status ${res.status}`);
                const text = await res.text();
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
