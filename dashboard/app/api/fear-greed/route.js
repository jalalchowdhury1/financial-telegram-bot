import fs from 'fs';
import { EXTERNAL_URLS, DEFAULT_HEADERS } from '../../../lib/constants';
import { proxyFetch, fetchJson } from '../../../lib/fetcher';

export const dynamic = 'force-dynamic';

const CACHE_FILE = '/tmp/fear-greed-cache.json';

function getRatingFromScore(score) {
    if (score < 25) return 'EXTREME FEAR';
    if (score < 45) return 'FEAR';
    if (score <= 55) return 'NEUTRAL';
    if (score <= 75) return 'GREED';
    return 'EXTREME GREED';
}

function vixToScore(vix) {
    if (vix == null || isNaN(vix)) return null;
    return Math.max(0, Math.min(100, 100 - ((vix - 10) / 25) * 100));
}

function saveCache(data) {
    try { fs.writeFileSync(CACHE_FILE, JSON.stringify({ ...data, cachedAt: new Date().toISOString() })); } catch {}
}

function loadCache() {
    try {
        if (fs.existsSync(CACHE_FILE)) return JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
    } catch {}
    return null;
}

export async function GET() {
    const messages = [];

    // Layer 1: CNN Business API
    try {
        const res = await proxyFetch(EXTERNAL_URLS.CNN_FEAR_GREED, {
            headers: { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36', 'Referer': 'https://edition.cnn.com/', 'Accept': 'application/json' },
            next: { revalidate: 0 }
        });
        if (!res.ok) throw new Error(`CNN returned ${res.status}`);
        const data = await res.json();
        const fg = data.fear_and_greed;
        const result = {
            score: fg?.score ?? 'N/A',
            rating: fg?.rating?.toUpperCase() || getRatingFromScore(fg?.score),
            previousClose: fg?.previous_close ?? 'N/A',
            previousWeek: fg?.previous_1_week ?? 'N/A',
            previousMonth: fg?.previous_1_month ?? 'N/A',
            previousYear: fg?.previous_1_year ?? 'N/A',
            _meta: { source: 'CNN', hasErrors: false, messages: ['CNN parsed successfully'] }
        };
        saveCache(result);
        return Response.json(result);
    } catch (e) { messages.push(`Layer 1 (CNN) failed: ${e.message}`); }

    // Layer 2: RapidAPI
    try {
        const rapidApiKey = process.env.RAPIDAPI_KEY || '7a22060824mshabd4fb2494530d3p1cee20jsnb5257930e869';
        const res = await proxyFetch(EXTERNAL_URLS.RAPIDAPI_FEAR_GREED, {
            headers: { 'X-RapidAPI-Key': rapidApiKey, 'X-RapidAPI-Host': 'fear-and-greed-index.p.rapidapi.com' },
            next: { revalidate: 0 }
        });
        if (!res.ok) throw new Error(`RapidAPI returned ${res.status}`);
        const data = await res.json();
        const fg = data.fgi;
        if (!fg?.now) throw new Error('RapidAPI data malformed');
        const result = {
            score: fg.now?.value ?? 'N/A',
            rating: fg.now?.valueText?.toUpperCase() || getRatingFromScore(fg.now?.value),
            previousClose: fg.previousClose?.value ?? 'N/A',
            previousWeek: fg.oneWeekAgo?.value ?? 'N/A',
            previousMonth: fg.oneMonthAgo?.value ?? 'N/A',
            previousYear: fg.oneYearAgo?.value ?? 'N/A',
            _meta: { source: 'RapidAPI', hasErrors: false, messages }
        };
        saveCache(result);
        return Response.json(result);
    } catch (e) { messages.push(`Layer 2 (RapidAPI) failed: ${e.message}`); }

    // Layer 3: Yahoo Finance ^VIX proxy
    try {
        const res = await proxyFetch(EXTERNAL_URLS.YAHOO_VIX, { headers: DEFAULT_HEADERS, next: { revalidate: 0 } });
        if (!res.ok) throw new Error(`Yahoo VIX returned ${res.status}`);
        const data = await res.json();
        const quotes = data.chart.result[0].indicators.quote[0].close;
        const n = quotes.length;
        if (n < 1) throw new Error('No VIX data');
        const score = vixToScore(quotes[n - 1]);
        const result = {
            score,
            rating: getRatingFromScore(score),
            previousClose: vixToScore(n > 1 ? quotes[n - 2] : null),
            previousWeek: vixToScore(n > 5 ? quotes[n - 6] : null),
            previousMonth: vixToScore(n > 21 ? quotes[n - 22] : null),
            previousYear: 'N/A',
            _meta: { source: 'Yahoo ^VIX Proxy', hasErrors: true, messages }
        };
        saveCache(result);
        return Response.json(result);
    } catch (e) { messages.push(`Layer 3 (Yahoo VIX) failed: ${e.message}`); }

    // Layer 4: FRED VIXCLS (official VIX close from Fed Reserve — uses existing API key)
    try {
        const fredKey = process.env.FRED_API_KEY;
        if (!fredKey) throw new Error('No FRED key');
        const data = await fetchJson(
            `https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key=${fredKey}&file_type=json&sort_order=desc&limit=260`,
            { revalidate: 0 }
        );
        const obs = data.observations.filter(o => o.value !== '.').map(o => parseFloat(o.value));
        if (obs.length < 1) throw new Error('No FRED VIXCLS data');
        const score = vixToScore(obs[0]);
        const result = {
            score,
            rating: getRatingFromScore(score),
            previousClose: vixToScore(obs[1] ?? null),
            previousWeek: vixToScore(obs[5] ?? null),
            previousMonth: vixToScore(obs[21] ?? null),
            previousYear: vixToScore(obs[252] ?? null),
            _meta: { source: 'FRED VIXCLS Proxy', hasErrors: true, messages }
        };
        saveCache(result);
        return Response.json(result);
    } catch (e) { messages.push(`Layer 4 (FRED VIXCLS) failed: ${e.message}`); }

    // Layer 5: Stale /tmp cache
    const cached = loadCache();
    if (cached) {
        messages.push(`Serving stale cache from ${cached.cachedAt}`);
        return Response.json({
            ...cached,
            _meta: { source: 'Stale Cache', hasErrors: true, messages }
        });
    }

    messages.push('Layer 5 (cache) empty');
    return Response.json({
        score: 'N/A', rating: 'N/A', previousClose: 'N/A', previousWeek: 'N/A', previousMonth: 'N/A', previousYear: 'N/A',
        _meta: { source: 'Failed', hasErrors: true, messages }
    }, { status: 500 });
}
