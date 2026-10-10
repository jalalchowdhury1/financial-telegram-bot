import { EXTERNAL_URLS, DEFAULT_HEADERS } from '../../../lib/constants';
import { proxyFetch, fetchJson, fetchText } from '../../../lib/fetcher';
import { parseCboeCsv } from '../../../lib/vol';
import { CBOE_VIX_URL } from '../../../lib/vixFearGreed';
import { cacheHeaders } from '../../../lib/cdn';
import { faultsFrom, gate } from '../../../lib/faults';
import { isStale } from '../../../lib/freshness';
import { loadLastGood, saveLastGood, loadLastGoodKV, saveLastGoodKV } from '../../../lib/store';

export const dynamic = 'force-dynamic';

/**
 * CNN's 0–100 Fear & Greed index. Layers, each behind a `?_fail=` switch:
 *   1. CNN dataviz API                     `cnn`
 *   2. RapidAPI (a CNN F&G reseller)       `rapidapi`
 *   3. Yahoo ^VIX proxy                    `fg_yahoo`  ┐ NOT the CNN index: a VIX level
 *   4. CBOE VIX_History.csv proxy          `fg_cboe`   │ mapped onto 0–100. `_meta.proxy`
 *   5. FRED VIXCLS proxy (lags a day)      `fg_fred`   ┘ + `_meta.note` say so, the card shows it.
 *      (CBOE added 2026-10-09: Yahoo answers Node's fetch with 429 even when curl gets 200.)
 *   6. /tmp last-good (CNN/RapidAPI only)  `fg_cache`  ┐ `lastgood` disables both,
 *   7. KV last-good  `ftb:lg:fear-greed`   `fg_kvlg`   ┘ `kvlg` only the KV tier
 *   8. N/A (HTTP 500 + `error`, so loadJson retries and the card shows its skeleton)
 * Only a real CNN-index answer is ever saved as last-good (a proxy must never be
 * replayed later as if it were the index), and never on a fault test.
 */
const LG_KEY = 'fear-greed';
const LG_MAX_AGE_MS = 3 * 864e5;   // a weekend; older than that is not "the index" any more
const FRESH_DAYS = 4;              // a source whose newest print is older than this is frozen
const PROXY_NOTE = 'VIX-derived proxy: NOT the CNN Fear & Greed index';

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

async function fromCnn() {
    const res = await proxyFetch(EXTERNAL_URLS.CNN_FEAR_GREED, {
        headers: { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36', 'Referer': 'https://edition.cnn.com/', 'Accept': 'application/json' },
        next: { revalidate: 0 }
    });
    if (!res.ok) throw new Error(`CNN returned ${res.status}`);
    const data = await res.json();
    const fg = data.fear_and_greed;
    if (!Number.isFinite(fg?.score)) throw new Error('CNN data malformed');
    if (fg.timestamp && isStale(fg.timestamp, FRESH_DAYS)) throw new Error(`CNN frozen (timestamp ${fg.timestamp})`);
    return {
        score: fg.score,
        rating: fg.rating?.toUpperCase() || getRatingFromScore(fg.score),
        previousClose: fg.previous_close ?? 'N/A',
        previousWeek: fg.previous_1_week ?? 'N/A',
        previousMonth: fg.previous_1_month ?? 'N/A',
        previousYear: fg.previous_1_year ?? 'N/A',
        ...(fg.timestamp ? { asOf: fg.timestamp } : {}),
        _meta: { source: 'CNN', hasErrors: false, messages: ['CNN parsed successfully'] }
    };
}

async function fromRapidApi(messages) {
    // No hardcoded fallback — a committed key is a leaked key. If RAPIDAPI_KEY
    // isn't set, skip this layer (CNN is primary; the proxies follow).
    const rapidApiKey = process.env.RAPIDAPI_KEY;
    if (!rapidApiKey) throw new Error('RAPIDAPI_KEY not configured — skipping RapidAPI layer');
    const res = await proxyFetch(EXTERNAL_URLS.RAPIDAPI_FEAR_GREED, {
        headers: { 'X-RapidAPI-Key': rapidApiKey, 'X-RapidAPI-Host': 'fear-and-greed-index.p.rapidapi.com' },
        next: { revalidate: 0 }
    });
    if (!res.ok) throw new Error(`RapidAPI returned ${res.status}`);
    const data = await res.json();
    const fg = data.fgi;
    if (!Number.isFinite(fg?.now?.value)) throw new Error('RapidAPI data malformed');
    return {
        score: fg.now.value,
        rating: fg.now?.valueText?.toUpperCase() || getRatingFromScore(fg.now?.value),
        previousClose: fg.previousClose?.value ?? 'N/A',
        previousWeek: fg.oneWeekAgo?.value ?? 'N/A',
        previousMonth: fg.oneMonthAgo?.value ?? 'N/A',
        previousYear: fg.oneYearAgo?.value ?? 'N/A',
        _meta: { source: 'RapidAPI', hasErrors: false, messages: [...messages] }
    };
}

async function fromYahooVix(messages) {
    const res = await proxyFetch(EXTERNAL_URLS.YAHOO_VIX, { headers: DEFAULT_HEADERS, next: { revalidate: 0 } });
    if (!res.ok) throw new Error(`Yahoo VIX returned ${res.status}`);
    const data = await res.json();
    const r = data.chart.result[0];
    const ts = r.timestamp || [];
    const closes = r.indicators.quote[0].close || [];
    // Drop null bars (Yahoo's in-progress day) so "latest" is a real print.
    const pts = ts.map((t, i) => ({ date: new Date(t * 1000).toISOString().slice(0, 10), v: closes[i] }))
        .filter((p) => Number.isFinite(p.v));
    const n = pts.length;
    if (n < 1) throw new Error('No VIX data');
    if (isStale(pts[n - 1].date, FRESH_DAYS)) throw new Error(`Yahoo VIX frozen (newest ${pts[n - 1].date})`);
    const score = vixToScore(pts[n - 1].v);
    return {
        score,
        rating: getRatingFromScore(score),
        previousClose: vixToScore(n > 1 ? pts[n - 2].v : null),
        previousWeek: vixToScore(n > 5 ? pts[n - 6].v : null),
        previousMonth: vixToScore(n > 21 ? pts[n - 22].v : null),
        previousYear: 'N/A',
        asOf: pts[n - 1].date,
        _meta: { source: 'Yahoo ^VIX Proxy', proxy: true, note: PROXY_NOTE, hasErrors: true, messages: [...messages, `${PROXY_NOTE} (Yahoo ^VIX ${pts[n - 1].date})`] }
    };
}

// CBOE's own keyless daily VIX history (same CSV + parser as lib/vixFearGreed.js / /api/vol).
async function fromCboeVix(messages) {
    const pts = parseCboeCsv(await fetchText(CBOE_VIX_URL, { revalidate: 0 }));
    const n = pts.length;
    if (n < 1) throw new Error('No CBOE VIX data');
    if (isStale(pts[n - 1].date, FRESH_DAYS)) throw new Error(`CBOE VIX frozen (newest ${pts[n - 1].date})`);
    const score = vixToScore(pts[n - 1].value);
    return {
        score,
        rating: getRatingFromScore(score),
        previousClose: vixToScore(n > 1 ? pts[n - 2].value : null),
        previousWeek: vixToScore(n > 5 ? pts[n - 6].value : null),
        previousMonth: vixToScore(n > 21 ? pts[n - 22].value : null),
        previousYear: vixToScore(n > 252 ? pts[n - 253].value : null),
        asOf: pts[n - 1].date,
        _meta: { source: 'CBOE VIX Proxy', proxy: true, note: PROXY_NOTE, hasErrors: true, messages: [...messages, `${PROXY_NOTE} (CBOE VIX ${pts[n - 1].date})`] }
    };
}

async function fromFredVix(messages) {
    // FRED VIXCLS — official VIX close, uses the existing API key, one trading day behind.
    const fredKey = process.env.FRED_API_KEY;
    if (!fredKey) throw new Error('No FRED key');
    const data = await fetchJson(
        `https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key=${fredKey}&file_type=json&sort_order=desc&limit=260`,
        { revalidate: 0 }
    );
    const valid = data.observations.filter(o => o.value !== '.' && Number.isFinite(parseFloat(o.value)));
    if (valid.length < 1) throw new Error('No FRED VIXCLS data');
    if (isStale(valid[0].date, FRESH_DAYS + 1)) throw new Error(`FRED VIXCLS frozen (newest ${valid[0].date})`);
    const obs = valid.map(o => parseFloat(o.value));
    const score = vixToScore(obs[0]);
    return {
        score,
        rating: getRatingFromScore(score),
        previousClose: vixToScore(obs[1] ?? null),
        previousWeek: vixToScore(obs[5] ?? null),
        previousMonth: vixToScore(obs[21] ?? null),
        previousYear: vixToScore(obs[252] ?? null),
        asOf: valid[0].date,
        _meta: { source: 'FRED VIXCLS Proxy', proxy: true, stale: true, note: PROXY_NOTE, hasErrors: true, messages: [...messages, `${PROXY_NOTE} (FRED VIXCLS ${valid[0].date}, lags a trading day)`] }
    };
}

// A cached copy is relabelled so it can never read as live: the source names the copy
// and its savedAt (status bar keys on "Stale"), and `_meta.stale` is set.
function relabel(lg, label, messages) {
    const d = lg.data || {};
    return {
        ...d,
        _meta: {
            ...(d._meta || {}),
            source: `Stale ${label} (${lg.savedAt}) ← ${d._meta?.source || 'unknown'}`,
            hasErrors: true,
            stale: true,
            lastGoodAt: lg.savedAt,
            messages: [...messages, `serving ${label} from ${lg.savedAt}`],
        },
    };
}

export async function GET(request) {
    request.headers.get('user-agent'); // touch the request: keeps the route dynamic (AGENTS.md §3)
    const faults = faultsFrom(request);
    const testMode = faults.size > 0;
    const messages = [];
    const store = async (payload) => {
        if (testMode) return; // a fault test never writes /tmp or KV
        saveLastGood(LG_KEY, payload);
        await saveLastGoodKV(LG_KEY, payload);
    };

    try {
        // Layer 1: CNN Business API
        try {
            const result = await gate('cnn', faults, fromCnn);
            await store(result);
            // Only the healthy CNN answer is edge-cached (lib/cdn.js); every fallback layer is no-store.
            return Response.json(result, { headers: cacheHeaders('fear-greed', { payload: result, testMode }) });
        } catch (e) { messages.push(`Layer 1 (CNN) failed: ${e.message}`); }

        // Layer 2: RapidAPI
        try {
            const result = await gate('rapidapi', faults, () => fromRapidApi(messages));
            await store(result);
            return Response.json(result);
        } catch (e) { messages.push(`Layer 2 (RapidAPI) failed: ${e.message}`); }

        // Layer 3: Yahoo Finance ^VIX proxy (not saved as last-good)
        try {
            return Response.json(await gate('fg_yahoo', faults, () => fromYahooVix(messages)));
        } catch (e) { messages.push(`Layer 3 (Yahoo VIX) failed: ${e.message}`); }

        // Layer 4: CBOE VIX proxy (not saved as last-good)
        try {
            return Response.json(await gate('fg_cboe', faults, () => fromCboeVix(messages)));
        } catch (e) { messages.push(`Layer 4 (CBOE VIX) failed: ${e.message}`); }

        // Layer 5: FRED VIXCLS proxy (not saved as last-good)
        try {
            return Response.json(await gate('fg_fred', faults, () => fromFredVix(messages)));
        } catch (e) { messages.push(`Layer 5 (FRED VIXCLS) failed: ${e.message}`); }

        // Layer 6: /tmp last-good (the CNN index itself, flagged stale)
        if (!faults.has('lastgood') && !faults.has('fg_cache')) {
            const lg = loadLastGood(LG_KEY, LG_MAX_AGE_MS);
            if (lg) return Response.json(relabel(lg, 'cache', messages));
            messages.push('Layer 6 (/tmp cache) empty');
        } else messages.push('Layer 6 (/tmp cache) disabled');

        // Layer 7: KV last-good (survives cold starts)
        if (!faults.has('lastgood') && !faults.has('kvlg') && !faults.has('fg_kvlg')) {
            const lg = await loadLastGoodKV(LG_KEY, LG_MAX_AGE_MS);
            if (lg) return Response.json(relabel(lg, 'KV last-good', messages));
            messages.push('Layer 7 (KV last-good) empty');
        } else messages.push('Layer 7 (KV last-good) disabled');
    } catch (e) {
        messages.push(`fear-greed route error: ${String(e?.message).slice(0, 160)}`);
    }

    return Response.json({
        score: 'N/A', rating: 'N/A', previousClose: 'N/A', previousWeek: 'N/A', previousMonth: 'N/A', previousYear: 'N/A',
        error: 'Fear & Greed unavailable',
        _meta: { source: 'Failed', hasErrors: true, messages }
    }, { status: 500, headers: { 'cache-control': 'no-store' } });
}
