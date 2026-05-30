import { FRED_SERIES, FRED_FRESHNESS, EXTERNAL_URLS } from '../../../lib/constants';
import { fetchJson, proxyFetch } from '../../../lib/fetcher';
import { withFreshness } from '../../../lib/freshness';
import { serve } from '../../../lib/store';
import { faultsFrom } from '../../../lib/faults';

// In Next 13.5 the route's default fetchCache is 'only-no-store', which ERRORS
// on cached fetches. 'default-cache' permits caching (and never errors), so the
// FRED calls below can use the 30-min Data Cache. Reading the request in GET()
// keeps the handler running per request (fresh fetchedAt, no build-time bake).
export const fetchCache = 'default-cache';

const REVALIDATE_SECONDS = 1800; // 30 minutes
const RETRY_DELAYS_MS = [400, 900, 1800]; // back-off on 429

// Hide the API key if it ever ends up in an error message / URL.
const maskKey = (s) => (typeof s === 'string' ? s.replace(/api_key=[^&\s]+/g, 'api_key=***') : s);

// ─────────────────────────────────────────────────────────────────────────────
// FRED series fetch — 429 back-off, wrapped in a 30-min server cache.
//
// unstable_cache stores the *result* for 30 min regardless of force-dynamic, and
// serves it on a cache hit WITHOUT calling FRED again. That is both the fix for
// the 429 rate-limiting AND the "remember each number for 30 minutes" behavior:
// a throttled load reuses the cached copy instead of blanking out. A thrown error
// (final 429) is NOT cached, so the next load retries.
// ─────────────────────────────────────────────────────────────────────────────

async function fetchSeriesRaw(seriesId, apiKey, limit) {
    const url = `${EXTERNAL_URLS.FRED_BASE}?series_id=${seriesId}&api_key=${apiKey}&file_type=json&sort_order=desc&limit=${limit}`;
    let lastErr;
    for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
        try {
            // Cached in the Data Cache for 30 min (fetchCache='default-cache').
            const data = await fetchJson(url, { revalidate: REVALIDATE_SECONDS });
            return data.observations
                .filter(o => o.value !== '.')
                .map(o => ({ date: o.date, value: parseFloat(o.value) }));
        } catch (e) {
            lastErr = e;
            const is429 = /\b429\b/.test(e?.message || '');
            if (!is429 || attempt === RETRY_DELAYS_MS.length) break;
            await new Promise(r => setTimeout(r, RETRY_DELAYS_MS[attempt]));
        }
    }
    // Re-throw with the key masked so it never leaks into _meta.messages.
    throw new Error(maskKey(lastErr?.message || `Failed to fetch ${seriesId}`));
}

function fetchSeries(seriesId, apiKey, limit = 15) {
    return fetchSeriesRaw(seriesId, apiKey, limit);
}

// Cache the slow HTML P/E scrapes the same way (30 min).
const cachedText = async (key, url, ms) => {
    const res = await proxyFetch(url, { revalidate: REVALIDATE_SECONDS, timeout: ms });
    return res.text();
};

function findByMonthOffset(arr, nMonths) {
    if (!arr?.length) return undefined;
    const target = new Date(arr[0].date);
    target.setMonth(target.getMonth() - nMonths);
    return arr.reduce((best, obs) => {
        const d = Math.abs(new Date(obs.date) - target);
        return d < Math.abs(new Date(best.date) - target) ? obs : best;
    });
}

const dateOf = (arr) => arr?.[0]?.date ?? null;

// ─────────────────────────────────────────────────────────────────────────────
// Build the dashboard payload, stamping each metric with asOf + staleness.
// A value older than its series' freshness deadline (or missing) becomes null,
// so the UI shows N/A — never a misleadingly old number.
// ─────────────────────────────────────────────────────────────────────────────

function buildResponse(series, peRatio, now) {
    const {
        T10Y2Y: t10y2y, UNRATE: unrate, UMCSENT: umcsent, ICSA: icsa, BAMLC0A4CBBB: bbb,
        DFII10: dfii10, USSLIND: usslind, NFCI: nfci, M2SL: m2sl, RSXFS: rsxfs,
        HOUST: houst, INDPRO: indpro, JTSJOL: jtsjol, DGORDER: dgorder, PSAVERT: psavert,
        A053RC1Q027SBEA: corpProfits, GDP: gdpData, USREC: usrec,
    } = series;

    const F = (value, seriesId, asOf, extra = {}) => ({
        ...withFreshness(value, asOf, FRED_FRESHNESS[seriesId], now),
        ...extra,
    });

    // Recession shading
    const recessionPeriods = [];
    const recSorted = [...usrec].reverse();
    let recStart = null;
    for (let i = 0; i < recSorted.length; i++) {
        if (recSorted[i].value === 1 && recStart === null) recStart = recSorted[i].date;
        else if (recSorted[i].value === 0 && recStart !== null) {
            recessionPeriods.push({ start: recStart, end: recSorted[i].date });
            recStart = null;
        }
    }
    if (recStart !== null) recessionPeriods.push({ start: recStart, end: recSorted[recSorted.length - 1].date });

    // ── Top cards ──
    const yc = withFreshness(t10y2y[0]?.value, dateOf(t10y2y), FRED_FRESHNESS.T10Y2Y, now);
    const yieldCurve = { current: yc.value, asOf: yc.asOf, stale: yc.stale, date: t10y2y[0]?.date, history: [...t10y2y].reverse() };

    const gdpMap = new Map();
    for (const gd of gdpData) gdpMap.set(gd.date, gd.value);
    const profitMarginHistory = [];
    for (const cp of corpProfits) {
        const gdpValue = gdpMap.get(cp.date);
        if (gdpValue && gdpValue !== 0) profitMarginHistory.push({ date: cp.date, value: (cp.value / gdpValue) * 100 });
    }
    const pm = withFreshness(profitMarginHistory[0]?.value, profitMarginHistory[0]?.date, FRED_FRESHNESS.A053RC1Q027SBEA, now);
    const profitMargin = { current: pm.value, asOf: pm.asOf, stale: pm.stale, date: profitMarginHistory[0]?.date || '', history: [...profitMarginHistory].reverse() };

    // ── Economic indicators ──
    const unrate3mo = unrate.length >= 3 ? unrate.slice(0, 3).reduce((s, v) => s + v.value, 0) / 3 : null;
    const unrate12moLow = unrate.length > 0 ? Math.min(...unrate.map(u => u.value)) : null;
    const sahmRule = unrate3mo !== null && unrate12moLow !== null ? unrate3mo - unrate12moLow : undefined;

    const sentimentCurrent = umcsent[0]?.value;
    const sentimentPrev = umcsent[1]?.value;
    const claims4wk = icsa.length >= 4 ? icsa.slice(0, 4).reduce((s, v) => s + v.value, 0) / 4 : undefined;
    const bbbCurrent = bbb[0]?.value;
    const tipsCurrent = dfii10[0]?.value;
    const tipsPrev = dfii10[1]?.value;
    const leiCurrent = usslind[0]?.value;
    const leiPrev = usslind[1]?.value;
    const leiChange = leiPrev ? ((leiCurrent - leiPrev) / leiPrev) * 100 : leiCurrent;

    // ── Bull checklist ──
    const nfciCurrent = nfci[0]?.value;
    const m2Current = m2sl[0]?.value;
    const m2YearAgo = findByMonthOffset(m2sl, 12)?.value;
    const m2Growth = m2YearAgo ? ((m2Current - m2YearAgo) / m2YearAgo) * 100 : undefined;
    const retailCurrent = rsxfs[0]?.value;
    const retail3mo = findByMonthOffset(rsxfs, 3)?.value;
    const retailGrowth = retail3mo ? ((retailCurrent - retail3mo) / retail3mo) * 100 : undefined;
    const housingCurrent = houst[0]?.value;
    const housing6moAvg = houst.length >= 6 ? houst.slice(0, 6).reduce((s, v) => s + v.value, 0) / 6 : 0;
    const indproCurrent = indpro[0]?.value;
    const indpro6mo = findByMonthOffset(indpro, 6)?.value;
    const indproChange = indpro6mo ? ((indproCurrent - indpro6mo) / indpro6mo) * 100 : undefined;
    const joltsCurrent = jtsjol[0]?.value;
    const durableCurrent = dgorder[0]?.value;
    const durable3mo = findByMonthOffset(dgorder, 3)?.value;
    const durableChange = durable3mo ? ((durableCurrent - durable3mo) / durable3mo) * 100 : undefined;
    const savingsCurrent = psavert[0]?.value;

    return {
        yieldCurve,
        profitMargin,
        peRatio,
        peRatioAsOf: now.toISOString(), // scraped live each cache cycle
        recessions: recessionPeriods,
        indicators: {
            sahmRule: F(sahmRule, 'UNRATE', dateOf(unrate), { status: sahmRule >= 0.5 ? 'danger' : 'safe' }),
            sentiment: F(sentimentCurrent, 'UMCSENT', dateOf(umcsent), { change: sentimentCurrent - sentimentPrev, status: sentimentCurrent > 80 ? 'strong' : sentimentCurrent > 60 ? 'neutral' : 'weak' }),
            claims: F(claims4wk !== undefined ? claims4wk / 1000 : undefined, 'ICSA', dateOf(icsa), { status: claims4wk < 250000 ? 'healthy' : claims4wk < 350000 ? 'elevated' : 'weak' }),
            creditSpread: F(bbbCurrent, 'BAMLC0A4CBBB', dateOf(bbb), { status: bbbCurrent < 1.5 ? 'tight' : bbbCurrent < 2.5 ? 'normal' : 'stressed' }),
            realYields: F(tipsCurrent, 'DFII10', dateOf(dfii10), { change: tipsCurrent - tipsPrev, status: tipsCurrent > 2.0 ? 'restrictive' : tipsCurrent > 0 ? 'neutral' : 'easy' }),
            lei: F(leiChange, 'USSLIND', dateOf(usslind), { status: leiCurrent > 0 ? 'rising' : 'falling' }),
        },
        checklist: {
            nfci: F(nfciCurrent, 'NFCI', dateOf(nfci), { bullish: nfciCurrent < 0, status: nfciCurrent < -0.5 ? 'strong' : nfciCurrent < 0 ? 'good' : 'weak', label: 'Financial Conditions' }),
            m2: F(m2Growth, 'M2SL', dateOf(m2sl), { bullish: m2Growth > 2.0, status: m2Growth > 4.0 ? 'strong' : m2Growth > 2.0 ? 'good' : 'weak', label: 'M2 Money Supply' }),
            retail: F(retailGrowth, 'RSXFS', dateOf(rsxfs), { bullish: retailGrowth > 0, status: retailGrowth > 1.0 ? 'strong' : retailGrowth > 0 ? 'good' : 'weak', label: 'Retail Sales (3mo)' }),
            housing: F(housingCurrent, 'HOUST', dateOf(houst), { bullish: housingCurrent > housing6moAvg && housingCurrent > 1300, status: housingCurrent > 1400 ? 'strong' : (housingCurrent > housing6moAvg && housingCurrent > 1300) ? 'good' : 'weak', label: 'Housing Starts' }),
            indpro: F(indproChange, 'INDPRO', dateOf(indpro), { bullish: indproChange > 0, status: indproChange > 1.0 ? 'strong' : indproChange > 0 ? 'good' : 'weak', label: 'Industrial Production' }),
            jolts: F(joltsCurrent, 'JTSJOL', dateOf(jtsjol), { bullish: joltsCurrent > 6000, status: joltsCurrent > 7000 ? 'strong' : joltsCurrent > 6000 ? 'good' : 'weak', label: 'Job Openings (JOLTS)' }),
            durable: F(durableChange, 'DGORDER', dateOf(dgorder), { bullish: durableChange > 0, status: durableChange > 2.0 ? 'strong' : durableChange > 0 ? 'good' : 'weak', label: 'Durable Goods Orders' }),
            savings: F(savingsCurrent, 'PSAVERT', dateOf(psavert), { bullish: savingsCurrent >= 3.5, status: savingsCurrent >= 5.0 ? 'strong' : savingsCurrent >= 3.5 ? 'good' : 'weak', label: 'Savings Rate' }),
        },
    };
}

// ─────────────────────────────────────────────────────────────────────────────

export async function GET(request) {
    // Touch the request so Next renders this handler dynamically (per request),
    // while individual FRED fetches still come from the 30-min Data Cache.
    request.headers.get('user-agent');

    const apiKey = process.env.FRED_API_KEY;
    if (!apiKey) return Response.json({ error: 'FRED_API_KEY not configured' }, { status: 500 });

    const now = new Date();

    const REQUESTS = [
        [FRED_SERIES.YIELD_CURVE, 100000],
        [FRED_SERIES.UNEMPLOYMENT, 15],
        [FRED_SERIES.SENTIMENT, 5],
        [FRED_SERIES.CLAIMS, 10],
        [FRED_SERIES.CREDIT_SPREAD, 252],
        [FRED_SERIES.REAL_YIELDS, 5],
        [FRED_SERIES.LEI, 5],
        [FRED_SERIES.NFCI, 5],
        [FRED_SERIES.M2_MONEY, 15],
        [FRED_SERIES.RETAIL_SALES, 5],
        [FRED_SERIES.HOUSING_STARTS, 10],
        [FRED_SERIES.INDUSTRIAL_PROD, 10],
        [FRED_SERIES.JOLTS, 5],
        [FRED_SERIES.DURABLE_GOODS, 5],
        [FRED_SERIES.SAVINGS_RATE, 5],
        [FRED_SERIES.CORP_PROFITS, 100000],
        [FRED_SERIES.GDP, 100000],
        [FRED_SERIES.RECESSIONS, 100000],
    ];

    // Fetch in small batches with a short stagger to be polite to FRED on a cold
    // cache. Cache hits resolve instantly, so the stagger is kept light (it adds
    // latency on every load regardless of cache state).
    const BATCH_SIZE = 4;
    const STAGGER_MS = 150;
    const settled = [];
    for (let i = 0; i < REQUESTS.length; i += BATCH_SIZE) {
        const batch = REQUESTS.slice(i, i + BATCH_SIZE).map(([id, limit]) =>
            fetchSeries(id, liveKey, limit)
                .then(value => ({ id, status: 'fulfilled', value }))
                .catch(e => ({ id, status: 'rejected', reason: e })),
        );
        settled.push(...(await Promise.all(batch)));
        if (i + BATCH_SIZE < REQUESTS.length) await new Promise(r => setTimeout(r, STAGGER_MS));
    }

    const series = {};
    const failed = [];
    for (const s of settled) {
        series[s.id] = s.status === 'fulfilled' && s.value?.length ? s.value : [];
        if (s.status === 'rejected' || !s.value?.length) failed.push(s.id);
    }

    const messages = [`Loaded ${REQUESTS.length - failed.length}/${REQUESTS.length} series`];
    for (const s of settled) {
        if (s.status === 'rejected') messages.push(`Series failed: ${maskKey(s.reason?.message || s.id)}`);
    }
    if (failed.length) console.warn(`[FRED] ${failed.length} series unavailable this load:`, failed.join(', '));

    // P/E ratio — layered, cached scrapes.
    let peRatio = null;
    try {
        const peHtml = await cachedText('multpl', EXTERNAL_URLS.MULTPL_PE, 8000);
        const m = peHtml.match(/Current S&P 500 PE Ratio[^\d]*(\d+\.\d+)/);
        if (m) peRatio = parseFloat(m[1]);
    } catch (e) { messages.push(`P/E multpl failed: ${maskKey(e.message)}`); }
    if (!peRatio) {
        try {
            const yHtml = await cachedText('yahoo-pe', EXTERNAL_URLS.YAHOO_PE, 8000);
            const m = yHtml.match(/PE Ratio \(TTM\)[\s\S]*?(\d+\.\d+)/i);
            if (m) peRatio = parseFloat(m[1]) * 1.07;
        } catch (e) { messages.push(`P/E Yahoo failed: ${maskKey(e.message)}`); }
    }
    if (!peRatio) {
        try {
            const cape = await fetchSeries('PE10', liveKey, 3);
            if (cape.length > 0) peRatio = cape[0].value;
        } catch (e) { messages.push(`P/E CAPE failed: ${maskKey(e.message)}`); }
    }

    const responseData = buildResponse(series, peRatio, now);
    const payload = {
        ...responseData,
        _meta: { source: 'St. Louis Fed', hasErrors: failed.length > 0, fetchedAt: now.toISOString(), messages },
    };

    // Never-throws: save as last-known-good if we got real data; if a total
    // outage left us with 0/18 series, serve the last-known-good instead of N/A.
    return serve('fred', async () => payload, {
        isGood: () => failed.length < REQUESTS.length,
        fallback: { error: 'FRED temporarily unavailable' },
        faults,
    });
}
