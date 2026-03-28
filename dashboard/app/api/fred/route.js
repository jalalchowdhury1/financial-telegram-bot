import fs from 'fs';
import { FRED_SERIES, EXTERNAL_URLS } from '../../../lib/constants';
import { fetchJson, proxyFetch } from '../../../lib/fetcher';

export const dynamic = 'force-dynamic';

const CACHE_FILE = '/tmp/fred-data-cache.json';

async function fetchSeries(seriesId, apiKey, limit = 15) {
    const url = `${EXTERNAL_URLS.FRED_BASE}?series_id=${seriesId}&api_key=${apiKey}&file_type=json&sort_order=desc&limit=${limit}`;
    const data = await fetchJson(url, { revalidate: 0 });
    return data.observations
        .filter(o => o.value !== '.')
        .map(o => ({ date: o.date, value: parseFloat(o.value) }));
}

function findByMonthOffset(arr, nMonths) {
    if (!arr?.length) return undefined;
    const target = new Date(arr[0].date);
    target.setMonth(target.getMonth() - nMonths);
    return arr.reduce((best, obs) => {
        const d = Math.abs(new Date(obs.date) - target);
        return d < Math.abs(new Date(best.date) - target) ? obs : best;
    });
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

function buildResponse(t10y2y, unrate, umcsent, icsa, bbb, dfii10, usslind, nfci, m2sl, rsxfs, houst, indpro, jtsjol, dgorder, psavert, corpProfits, gdpData, usrec, peRatio) {
    // Recession periods
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

    const yieldCurve = { current: t10y2y[0]?.value, date: t10y2y[0]?.date, history: [...t10y2y].reverse() };
    const unrate3mo = unrate.length >= 3 ? unrate.slice(0, 3).reduce((s, v) => s + v.value, 0) / 3 : null;
    const unrate12moLow = unrate.length > 0 ? Math.min(...unrate.map(u => u.value)) : null;
    const sahmRule = unrate3mo !== null && unrate12moLow !== null ? unrate3mo - unrate12moLow : 0;
    const sentimentCurrent = umcsent[0]?.value;
    const sentimentPrev = umcsent[1]?.value;
    const claims4wk = icsa.length >= 4 ? icsa.slice(0, 4).reduce((s, v) => s + v.value, 0) / 4 : 0;
    const bbbCurrent = bbb[0]?.value;
    const tipsCurrent = dfii10[0]?.value;
    const tipsPrev = dfii10[1]?.value;
    const leiCurrent = usslind[0]?.value;
    const leiPrev = usslind[1]?.value;
    const leiChange = leiPrev ? ((leiCurrent - leiPrev) / leiPrev) * 100 : 0;
    const nfciCurrent = nfci[0]?.value;
    const m2Current = m2sl[0]?.value;
    const m2YearAgo = findByMonthOffset(m2sl, 12)?.value;
    const m2Growth = m2YearAgo ? ((m2Current - m2YearAgo) / m2YearAgo) * 100 : 0;
    const retailCurrent = rsxfs[0]?.value;
    const retail3mo = findByMonthOffset(rsxfs, 3)?.value;
    const retailGrowth = retail3mo ? ((retailCurrent - retail3mo) / retail3mo) * 100 : 0;
    const housingCurrent = houst[0]?.value;
    const housing6moAvg = houst.length >= 6 ? houst.slice(0, 6).reduce((s, v) => s + v.value, 0) / 6 : 0;
    const indproCurrent = indpro[0]?.value;
    const indpro6mo = findByMonthOffset(indpro, 6)?.value;
    const indproChange = indpro6mo ? ((indproCurrent - indpro6mo) / indpro6mo) * 100 : 0;
    const joltsCurrent = jtsjol[0]?.value;
    const durableCurrent = dgorder[0]?.value;
    const durable3mo = findByMonthOffset(dgorder, 3)?.value;
    const durableChange = durable3mo ? ((durableCurrent - durable3mo) / durable3mo) * 100 : 0;
    const savingsCurrent = psavert[0]?.value;

    const gdpMap = new Map();
    for (const gd of gdpData) gdpMap.set(gd.date, gd.value);
    const profitMarginHistory = [];
    for (const cp of corpProfits) {
        const gdpValue = gdpMap.get(cp.date);
        if (gdpValue && gdpValue !== 0) profitMarginHistory.push({ date: cp.date, value: (cp.value / gdpValue) * 100 });
    }
    const profitMargin = { current: profitMarginHistory[0]?.value || 0, date: profitMarginHistory[0]?.date || '', history: [...profitMarginHistory].reverse() };

    return {
        yieldCurve, profitMargin, peRatio, recessions: recessionPeriods,
        indicators: {
            sahmRule: { value: sahmRule, status: sahmRule >= 0.5 ? 'danger' : 'safe' },
            sentiment: { value: sentimentCurrent, change: sentimentCurrent - sentimentPrev, status: sentimentCurrent > 80 ? 'strong' : sentimentCurrent > 60 ? 'neutral' : 'weak' },
            claims: { value: claims4wk / 1000, status: claims4wk < 250000 ? 'healthy' : claims4wk < 350000 ? 'elevated' : 'weak' },
            creditSpread: { value: bbbCurrent, status: bbbCurrent < 1.5 ? 'tight' : bbbCurrent < 2.5 ? 'normal' : 'stressed' },
            realYields: { value: tipsCurrent, change: tipsCurrent - tipsPrev, status: tipsCurrent > 2.0 ? 'restrictive' : tipsCurrent > 0 ? 'neutral' : 'easy' },
            lei: { value: leiCurrent, change: leiChange, status: leiCurrent > 0 ? 'rising' : 'falling' }
        },
        checklist: {
            nfci: { value: nfciCurrent, bullish: nfciCurrent < 0, status: nfciCurrent < -0.5 ? 'strong' : nfciCurrent < 0 ? 'good' : 'weak', label: 'Financial Conditions' },
            m2: { value: m2Growth, bullish: m2Growth > 2.0, status: m2Growth > 4.0 ? 'strong' : m2Growth > 2.0 ? 'good' : 'weak', label: 'M2 Money Supply' },
            retail: { value: retailGrowth, bullish: retailGrowth > 0, status: retailGrowth > 1.0 ? 'strong' : retailGrowth > 0 ? 'good' : 'weak', label: 'Retail Sales (3mo)' },
            housing: { value: housingCurrent, bullish: housingCurrent > housing6moAvg && housingCurrent > 1300, status: housingCurrent > 1400 ? 'strong' : (housingCurrent > housing6moAvg && housingCurrent > 1300) ? 'good' : 'weak', label: 'Housing Starts' },
            indpro: { value: indproChange, bullish: indproChange > 0, status: indproChange > 1.0 ? 'strong' : indproChange > 0 ? 'good' : 'weak', label: 'Industrial Production' },
            jolts: { value: joltsCurrent, bullish: joltsCurrent > 6000, status: joltsCurrent > 7000 ? 'strong' : joltsCurrent > 6000 ? 'good' : 'weak', label: 'Job Openings (JOLTS)' },
            durable: { value: durableChange, bullish: durableChange > 0, status: durableChange > 2.0 ? 'strong' : durableChange > 0 ? 'good' : 'weak', label: 'Durable Goods Orders' },
            savings: { value: savingsCurrent, bullish: savingsCurrent >= 3.5, status: savingsCurrent >= 5.0 ? 'strong' : savingsCurrent >= 3.5 ? 'good' : 'weak', label: 'Savings Rate' }
        }
    };
}

export async function GET() {
    const apiKey = process.env.FRED_API_KEY;
    if (!apiKey) return Response.json({ error: 'FRED_API_KEY not configured' }, { status: 500 });

    // ── Layer 1: FRED API (all 18 series, batched) ──
    try {
        const fredRequests = [
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
            [FRED_SERIES.RECESSIONS, 100000]
        ];

        const results = [];
        for (let i = 0; i < fredRequests.length; i += 3) {
            const batch = fredRequests.slice(i, i + 3).map(([id, limit]) =>
                fetchSeries(id, apiKey, limit)
                    .then(v => ({ status: 'fulfilled', value: v }))
                    .catch(e => ({ status: 'rejected', reason: e }))
            );
            results.push(...(await Promise.all(batch)));
            if (i + 3 < fredRequests.length) await new Promise(r => setTimeout(r, 300));
        }

        const safeValue = (res) => res.status === 'fulfilled' ? res.value : [];
        const [t10y2y, unrate, umcsent, icsa, bbb, dfii10, usslind, nfci, m2sl, rsxfs, houst, indpro, jtsjol, dgorder, psavert, corpProfits, gdpData, usrec] = results.map(safeValue);

        // P/E Ratio — 3 sub-layers
        let peRatio = null;

        // P/E Layer 1: multpl.com
        try {
            const peRes = await proxyFetch(EXTERNAL_URLS.MULTPL_PE, { revalidate: 0 });
            const peHtml = await peRes.text();
            const peMatch = peHtml.match(/Current S&P 500 PE Ratio[^\d]*(\d+\.\d+)/);
            if (peMatch) peRatio = parseFloat(peMatch[1]);
        } catch (e) { console.warn('P/E Layer 1 (multpl.com) failed:', e.message); }

        // P/E Layer 2: Yahoo Finance key-statistics scrape
        if (!peRatio) {
            try {
                const yRes = await proxyFetch(EXTERNAL_URLS.YAHOO_PE, { revalidate: 0 });
                const yHtml = await yRes.text();
                const peMatch = yHtml.match(/PE Ratio \(TTM\)[\s\S]*?(\d+\.\d+)/i);
                if (peMatch) peRatio = parseFloat(peMatch[1]) * 1.07; // GAAP adjustment
            } catch (e) { console.warn('P/E Layer 2 (Yahoo) failed:', e.message); }
        }

        // P/E Layer 3: FRED PE10 (Shiller CAPE) — uses existing API key
        if (!peRatio) {
            try {
                const capeData = await fetchSeries('PE10', apiKey, 3);
                if (capeData.length > 0) peRatio = capeData[0].value; // CAPE is higher than trailing, mark as-is
            } catch (e) { console.warn('P/E Layer 3 (FRED CAPE) failed:', e.message); }
        }

        // P/E Layer 4: Derive from Corp Profits / GDP (approximate)
        if (!peRatio && corpProfits.length > 0 && gdpData.length > 0) {
            try {
                const latestCP = corpProfits[0].value;
                const latestGDP = gdpData.find(g => g.date === corpProfits[0].date)?.value || gdpData[0].value;
                // Corp profits as % of GDP, historically PE ≈ 1/(profit margin as % of GDP) * GDP/market multiplier
                // Rough approximation: trailing PE ≈ 16 * (10% / profit_pct_GDP)
                const profitPct = (latestCP / latestGDP) * 100;
                if (profitPct > 0) peRatio = Math.round((10 / profitPct) * 18 * 10) / 10;
            } catch (e) { console.warn('P/E Layer 4 (derived) failed:', e.message); }
        }

        // P/E Layer 5: stale from cache
        if (!peRatio) {
            const cached = loadCache();
            if (cached?.peRatio) { peRatio = cached.peRatio; console.warn('P/E Layer 5: using cached P/E'); }
        }

        const responseData = buildResponse(t10y2y, unrate, umcsent, icsa, bbb, dfii10, usslind, nfci, m2sl, rsxfs, houst, indpro, jtsjol, dgorder, psavert, corpProfits, gdpData, usrec, peRatio);
        const hasErrors = results.some(r => r.status === 'rejected');
        const fullResponse = {
            ...responseData,
            _meta: {
                source: 'St. Louis Fed',
                hasErrors,
                messages: [
                    `Loaded ${results.filter(r => r.status === 'fulfilled').length}/${fredRequests.length} series`,
                    ...results.filter(r => r.status === 'rejected').map(r => `Series failed: ${r.reason?.message}`)
                ]
            }
        };

        // Save successful response to cache (strip _meta to keep cache clean)
        if (!hasErrors) saveCache(fullResponse);

        return Response.json(fullResponse);

    } catch (outerError) {
        // ── Layer 2–5: Full stale cache (FRED API completely down) ──
        console.warn('[FRED] Primary fetch failed, trying stale cache:', outerError.message);
        const cached = loadCache();
        if (cached) {
            return Response.json({
                ...cached,
                _meta: {
                    source: 'Stale Cache (FRED unavailable)',
                    hasErrors: true,
                    messages: [`FRED API failed: ${outerError.message}`, `Serving cached data from ${cached._cachedAt || 'unknown'}`]
                }
            });
        }

        return Response.json({ error: outerError.message }, { status: 500 });
    }
}
