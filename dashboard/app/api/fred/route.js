// /api/fred - Fetch all FRED economic data
export const dynamic = 'force-dynamic';
const FRED_BASE = 'https://api.stlouisfed.org/fred/series/observations';

async function fetchSeries(seriesId, apiKey, limit = 15) {
    const url = `${FRED_BASE}?series_id=${seriesId}&api_key=${apiKey}&file_type=json&sort_order=desc&limit=${limit}`;
    const res = await fetch(url, { next: { revalidate: 0 } });
    if (!res.ok) throw new Error(`FRED ${seriesId}: ${res.status}`);
    const data = await res.json();
    return data.observations
        .filter(o => o.value !== '.')
        .map(o => ({ date: o.date, value: parseFloat(o.value) }));
}

export async function GET() {
    const apiKey = process.env.FRED_API_KEY;
    if (!apiKey) {
        return Response.json({ error: 'FRED_API_KEY not configured' }, { status: 500 });
    }

    try {
        const results = await Promise.allSettled([
            fetchSeries('T10Y2Y', apiKey, 100000),
            fetchSeries('UNRATE', apiKey, 15),
            fetchSeries('UMCSENT', apiKey, 5),
            fetchSeries('ICSA', apiKey, 10),
            fetchSeries('BAMLC0A4CBBB', apiKey, 252),
            fetchSeries('DFII10', apiKey, 5),
            fetchSeries('USSLIND', apiKey, 5),
            fetchSeries('NFCI', apiKey, 5),
            fetchSeries('M2SL', apiKey, 15),
            fetchSeries('RSXFS', apiKey, 5),
            fetchSeries('HOUST', apiKey, 10),
            fetchSeries('INDPRO', apiKey, 10),
            fetchSeries('JTSJOL', apiKey, 5),
            fetchSeries('DGORDER', apiKey, 5),
            fetchSeries('PSAVERT', apiKey, 5),
            fetchSeries('A053RC1Q027SBEA', apiKey, 100000),
            fetchSeries('GDP', apiKey, 100000),
            fetchSeries('USREC', apiKey, 100000)
        ]);

        const safeValue = (res) => res.status === 'fulfilled' ? res.value : [];

        const [
            t10y2y, unrate, umcsent, icsa, bbb, dfii10, usslind,
            nfci, m2sl, rsxfs, houst, indpro, jtsjol, dgorder, psavert,
            corpProfits, gdpData, usrec
        ] = results.map(safeValue);

        // Fetch current S&P 500 P/E ratio
        let peRatio = null;

        // Method 1: multpl.com (Current S&P 500)
        try {
            const peRes = await fetch('https://www.multpl.com/s-p-500-pe-ratio', {
                headers: { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36' },
                next: { revalidate: 3600 }
            });
            const peHtml = await peRes.text();
            const peMatch = peHtml.match(/Current S&P 500 PE Ratio[^\d]*(\d+\.\d+)/);
            if (peMatch) peRatio = parseFloat(peMatch[1]);
        } catch (e) {
            console.warn("multpl.com P/E fetch failed:", e.message);
        }

        // Method 2: Yahoo Finance Fallback (SPY Trailing P/E)
        if (!peRatio) {
            try {
                const yRes = await fetch('https://finance.yahoo.com/quote/SPY/key-statistics', {
                    headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' },
                    next: { revalidate: 3600 }
                });
                const yHtml = await yRes.text();
                const peMatch = yHtml.match(/PE Ratio \(TTM\)[\s\S]*?(\d+\.\d+)/i);
                if (peMatch) peRatio = parseFloat(peMatch[1]);
            } catch (e) {
                console.warn("Yahoo Finance P/E fallback failed:", e.message);
            }
        }

        // Compute recession periods from USREC (1=recession, 0=expansion)
        const recessionPeriods = [];
        const recSorted = [...usrec].reverse(); // oldest first
        let recStart = null;
        for (let i = 0; i < recSorted.length; i++) {
            if (recSorted[i].value === 1 && recStart === null) {
                recStart = recSorted[i].date;
            } else if (recSorted[i].value === 0 && recStart !== null) {
                recessionPeriods.push({ start: recStart, end: recSorted[i].date });
                recStart = null;
            }
        }
        if (recStart !== null) {
            recessionPeriods.push({ start: recStart, end: recSorted[recSorted.length - 1].date });
        }

        // Yield Curve
        const yieldCurve = {
            current: t10y2y[0]?.value,
            date: t10y2y[0]?.date,
            history: [...t10y2y].reverse()
        };

        // Sahm Rule
        const unrate3mo = unrate.length >= 3 ? unrate.slice(0, 3).reduce((s, v) => s + v.value, 0) / 3 : null;
        const unrate12moLow = unrate.length > 0 ? Math.min(...unrate.map(u => u.value)) : null;
        const sahmRule = unrate3mo !== null && unrate12moLow !== null ? unrate3mo - unrate12moLow : 0;

        // Consumer Sentiment
        const sentimentCurrent = umcsent[0]?.value;
        const sentimentPrev = umcsent[1]?.value;

        // Claims
        const claims4wk = icsa.length >= 4 ? icsa.slice(0, 4).reduce((s, v) => s + v.value, 0) / 4 : 0;

        // Credit Spread
        const bbbCurrent = bbb[0]?.value;

        // Real Yields
        const tipsCurrent = dfii10[0]?.value;
        const tipsPrev = dfii10[1]?.value;

        // LEI
        const leiCurrent = usslind[0]?.value;
        const leiPrev = usslind[1]?.value;
        const leiChange = ((leiCurrent - leiPrev) / leiPrev) * 100;

        // Bull Market Checklist
        const nfciCurrent = nfci[0]?.value;
        const m2Current = m2sl[0]?.value;
        const m2YearAgo = m2sl[12]?.value;
        const m2Growth = m2YearAgo ? ((m2Current - m2YearAgo) / m2YearAgo) * 100 : 0;
        const retailCurrent = rsxfs[0]?.value;
        const retail3mo = rsxfs[3]?.value;
        const retailGrowth = retail3mo ? ((retailCurrent - retail3mo) / retail3mo) * 100 : 0;
        const housingCurrent = houst[0]?.value;
        const housing6moAvg = houst.length >= 6 ? houst.slice(0, 6).reduce((s, v) => s + v.value, 0) / 6 : 0;
        const indproCurrent = indpro[0]?.value;
        const indpro6mo = indpro[6]?.value;
        const indproChange = indpro6mo ? ((indproCurrent - indpro6mo) / indpro6mo) * 100 : 0;
        const joltsCurrent = jtsjol[0]?.value;
        const durableCurrent = dgorder[0]?.value;
        const durable3mo = dgorder[3]?.value;
        const durableChange = durable3mo ? ((durableCurrent - durable3mo) / durable3mo) * 100 : 0;
        const savingsCurrent = psavert[0]?.value;

        // Profit Margin
        const gdpMap = new Map();
        for (const gd of gdpData) {
            gdpMap.set(gd.date, gd.value);
        }

        const profitMarginHistory = [];
        for (const cp of corpProfits) {
            const gdpValue = gdpMap.get(cp.date);
            if (gdpValue && gdpValue !== 0) {
                profitMarginHistory.push({ date: cp.date, value: (cp.value / gdpValue) * 100 });
            }
        }
        const profitMargin = {
            current: profitMarginHistory[0]?.value || 0,
            date: profitMarginHistory[0]?.date || '',
            history: [...profitMarginHistory].reverse()
        };

        return Response.json({
            yieldCurve,
            profitMargin,
            peRatio,
            recessions: recessionPeriods,
            indicators: {
                sahmRule: { value: sahmRule, status: sahmRule >= 0.5 ? 'danger' : 'safe' },
                sentiment: { value: sentimentCurrent, change: sentimentCurrent - sentimentPrev, status: sentimentCurrent > 80 ? 'strong' : sentimentCurrent > 60 ? 'neutral' : 'weak' },
                claims: { value: claims4wk / 1000, status: claims4wk < 250000 ? 'healthy' : claims4wk < 350000 ? 'elevated' : 'weak' },
                creditSpread: { value: bbbCurrent, status: bbbCurrent < 1.5 ? 'tight' : bbbCurrent < 2.5 ? 'normal' : 'stressed' },
                realYields: { value: tipsCurrent, change: tipsCurrent - tipsPrev, status: tipsCurrent > 2.0 ? 'restrictive' : tipsCurrent > 0 ? 'neutral' : 'easy' },
                lei: { value: leiCurrent, change: leiChange, status: leiChange > 0 ? 'rising' : 'falling' }
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
            },
            _meta: {
                source: 'St. Louis Fed',
                hasErrors: results.some(r => r.status === 'rejected'),
                messages: [
                    `Loaded ${results.filter(r => r.status === 'fulfilled').length}/${results.length} series`,
                    ...results.filter(r => r.status === 'rejected').map(r => `Failed to load a series: ${r.reason.message}`)
                ]
            }
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
