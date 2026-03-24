import { FRED_SERIES, EXTERNAL_URLS } from '../../../lib/constants';
import { fetchJson, fetchText } from '../../../lib/fetcher';

export const dynamic = 'force-dynamic';

// ---- FRED ----
async function fetchFredSeries(seriesId, apiKey, limit = 35) {
    const url = `${EXTERNAL_URLS.FRED_BASE}?series_id=${seriesId}&api_key=${apiKey}&file_type=json&sort_order=desc&limit=${limit}`;
    const data = await fetchJson(url, { revalidate: 0 });
    return data.observations
        .filter(o => o.value !== '.')
        .map(o => ({ date: o.date, value: parseFloat(o.value) }));
}

// ---- Stooq (sequential fetch with delay to avoid 429) ----
async function fetchStooq(symbol) {
    try {
        const url = `https://stooq.com/q/d/l/?s=${symbol}&i=d`;
        const text = await fetchText(url, { revalidate: 0 });
        const lines = text.trim().split('\n').slice(1);
        if (!lines.length || lines[0].startsWith('No data')) return null;
        const rows = lines.slice(-30).map(line => {
            const [date, , , , close] = line.split(',');
            return { date, price: parseFloat(close) };
        }).filter(r => r.date && !isNaN(r.price));
        if (rows.length < 2) return null;
        const current = rows[rows.length - 1].price;
        const prev = rows[rows.length - 2].price;
        return {
            current,
            dailyChange: { value: current - prev, pct: prev ? ((current - prev) / prev) * 100 : 0 },
            history: rows
        };
    } catch (e) {
        console.warn(`[Stooq] ${symbol}: ${e.message}`);
        return null;
    }
}

async function fetchStooqSequential(symbols) {
    const results = [];
    for (const sym of symbols) {
        results.push(await fetchStooq(sym));
        await new Promise(r => setTimeout(r, 150));
    }
    return results;
}

// ---- ExchangeRate API — free, no key, live spot rates ----
async function fetchExchangeRates() {
    try {
        const data = await fetchJson('https://open.er-api.com/v6/latest/USD', { revalidate: 0 });
        if (data.result !== 'success') return null;
        return data.rates;
    } catch (e) {
        console.warn('[ExchangeRate] fetch failed:', e.message);
        return null;
    }
}

// ---- Compute DXY from official ICE basket formula ----
// Rates from ExchangeRate-API are expressed as "units of foreign currency per 1 USD"
// DXY = 50.14348112 × EURUSD^(-0.576) × USDJPY^(0.136) × GBPUSD^(-0.119) × USDCAD^(0.091) × USDSEK^(0.042) × USDCHF^(0.036)
// With ER-API convention: EURUSD = 1/EUR_rate, GBPUSD = 1/GBP_rate
// → EUR_rate^(0.576) × JPY_rate^(0.136) × GBP_rate^(0.119) × CAD_rate^(0.091) × SEK_rate^(0.042) × CHF_rate^(0.036)
function computeDXY(rates) {
    if (!rates) return null;
    const EUR = rates['EUR'], JPY = rates['JPY'], GBP = rates['GBP'];
    const CAD = rates['CAD'], SEK = rates['SEK'], CHF = rates['CHF'];
    if (!EUR || !JPY || !GBP || !CAD || !SEK || !CHF) return null;
    return Math.round(
        50.14348112 *
        Math.pow(EUR, 0.576) *
        Math.pow(JPY, 0.136) *
        Math.pow(GBP, 0.119) *
        Math.pow(CAD, 0.091) *
        Math.pow(SEK, 0.042) *
        Math.pow(CHF, 0.036)
        * 100) / 100;
}

// ---- Standardize FRED series → oldest-first history ----
function standardizeFred(data, multiplier = 1) {
    if (!data || data.length < 2) return null;
    const history = [...data].reverse().map(d => ({ date: d.date, price: d.value * multiplier }));
    const current = history[history.length - 1].price;
    const prev = history[history.length - 2].price;
    return { current, dailyChange: { value: current - prev, pct: prev ? ((current - prev) / prev) * 100 : 0 }, history };
}

// ---- Build a cross-rate series from two USD-quoted FRED arrays ----
// e.g. INR/BDT = BDT_per_USD / INR_per_USD = BDT/INR
function crossRateFred(usdBaseArr, usdTargetArr) {
    if (!usdBaseArr?.length || !usdTargetArr?.length) return null;
    const bMap = new Map(usdBaseArr.map(x => [x.date, x.value]));
    const history = [];
    for (const t of usdTargetArr) {
        const b = bMap.get(t.date);
        if (b && b !== 0) history.push({ date: t.date, price: t.value / b });
    }
    if (history.length < 2) return null;
    history.reverse();
    const current = history[history.length - 1].price;
    const prev = history[history.length - 2].price;
    return { current, dailyChange: { value: current - prev, pct: prev ? ((current - prev) / prev) * 100 : 0 }, history };
}

// ---- Spot-only (no sparkline) for live rate with no history ----
function spotOnly(value) {
    if (value == null) return null;
    return { current: value, dailyChange: { value: 0, pct: 0 }, history: [] };
}

export async function GET() {
    const apiKey = process.env.FRED_API_KEY;
    if (!apiKey) return Response.json({ error: 'FRED_API_KEY not configured' }, { status: 500 });

    try {
        const fredRequests = [
            [FRED_SERIES.MORTGAGE30, 30],         // 0 30Y rate
            [FRED_SERIES.RENT_INDEX, 30],          // 1 rent CPI
            [FRED_SERIES.MEDIAN_HOME_PRICE, 5],    // 2 home price
            ['DCOILWTICO', 35],                    // 3 WTI oil
            ['DGS10', 35],                         // 4 10Y treasury
            ['DGS2', 35],                          // 5 2Y treasury
            ['DEXCAUS', 35],                       // 6 USD/CAD
            ['DEXINUS', 35]                        // 7 USD/INR
        ];

        const fredResultsRaw = [];
        for (let i = 0; i < fredRequests.length; i += 3) {
            const batch = fredRequests.slice(i, i + 3).map(([id, limit]) =>
                fetchFredSeries(id, apiKey, limit)
                    .then(v => ({ status: 'fulfilled', value: v }))
                    .catch(e => ({ status: 'rejected', reason: e }))
            );
            fredResultsRaw.push(...(await Promise.all(batch)));
            if (i + 3 < fredRequests.length) {
                await new Promise(r => setTimeout(r, 300));
            }
        }

        const [fredResults, erRates] = await Promise.all([
            Promise.resolve(fredResultsRaw),
            fetchExchangeRates()
        ]);

        const [mortgageRateData, rentData, homeData, oilData, tnxData, t2yData, cadData, inrData] =
            fredResults.map(r => r.status === 'fulfilled' ? r.value : []);

        // --- 2. Fetch Stooq sequentially (BTC + Gold only — minimize rate-limit risk) ---
        const [btc, gold] = await fetchStooqSequential(['btc.v', 'xauusd']);

        // --- 3. Compute from live ER-API ---
        const bdtRate = erRates?.BDT;
        const inrRate = erRates?.INR;
        const dxyValue = computeDXY(erRates);

        // USD/BDT from ER-API (live, ~122.5 BDT per USD)
        const usdbdt = bdtRate ? spotOnly(bdtRate) : null;

        // INR/BDT = BDT_per_USD / INR_per_USD (live spot)
        const inrbdt = (bdtRate && inrRate) ? spotOnly(bdtRate / inrRate) : null;

        // DXY computed via ICE formula from live rates
        const dxy = dxyValue ? spotOnly(dxyValue) : null;

        // --- 4. Standardize FRED series ---
        const usdcad = standardizeFred(cadData);
        const usdinr = standardizeFred(inrData);
        const cadinr = crossRateFred(cadData, inrData); // INR/CAD = DEXINUS / DEXCAUS
        const cl = standardizeFred(oilData);
        const tnx = standardizeFred(tnxData);
        const t2y = standardizeFred(t2yData);
        const mortStd = standardizeFred(mortgageRateData);
        const rentStd = standardizeFred(rentData, 4.41);

        // --- 5. Compute Mortgage Payment ---
        let mortPayment = null;
        if (homeData?.length > 0 && mortgageRateData?.length > 0) {
            const principal = homeData[0].value * 0.80;
            const history = [...mortgageRateData].reverse().map(m => {
                const r = (m.value / 100) / 12;
                const pmt = (principal * r * Math.pow(1 + r, 360)) / (Math.pow(1 + r, 360) - 1);
                return { date: m.date, price: pmt };
            });
            const current = history[history.length - 1].price;
            const prev = history[history.length - 2].price;
            mortPayment = { current, dailyChange: { value: current - prev, pct: prev ? ((current - prev) / prev) * 100 : 0 }, history };
        }

        return Response.json({
            fx: { usdcad, usdinr, usdbdt, inrbdt, cadinr, dxy },
            commodities: { cl, gc: gold, btc },
            rates: { tnx, t2y, mortgageRate: mortStd },
            realEstate: { rentIndex: rentStd, mortgagePayment: mortPayment },
            _meta: {
                source: 'Stooq, ExchangeRate-API & FRED',
                hasErrors: !btc || !usdcad,
                messages: ['Stooq: BTC, Gold | ER-API: BDT, DXY | FRED: FX, Oil, Rates, Real Estate']
            }
        });
    } catch (error) {
        return Response.json({ error: error.message }, { status: 500 });
    }
}
