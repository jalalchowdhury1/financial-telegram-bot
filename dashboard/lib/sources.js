/**
 * Direct market-data sources used as FALLBACKS beneath the primary feeds
 * (AWS Lambda / FRED). Each returns normalized data and is independently
 * cacheable. These exist so no dashboard number ever goes blank.
 *
 * Conventions:
 *   - history is oldest -> newest: [{ date: 'YYYY-MM-DD', price: Number }]
 *   - revalidate is the Data Cache TTL in seconds (routes set fetchCache so
 *     these can cache even though the handler is dynamic).
 */

import { fetchJson, fetchText, proxyFetch } from './fetcher';

const day = (epochSec) => new Date(epochSec * 1000).toISOString().slice(0, 10);

/**
 * Yahoo Finance chart endpoint -> { current, prevClose, history, meta }.
 * Works for equities (SPY), FX (CAD=X), futures (GC=F, CL=F), crypto (BTC-USD),
 * indices (DX-Y.NYB, ^TNX). Uses adjusted close when available.
 */
export async function yahooChart(ticker, { range = '1mo', interval = '1d', revalidate = 300 } = {}) {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?range=${range}&interval=${interval}`;
    const data = await fetchJson(url, { revalidate });
    const r = data?.chart?.result?.[0];
    if (!r) throw new Error(`Yahoo: no result for ${ticker}`);
    const ts = r.timestamp || [];
    const closes = r.indicators?.quote?.[0]?.close || [];
    const adj = r.indicators?.adjclose?.[0]?.adjclose;
    const history = [];
    for (let i = 0; i < ts.length; i++) {
        const px = adj && adj[i] != null ? adj[i] : closes[i];
        if (px != null && !Number.isNaN(px)) history.push({ date: day(ts[i]), price: px });
    }
    if (!history.length) throw new Error(`Yahoo: empty history for ${ticker}`);
    const meta = r.meta || {};
    const current = meta.regularMarketPrice ?? history[history.length - 1].price;
    const prevClose = meta.chartPreviousClose ?? meta.previousClose ?? history[history.length - 2]?.price ?? current;
    return { current, prevClose, history, meta };
}

/**
 * Stooq daily CSV -> { current, prevClose, history }.
 * symbol examples: 'spy.us', 'usdcad', 'xauusd', 'cl.f'
 */
export async function stooqDaily(symbol, { revalidate = 300 } = {}) {
    const url = `https://stooq.com/q/d/l/?s=${encodeURIComponent(symbol)}&i=d`;
    const text = await fetchText(url, { revalidate });
    const lines = text.trim().split('\n');
    if (lines.length < 2 || !/^Date,/i.test(lines[0])) throw new Error(`Stooq: bad CSV for ${symbol}`);
    const history = [];
    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',');
        const date = cols[0];
        const close = parseFloat(cols[4]);
        if (date && !Number.isNaN(close)) history.push({ date, price: close });
    }
    if (!history.length) throw new Error(`Stooq: empty for ${symbol}`);
    return {
        current: history[history.length - 1].price,
        prevClose: history[history.length - 2]?.price ?? history[history.length - 1].price,
        history,
    };
}

/** CoinGecko simple price (keyless) -> { current, prevClose } for a coin id. */
export async function coingeckoPrice(id = 'bitcoin', { revalidate = 300 } = {}) {
    const url = `https://api.coingecko.com/api/v3/simple/price?ids=${id}&vs_currencies=usd&include_24hr_change=true`;
    const data = await fetchJson(url, { revalidate });
    const px = data?.[id]?.usd;
    if (px == null) throw new Error('CoinGecko: no price');
    const pct = data[id].usd_24h_change ?? 0;
    const prevClose = pct ? px / (1 + pct / 100) : px;
    return { current: px, prevClose };
}

/** Coinbase spot (keyless) -> { current } for a pair like 'BTC-USD'. */
export async function coinbaseSpot(pair = 'BTC-USD', { revalidate = 300 } = {}) {
    const data = await fetchJson(`https://api.coinbase.com/v2/prices/${pair}/spot`, { revalidate });
    const px = parseFloat(data?.data?.amount);
    if (Number.isNaN(px)) throw new Error('Coinbase: no amount');
    return { current: px };
}

/** ExchangeRate-API open endpoint (keyless) -> rates map for a base currency. */
export async function erApiRates(base = 'USD', { revalidate = 300 } = {}) {
    const data = await fetchJson(`https://open.er-api.com/v6/latest/${base}`, { revalidate });
    if (data?.result !== 'success' || !data.rates) throw new Error('ER-API: bad response');
    return data.rates; // { CAD: 1.37, INR: 94.5, BDT: 122.7, ... }
}

/** Frankfurter (ECB, keyless) -> rates map for a base currency. */
export async function frankfurterRates(base = 'USD', symbols = '', { revalidate = 300 } = {}) {
    const q = symbols ? `&symbols=${symbols}` : '';
    const data = await fetchJson(`https://api.frankfurter.app/latest?from=${base}${q}`, { revalidate });
    if (!data?.rates) throw new Error('Frankfurter: bad response');
    return data.rates;
}

/**
 * DBnomics mirror of a FRED series (keyless) -> [{date, value}] newest-first.
 * Numbers are identical to FRED (DBnomics ingests the same St. Louis Fed data).
 */
export async function dbnomicsFred(seriesId, { revalidate = 1800 } = {}) {
    const url = `https://api.db.nomics.world/v22/series?provider_code=FRED&series_code=${seriesId}&observations=1`;
    const data = await fetchJson(url, { revalidate });
    const doc = data?.series?.docs?.[0];
    if (!doc?.period || !doc?.value) throw new Error(`DBnomics: no series ${seriesId}`);
    const out = [];
    for (let i = 0; i < doc.period.length; i++) {
        const v = doc.value[i];
        if (v != null && v !== 'NA' && !Number.isNaN(Number(v))) {
            out.push({ date: doc.period[i], value: Number(v) });
        }
    }
    out.reverse(); // DBnomics is oldest-first; match FRED's newest-first
    if (!out.length) throw new Error(`DBnomics: empty ${seriesId}`);
    return out;
}

/** Build a {value,pct} daily-change object from current + previous close. */
export function dailyChange(current, prevClose) {
    const value = current - prevClose;
    const pct = prevClose ? (value / prevClose) * 100 : 0;
    return { value, pct };
}
