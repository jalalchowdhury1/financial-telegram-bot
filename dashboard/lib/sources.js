/**
 * Direct market-data sources used as FALLBACKS beneath the primary feeds
 * (AWS Lambda / FRED). Each returns normalized data and is independently
 * cacheable. These exist so no dashboard number ever goes blank.
 *
 * Conventions:
 *   - history is oldest -> newest: [{ date: 'YYYY-MM-DD', price: Number }]
 *   - revalidate is the Data Cache TTL in seconds.
 *   - Network calls retry with backoff on transient errors (429/5xx/timeouts),
 *     since free sources (esp. Yahoo) intermittently rate-limit cloud IPs.
 */

import { fetchJson, fetchText } from './fetcher';

const day = (epochSec) => new Date(epochSec * 1000).toISOString().slice(0, 10);

async function withRetry(fn, { tries = 3, base = 400 } = {}) {
    let last;
    for (let i = 0; i < tries; i++) {
        try { return await fn(); }
        catch (e) {
            last = e;
            const transient = /\b(429|50\d)\b|timed out|ECONNRESET|fetch failed|network/i.test(e?.message || '');
            if (!transient || i === tries - 1) break;
            await new Promise((r) => setTimeout(r, base * 2 ** i));
        }
    }
    throw last;
}

/**
 * Yahoo Finance chart -> { current, prevClose, history, meta }.
 * Works for equities (SPY), FX (CAD=X), futures (GC=F, CL=F), crypto (BTC-USD),
 * indices (DX-Y.NYB, ^TNX). Tries query1 then query2; uses adjusted close.
 */
export async function yahooChart(ticker, { range = '1mo', interval = '1d', revalidate = 300 } = {}) {
    const path = `/v8/finance/chart/${encodeURIComponent(ticker)}?range=${range}&interval=${interval}`;
    const data = await withRetry(async () => {
        try { return await fetchJson(`https://query1.finance.yahoo.com${path}`, { revalidate }); }
        catch { return await fetchJson(`https://query2.finance.yahoo.com${path}`, { revalidate }); }
    });
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

/** Stooq daily CSV -> { current, prevClose, history }. (Note: Stooq now gates
 *  bulk CSV behind an API key for many IPs; kept as a last-resort only.) */
export async function stooqDaily(symbol, { revalidate = 300 } = {}) {
    const text = await withRetry(() => fetchText(`https://stooq.com/q/d/l/?s=${encodeURIComponent(symbol)}&i=d`, { revalidate }));
    const lines = text.trim().split('\n');
    if (lines.length < 2 || !/^Date,/i.test(lines[0])) throw new Error(`Stooq: no CSV for ${symbol}`);
    const history = [];
    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',');
        const close = parseFloat(cols[4]);
        if (cols[0] && !Number.isNaN(close)) history.push({ date: cols[0], price: close });
    }
    if (!history.length) throw new Error(`Stooq: empty for ${symbol}`);
    return { current: history[history.length - 1].price, prevClose: history[history.length - 2]?.price ?? history[history.length - 1].price, history };
}

/** CoinGecko simple price (keyless) -> { current, prevClose }. */
export async function coingeckoPrice(id = 'bitcoin', { revalidate = 300 } = {}) {
    const data = await withRetry(() => fetchJson(`https://api.coingecko.com/api/v3/simple/price?ids=${id}&vs_currencies=usd&include_24hr_change=true`, { revalidate }));
    const px = data?.[id]?.usd;
    if (px == null) throw new Error('CoinGecko: no price');
    const pct = data[id].usd_24h_change ?? 0;
    return { current: px, prevClose: pct ? px / (1 + pct / 100) : px };
}

/** Coinbase spot (keyless) -> { current }. */
export async function coinbaseSpot(pair = 'BTC-USD', { revalidate = 300 } = {}) {
    const data = await withRetry(() => fetchJson(`https://api.coinbase.com/v2/prices/${pair}/spot`, { revalidate }));
    const px = parseFloat(data?.data?.amount);
    if (Number.isNaN(px)) throw new Error('Coinbase: no amount');
    return { current: px };
}

/** ExchangeRate-API open endpoint (keyless) -> rates map for a base currency. */
export async function erApiRates(base = 'USD', { revalidate = 300 } = {}) {
    const data = await withRetry(() => fetchJson(`https://open.er-api.com/v6/latest/${base}`, { revalidate }));
    if (data?.result !== 'success' || !data.rates) throw new Error('ER-API: bad response');
    return data.rates;
}

/** Frankfurter (ECB, keyless) -> rates map for a base currency. */
export async function frankfurterRates(base = 'USD', symbols = '', { revalidate = 300 } = {}) {
    const q = symbols ? `&symbols=${symbols}` : '';
    const data = await withRetry(() => fetchJson(`https://api.frankfurter.app/latest?from=${base}${q}`, { revalidate }));
    if (!data?.rates) throw new Error('Frankfurter: bad response');
    return data.rates;
}

/** DBnomics mirror of a FRED series (keyless) -> [{date,value}] newest-first. */
export async function dbnomicsFred(seriesId, { revalidate = 1800 } = {}) {
    // DBnomics indexes FRED series as FRED/<series>/<series>.
    const url = `https://api.db.nomics.world/v22/series/FRED/${seriesId}/${seriesId}?observations=1`;
    const data = await withRetry(() => fetchJson(url, { revalidate }));
    const doc = data?.series?.docs?.[0];
    if (!doc?.period || !doc?.value) throw new Error(`DBnomics: no series ${seriesId}`);
    const out = [];
    for (let i = 0; i < doc.period.length; i++) {
        const v = doc.value[i];
        if (v != null && v !== 'NA' && !Number.isNaN(Number(v))) out.push({ date: doc.period[i], value: Number(v) });
    }
    out.reverse();
    if (!out.length) throw new Error(`DBnomics: empty ${seriesId}`);
    return out;
}

/** FRED observations (newest-first [{date,value}]) with retry. */
export async function fredObservations(seriesId, apiKey, { limit = 400, revalidate = 1800 } = {}) {
    const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${seriesId}&api_key=${apiKey}&file_type=json&sort_order=desc&limit=${limit}`;
    const data = await withRetry(() => fetchJson(url, { revalidate }));
    const obs = (data.observations || []).filter((o) => o.value !== '.').map((o) => ({ date: o.date, value: parseFloat(o.value) }));
    if (!obs.length) throw new Error(`FRED: empty ${seriesId}`);
    return obs;
}

/** Build a {value,pct} daily-change object from current + previous close. */
export function dailyChange(current, prevClose) {
    const value = current - prevClose;
    return { value, pct: prevClose ? (value / prevClose) * 100 : 0 };
}
