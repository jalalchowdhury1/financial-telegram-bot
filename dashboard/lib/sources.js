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

/** Fawaz currency-api (keyless, CDN-hosted, very reliable) -> rates map (UPPERCASE). */
export async function fawazRates(base = 'usd', { revalidate = 600 } = {}) {
    const b = base.toLowerCase();
    const urls = [
        `https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/${b}.json`,
        `https://${b}.currency-api.pages.dev/v1/currencies/${b}.json`,
    ];
    let last;
    for (const u of urls) {
        try {
            const data = await withRetry(() => fetchJson(u, { revalidate }));
            const m = data?.[b];
            if (m) { const out = {}; for (const k in m) out[k.toUpperCase()] = m[k]; return out; }
        } catch (e) { last = e; }
    }
    throw new Error(`Fawaz FX: ${last?.message || 'unavailable'}`);
}

/** Kraken public ticker (keyless) -> { current } last-trade price for a pair. */
export async function krakenSpot(pair = 'XBTUSD', { revalidate = 120 } = {}) {
    const data = await withRetry(() => fetchJson(`https://api.kraken.com/0/public/Ticker?pair=${pair}`, { revalidate }));
    const res = data?.result;
    const k = res && Object.keys(res)[0];
    const px = k && parseFloat(res[k]?.c?.[0]);
    if (!px || Number.isNaN(px)) throw new Error('Kraken: no price');
    return { current: px };
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

/**
 * Polygon daily aggregates (server-friendly, the same vendor the Lambda uses) ->
 * { current, prevClose, history }. ticker examples: 'SPY', 'C:XAUUSD' (gold),
 * 'C:USDCAD' (fx), 'X:BTCUSD' (crypto). Requires a Polygon API key.
 */
export async function polygonDaily(ticker, key, { years = 2, revalidate = 1800 } = {}) {
    if (!key) throw new Error('Polygon: no API key');
    const to = new Date().toISOString().slice(0, 10);
    const from = new Date(Date.now() - years * 365 * 864e5).toISOString().slice(0, 10);
    const url = `https://api.polygon.io/v2/aggs/ticker/${encodeURIComponent(ticker)}/range/1/day/${from}/${to}?adjusted=true&sort=asc&limit=50000&apiKey=${key}`;
    const data = await withRetry(() => fetchJson(url, { revalidate }));
    if (data?.status === 'ERROR' || !Array.isArray(data?.results)) throw new Error(`Polygon: ${data?.error || data?.message || 'no results'} for ${ticker}`);
    const history = data.results.map((r) => ({ date: new Date(r.t).toISOString().slice(0, 10), price: r.c })).filter((h) => h.price != null);
    if (!history.length) throw new Error(`Polygon: empty ${ticker}`);
    return { current: history[history.length - 1].price, prevClose: history[history.length - 2]?.price ?? history[history.length - 1].price, history };
}

/**
 * CNBC quote service (KEYLESS, datacenter-friendly JSON) -> map symbol -> {price,
 * change, changePct, asOf, lastTime}. Pass continuous front-month future symbols, e.g.
 * '@HG.1' (Copper, USD/lb) and '@GC.1' (Gold, USD/oz). Multiple symbols MUST be
 * pipe-delimited in ONE request (repeated symbols= params error). last/change/
 * change_pct come back as STRINGS.
 */
export async function cnbcQuotes(symbols, { revalidate = 1800, timeout = 6000, tries = 2 } = {}) {
    const q = symbols.map((s) => encodeURIComponent(s)).join('%7C'); // %7C = '|'
    const url = `https://quote.cnbc.com/quote-html-webservice/quote.htm?symbols=${q}&output=json`;
    const data = await withRetry(() => fetchJson(url, { revalidate, timeout }), { tries });
    const raw = data?.QuickQuoteResult?.QuickQuote;
    const list = Array.isArray(raw) ? raw : raw ? [raw] : [];
    const out = {};
    for (const it of list) {
        const price = parseFloat(it.last);
        if (!Number.isFinite(price)) continue;
        out[it.symbol] = {
            price,
            change: parseFloat(it.change),
            changePct: parseFloat(it.change_pct),
            asOf: typeof it.last_time === 'string' ? it.last_time.slice(0, 10) : null,
            lastTime: typeof it.last_time === 'string' ? it.last_time : null,
        };
    }
    if (!Object.keys(out).length) throw new Error(`CNBC: no quotes for ${symbols.join(',')}`);
    return out;
}

/**
 * CNBC daily price history (KEYLESS) -> ascending [{date,price}]. The '3M.json'
 * range actually returns ~500 DAILY bars (~2yr) — plenty for 1mo/3mo deltas.
 * tradeTime is 'YYYYMMDDhhmmss'; close is a string.
 */
export async function cnbcHistory(symbol, { range = '3M', revalidate = 1800, timeout = 6000, tries = 2 } = {}) {
    const url = `https://ts-api.cnbc.com/harmony/app/charts/${range}.json?symbol=${encodeURIComponent(symbol)}`;
    const data = await withRetry(() => fetchJson(url, { revalidate, timeout }), { tries });
    const bars = data?.barData?.priceBars || [];
    const history = [];
    for (const b of bars) {
        const price = parseFloat(b.close);
        const tt = String(b.tradeTime || '');
        if (!Number.isFinite(price) || tt.length < 8) continue;
        history.push({ date: `${tt.slice(0, 4)}-${tt.slice(4, 6)}-${tt.slice(6, 8)}`, price });
    }
    if (!history.length) throw new Error(`CNBC: empty history ${symbol}`);
    history.sort((a, b) => (a.date < b.date ? -1 : 1)); // ISO 'YYYY-MM-DD' sorts chronologically as strings → oldest -> newest
    return history;
}

/** gold-api.com spot (KEYLESS) -> { current, asOf }. symbol 'XAU' (gold $/oz) or 'HG' (copper $/lb). */
export async function goldApiSpot(symbol, { revalidate = 600, timeout = 6000, tries = 2 } = {}) {
    const data = await withRetry(() => fetchJson(`https://api.gold-api.com/price/${encodeURIComponent(symbol)}`, { revalidate, timeout }), { tries });
    const px = Number(data?.price);
    if (!Number.isFinite(px)) throw new Error(`gold-api: no price ${symbol}`);
    return { current: px, asOf: typeof data.updatedAt === 'string' ? data.updatedAt.slice(0, 10) : null };
}

/** Finnhub real-time quote (stocks) -> { current, prevClose }. Requires a key. */
export async function finnhubQuote(symbol, key, { revalidate = 120 } = {}) {
    if (!key) throw new Error('Finnhub: no API key');
    const data = await withRetry(() => fetchJson(`https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${key}`, { revalidate }));
    if (data?.c == null || data.c === 0) throw new Error('Finnhub: no quote');
    return { current: data.c, prevClose: data.pc ?? data.c };
}

/** Compute the US Dollar Index (DXY) from USD-base rates (ICE formula). */
export function dxyFromUsdRates(r) {
    const need = ['EUR', 'JPY', 'GBP', 'CAD', 'SEK', 'CHF'];
    if (!r || need.some((k) => !(typeof r[k] === 'number'))) return null;
    const EURUSD = 1 / r.EUR, USDJPY = r.JPY, GBPUSD = 1 / r.GBP, USDCAD = r.CAD, USDSEK = r.SEK, USDCHF = r.CHF;
    return 50.14348112 * EURUSD ** -0.576 * USDJPY ** 0.136 * GBPUSD ** -0.119 * USDCAD ** 0.091 * USDSEK ** 0.042 * USDCHF ** 0.036;
}

/** FRED observations (newest-first [{date,value}]) with retry. */
export async function fredObservations(seriesId, apiKey, { limit = 400, revalidate = 1800 } = {}) {
    const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${seriesId}&api_key=${apiKey}&file_type=json&sort_order=desc&limit=${limit}`;
    const data = await withRetry(() => fetchJson(url, { revalidate }));
    const obs = (data.observations || []).filter((o) => o.value !== '.').map((o) => ({ date: o.date, value: parseFloat(o.value) }));
    if (!obs.length) throw new Error(`FRED: empty ${seriesId}`);
    return obs;
}

// ── Four Horsemen fallback providers (see lib/horsemen.js for the cascades) ──
// These are ORIGIN sources for the FRED series they back up: BLS publishes the
// unemployment rate, Treasury publishes the yield curve. They return RAW text /
// JSON; parsing lives in lib/horsemen.js so it stays pure and unit-testable.

/**
 * One calendar year of the US Treasury daily yield curve, as CSV.
 * Keyless and unthrottled, but strictly one year per request — callers that want
 * a multi-year history fetch several years and concatenate.
 */
export async function treasuryYieldCurveCsv(year, { revalidate = 1800, timeout = 8000 } = {}) {
    const url = `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/${year}/all`
        + `?type=daily_treasury_yield_curve&field_tdr_date_value=${year}&page&_format=csv`;
    return withRetry(() => fetchText(url, { revalidate, timeout }), { tries: 2 });
}

/**
 * BLS Public Data API v2 timeseries.
 *
 * GOTCHA: the keyless tier caps a request at 10 years and, when you ask for
 * more, silently returns the OLDEST 10 years of the span rather than erroring —
 * so a naive startYear=1948 request yields 1948-1957 and the series looks dead.
 * Callers must anchor the span to the CURRENT year. A free registered key
 * (BLS_API_KEY) raises the daily quota from 25 to 500 requests but not the span.
 */
// revalidate defaults to 0: Next never caches a non-GET fetch, and passing a
// revalidate on a POST is a runtime complaint in some Next versions.
export async function blsSeries(seriesId, { startYear, endYear, key = '', revalidate = 0, timeout = 8000 } = {}) {
    const body = { seriesid: [seriesId], startyear: String(startYear), endyear: String(endYear) };
    if (key) body.registrationkey = key;
    return withRetry(() => fetchJson('https://api.bls.gov/publicAPI/v2/timeseries/data/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        revalidate,
        timeout,
    }), { tries: 2 });
}

/**
 * FRED's keyless graph CSV for a series (full history, no api key, no quota).
 * Same servers as the JSON API, so it does NOT survive a FRED outage — but it
 * DOES survive a revoked/exhausted/misconfigured api key, which is the far more
 * likely failure. Hence its position as the last tier in every cascade.
 */
export async function fredGraphCsv(seriesId, { revalidate = 1800, timeout = 8000 } = {}) {
    const url = `https://fred.stlouisfed.org/graph/fredgraph.csv?id=${encodeURIComponent(seriesId)}`;
    return withRetry(() => fetchText(url, { revalidate, timeout }), { tries: 2 });
}

/** Build a {value,pct} daily-change object from current + previous close. */
export function dailyChange(current, prevClose) {
    const value = current - prevClose;
    return { value, pct: prevClose ? (value / prevClose) * 100 : 0 };
}
