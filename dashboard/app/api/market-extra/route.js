import { polygonDaily, erApiRates, frankfurterRates, fawazRates, dxyFromUsdRates, coinbaseSpot, coingeckoPrice, krakenSpot, fredObservations, dailyChange } from '../../../lib/sources';
import { serve } from '../../../lib/store';

export const fetchCache = 'default-cache';

/** USD-base FX rates with a 3-source fallback: ER-API -> Frankfurter -> Fawaz. */
async function usdRates() {
    for (const fn of [() => erApiRates('USD', { revalidate: 600 }), () => frankfurterRates('USD', '', { revalidate: 600 }), () => fawazRates('usd', { revalidate: 600 })]) {
        try { const r = await fn(); if (r && (r.CAD || r.INR || r.EUR)) return r; } catch { /* try next */ }
    }
    return null;
}

const HIST = 260;
const flat = (current) => ({ current, dailyChange: { value: 0, pct: 0 }, history: [], lastDate: null });
const safe = (fn) => fn().then((v) => v).catch(() => null);
const toMetric = (p) => { const history = p.history.slice(-HIST); return { current: p.current, dailyChange: dailyChange(p.current, p.prevClose), history, lastDate: history[history.length - 1]?.date ?? null }; };

async function fredMetric(seriesId, apiKey) {
    const obs = await fredObservations(seriesId, apiKey, { limit: 400, revalidate: 1800 });
    const asc = [...obs].reverse().map((o) => ({ date: o.date, price: o.value }));
    return { current: obs[0].value, dailyChange: dailyChange(obs[0].value, obs[1]?.value ?? obs[0].value), history: asc.slice(-HIST), lastDate: obs[0].date };
}
const polyMetric = (ticker, key) => async () => toMetric(await polygonDaily(ticker, key, { years: 2, revalidate: 1800 }));

const getPath = (o, p) => p.split('.').reduce((a, k) => (a ? a[k] : undefined), o);
const setPath = (o, p, v) => { const ks = p.split('.'); const last = ks.pop(); let cur = o; for (const k of ks) cur = cur[k] ??= {}; cur[last] = v; };

/** Build one metric from the best available direct source. Returns {metric, src} or null. */
async function directMetric(path, apiKey, poly, er) {
    switch (path) {
        case 'fx.usdcad': { if (poly) { const p = await safe(polyMetric('C:USDCAD', poly)); if (p) return { metric: p, src: 'Polygon' }; } return er?.CAD != null ? { metric: flat(er.CAD), src: 'ER-API' } : null; }
        case 'fx.usdinr': { if (poly) { const p = await safe(polyMetric('C:USDINR', poly)); if (p) return { metric: p, src: 'Polygon' }; } return er?.INR != null ? { metric: flat(er.INR), src: 'ER-API' } : null; }
        case 'fx.usdbdt': return er?.BDT != null ? { metric: flat(er.BDT), src: 'ER-API' } : null;
        case 'fx.dxy': { const d = dxyFromUsdRates(er); return d != null ? { metric: flat(d), src: 'computed (ER-API)' } : null; }
        case 'commodities.gc': { const p = poly && await safe(polyMetric('C:XAUUSD', poly)); if (p) return { metric: p, src: 'Polygon' }; const f = await safe(() => fredMetric('GOLDPMGBD228NLBM', apiKey)); return f ? { metric: f, src: 'FRED (London fix)' } : null; }
        case 'commodities.cl': { const f = await safe(() => fredMetric('DCOILWTICO', apiKey)); return f ? { metric: f, src: 'FRED' } : null; }
        case 'commodities.btc': {
            const p = poly && await safe(polyMetric('X:BTCUSD', poly)); if (p) return { metric: p, src: 'Polygon' };
            const cb = await safe(() => coinbaseSpot('BTC-USD', { revalidate: 300 })); if (cb) return { metric: flat(cb.current), src: 'Coinbase' };
            const cg = await safe(() => coingeckoPrice('bitcoin', { revalidate: 300 })); if (cg) return { metric: { current: cg.current, dailyChange: dailyChange(cg.current, cg.prevClose), history: [], lastDate: null }, src: 'CoinGecko' };
            const kr = await safe(() => krakenSpot('XBTUSD', { revalidate: 120 })); if (kr) return { metric: flat(kr.current), src: 'Kraken' };
            return null;
        }
        case 'rates.tnx': { const f = await safe(() => fredMetric('DGS10', apiKey)); return f ? { metric: f, src: 'FRED' } : null; }
        case 'rates.t2y': { const f = await safe(() => fredMetric('DGS2', apiKey)); return f ? { metric: f, src: 'FRED' } : null; }
        case 'rates.mortgageRate': { const f = await safe(() => fredMetric('MORTGAGE30US', apiKey)); return f ? { metric: f, src: 'FRED' } : null; }
        default: return null;
    }
}

const BASE_PATHS = ['fx.usdcad', 'fx.usdinr', 'fx.usdbdt', 'fx.dxy', 'commodities.gc', 'commodities.cl', 'commodities.btc', 'rates.tnx', 'rates.t2y', 'rates.mortgageRate'];

/** Add the computed cross-rates from whatever USD pairs are present. */
function addCrosses(out, log) {
    const cad = out.fx?.usdcad?.current, inr = out.fx?.usdinr?.current, bdt = out.fx?.usdbdt?.current;
    const cross = (a, b) => (a != null && b != null && b !== 0 ? flat(a / b) : null);
    const set = (k, m) => { if (m && (!out.fx[k] || out.fx[k].current == null)) { out.fx[k] = m; log[k] = 'computed'; } };
    if (!out.fx) out.fx = {};
    set('inrbdt', cross(bdt, inr));
    set('cadinr', cross(inr, cad));
    set('cadbdt', cross(bdt, cad));
}

async function buildDirect(apiKey, poly, er, messages) {
    const out = { fx: {}, commodities: {}, rates: {}, realEstate: {}, _meta: { source: 'Direct sources (fallback)', hasErrors: true, sourceLog: {}, messages } };
    const built = await Promise.all(BASE_PATHS.map((p) => directMetric(p, apiKey, poly, er).then((r) => [p, r]).catch(() => [p, null])));
    for (const [p, r] of built) if (r?.metric?.current != null) { setPath(out, p, r.metric); out._meta.sourceLog[p.split('.').pop()] = r.src; }
    addCrosses(out, out._meta.sourceLog);
    return out;
}

async function lambdaExtra(messages) {
    const lambdaUrl = process.env.LAMBDA_URL;
    if (!lambdaUrl) { messages.push('LAMBDA_URL not configured'); return null; }
    try {
        const res = await fetch(`${lambdaUrl}/api/market-extra`, { cache: 'no-store' });
        if (!res.ok) { messages.push(`Lambda HTTP ${res.status}`); return null; }
        const j = await res.json();
        if (j && (j.fx || j.commodities)) return j;
        messages.push('Lambda returned no usable market data');
    } catch (e) { messages.push(`Lambda failed: ${e.message}`); }
    return null;
}

export async function GET(request) {
    request.headers.get('user-agent');
    const debug = new URL(request.url).searchParams.get('debug');
    const apiKey = process.env.FRED_API_KEY;
    const poly = process.env.POLYGON_KEY || '';
    const messages = [];

    const er = await usdRates();
    const lam = await lambdaExtra(messages);

    if (debug === 'compare') {
        const direct = await buildDirect(apiKey, poly, er, []);
        const summ = (o) => o ? {
            usdcad: o.fx?.usdcad?.current, usdinr: o.fx?.usdinr?.current, usdbdt: o.fx?.usdbdt?.current, dxy: o.fx?.dxy?.current,
            gold: o.commodities?.gc?.current, crude: o.commodities?.cl?.current, btc: o.commodities?.btc?.current,
            tnx: o.rates?.tnx?.current, t2y: o.rates?.t2y?.current, mortgage: o.rates?.mortgageRate?.current,
        } : o;
        return Response.json({ polygonKeyPresent: !!poly, sourceLog: direct._meta.sourceLog, lambda: summ(lam), direct: summ(direct), messages });
    }

    // Never-throws: Lambda(+direct null-fill) -> full direct build -> last-known-good -> empty shape.
    return serve('market-extra', async () => {
        if (lam) {
            const log = (lam._meta = lam._meta || {}).sourceLog = lam._meta.sourceLog || {};
            const nullPaths = BASE_PATHS.filter((p) => { const m = getPath(lam, p); return !m || m.current == null; });
            const fills = await Promise.all(nullPaths.map((p) => directMetric(p, apiKey, poly, er).then((r) => [p, r]).catch(() => [p, null])));
            const filled = [];
            for (const [p, r] of fills) if (r?.metric?.current != null) { setPath(lam, p, r.metric); log[p.split('.').pop()] = `${r.src} (filled)`; filled.push(p.split('.').pop()); }
            if (!lam.fx) lam.fx = {};
            addCrosses(lam, log);
            if (filled.length) (lam._meta.messages = lam._meta.messages || []).push(`Filled from direct sources: ${filled.join(', ')}`);
            return lam;
        }
        const direct = await buildDirect(apiKey, poly, er, messages);
        const ok = Object.keys(direct.fx).length + Object.keys(direct.commodities).length + Object.keys(direct.rates).length;
        if (ok === 0) throw new Error(`All market sources failed: ${messages.join(' | ')}`);
        return direct;
    }, {
        isGood: (x) => x && (Object.keys(x.fx || {}).length + Object.keys(x.commodities || {}).length + Object.keys(x.rates || {}).length) > 0,
        fallback: { fx: {}, commodities: {}, rates: {}, realEstate: {} },
    });
}
