import { yahooChart, coinbaseSpot, erApiRates, fredObservations, dailyChange } from '../../../lib/sources';

export const fetchCache = 'default-cache';

const HIST = 260;
const isNum = (x) => typeof x === 'number' && !Number.isNaN(x);

/** A normalized market metric: { current, dailyChange:{value,pct}, history, lastDate }. */
async function yahooMetric(ticker, range = '1y') {
    const y = await yahooChart(ticker, { range, interval: '1d', revalidate: 300 });
    const history = y.history.slice(-HIST);
    return { current: y.current, dailyChange: dailyChange(y.current, y.prevClose), history, lastDate: history[history.length - 1]?.date ?? null };
}

async function fredMetric(seriesId, apiKey, limit = 400) {
    const obs = await fredObservations(seriesId, apiKey, { limit, revalidate: 1800 });
    const asc = [...obs].reverse().map((o) => ({ date: o.date, price: o.value }));
    return { current: obs[0].value, dailyChange: dailyChange(obs[0].value, obs[1]?.value ?? obs[0].value), history: asc.slice(-HIST), lastDate: obs[0].date };
}

const flat = (current) => ({ current, dailyChange: { value: 0, pct: 0 }, history: [], lastDate: null });
const safe = (fn) => fn().then((v) => v).catch(() => null);

/** Builders for each metric, used both to fill Lambda nulls and to build a full
 *  fallback when the Lambda is unreachable. Each is independent + reliable
 *  (Yahoo / ER-API / FRED / Coinbase — all measured reachable from Vercel). */
function builders(apiKey) {
    return {
        'fx.usdcad': () => yahooMetric('CAD=X'),
        'fx.usdinr': () => yahooMetric('INR=X'),
        'fx.dxy': () => yahooMetric('DX-Y.NYB'),
        'commodities.gc': () => yahooMetric('GC=F'),
        'commodities.cl': () => yahooMetric('CL=F'),
        'commodities.btc': () => yahooMetric('BTC-USD').catch(async () => flat((await coinbaseSpot('BTC-USD', { revalidate: 300 })).current)),
        'rates.tnx': () => fredMetric('DGS10', apiKey),
        'rates.t2y': () => fredMetric('DGS2', apiKey),
        'rates.mortgageRate': () => fredMetric('MORTGAGE30US', apiKey),
    };
}

const getPath = (o, p) => p.split('.').reduce((a, k) => (a ? a[k] : undefined), o);
const setPath = (o, p, v) => { const ks = p.split('.'); const last = ks.pop(); let cur = o; for (const k of ks) cur = cur[k] ??= {}; cur[last] = v; };

/** Fill ER-API-derived FX (BDT + cross rates) which Yahoo doesn't cover well. */
async function fillFxCrosses(fx, log) {
    let er = null;
    try { er = await erApiRates('USD', { revalidate: 600 }); } catch { return; }
    const cad = fx.usdcad?.current ?? er.CAD;
    const inr = fx.usdinr?.current ?? er.INR;
    const bdt = er.BDT;
    if ((!fx.usdbdt || fx.usdbdt.current == null) && isNum(bdt)) { fx.usdbdt = flat(bdt); log.usdbdt = 'ER-API'; }
    const cross = (a, b) => (isNum(a) && isNum(b) && b !== 0 ? flat(a / b) : null);
    const set = (key, a, b, src) => { if ((!fx[key] || fx[key].current == null) && cross(a, b)) { fx[key] = cross(a, b); log[key] = src; } };
    set('inrbdt', bdt, inr, 'computed'); // INR->BDT = BDT/INR
    set('cadinr', inr, cad, 'computed'); // CAD->INR = INR/CAD
    set('cadbdt', bdt, cad, 'computed'); // CAD->BDT = BDT/CAD
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

async function buildDirect(apiKey, messages) {
    const data = { fx: {}, commodities: {}, rates: {}, realEstate: {}, _meta: { source: 'Direct sources (fallback)', hasErrors: true, sourceLog: {}, messages } };
    const B = builders(apiKey);
    const results = await Promise.all(Object.keys(B).map((p) => safe(B[p]).then((v) => [p, v])));
    for (const [p, v] of results) if (v && v.current != null) { setPath(data, p, v); data._meta.sourceLog[p.split('.').pop()] = p.startsWith('rates') ? 'FRED' : 'Yahoo'; }
    await fillFxCrosses(data.fx, data._meta.sourceLog);
    return data;
}

export async function GET(request) {
    request.headers.get('user-agent');
    const debug = new URL(request.url).searchParams.get('debug');
    const apiKey = process.env.FRED_API_KEY;
    const messages = [];

    const lam = await lambdaExtra(messages);

    if (debug === 'compare') {
        const direct = await buildDirect(apiKey, []).catch((e) => ({ _err: e.message }));
        const summ = (o) => o && !o._err ? {
            usdcad: o.fx?.usdcad?.current, usdinr: o.fx?.usdinr?.current, dxy: o.fx?.dxy?.current,
            gold: o.commodities?.gc?.current, crude: o.commodities?.cl?.current, btc: o.commodities?.btc?.current,
            tnx: o.rates?.tnx?.current, t2y: o.rates?.t2y?.current, mortgage: o.rates?.mortgageRate?.current,
        } : o;
        return Response.json({ lambda: summ(lam), direct: summ(direct), messages });
    }

    // Lambda up: fill only the metrics it left null/missing (e.g. crude `cl`).
    if (lam) {
        const B = builders(apiKey);
        const log = (lam._meta = lam._meta || {}).sourceLog = lam._meta.sourceLog || {};
        const filled = [];
        const fills = await Promise.all(
            Object.keys(B).filter((p) => { const m = getPath(lam, p); return !m || m.current == null; })
                .map((p) => safe(B[p]).then((v) => [p, v])),
        );
        for (const [p, v] of fills) if (v && v.current != null) { setPath(lam, p, v); log[p.split('.').pop()] = (p.startsWith('rates') ? 'FRED' : 'Yahoo') + ' (filled)'; filled.push(p); }
        if (!lam.fx) lam.fx = {};
        await fillFxCrosses(lam.fx, log);
        if (filled.length) (lam._meta.messages = lam._meta.messages || []).push(`Filled from direct sources: ${filled.join(', ')}`);
        return Response.json(lam);
    }

    // Lambda down: build everything from direct sources.
    try {
        const direct = await buildDirect(apiKey, messages);
        return Response.json(direct);
    } catch (e) {
        return Response.json({ error: 'All market sources failed', detail: [...messages, e.message].join(' | ') }, { status: 500 });
    }
}
