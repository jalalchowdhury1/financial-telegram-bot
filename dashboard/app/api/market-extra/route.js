import { polygonDaily, erApiRates, frankfurterRates, fawazRates, dxyFromUsdRates, coinbaseSpot, coingeckoPrice, krakenSpot, fredObservations, dailyChange, goldApiSpot } from '../../../lib/sources';
import { serve } from '../../../lib/store';
import { faultsFrom, gate } from '../../../lib/faults';
import { makeTiers, mergeLastGood } from '../../../lib/marketExtraTiers';

export const fetchCache = 'default-cache';
export const maxDuration = 30;

// Timing budget (maxDuration 30 s). The Lambda hop can take up to API Gateway's 30 s
// cap, so the direct-source fill phase gets whatever is left of FILL_DEADLINE_MS from
// request start (each metric's chain races it -> null). The last-good merge after it
// is a local read, so even a Lambda that eats the whole budget still yields
// yesterday's numbers instead of blanks.
const FILL_DEADLINE_MS = 24000;
const MIN_FILL_MS = 1500;

/** USD-base FX rates with a 3-source fallback: ER-API -> Frankfurter -> Fawaz. */
async function usdRates(faults = new Set()) {
    const chain = [['erapi', () => erApiRates('USD', { revalidate: 600 })], ['frankfurter', () => frankfurterRates('USD', '', { revalidate: 600 })], ['fawaz', () => fawazRates('usd', { revalidate: 600 })]];
    for (const [name, fn] of chain) {
        if (faults.has(name)) continue;
        try { const r = await fn(); if (r && (r.CAD || r.INR || r.EUR)) return r; } catch { /* try next */ }
    }
    return null;
}

const HIST = 260;
const flat = (current) => ({ current, dailyChange: { value: 0, pct: 0 }, history: [], lastDate: null });
const safe = (fn) => Promise.resolve().then(fn).catch(() => null);
const toMetric = (p) => { const history = p.history.slice(-HIST); return { current: p.current, dailyChange: dailyChange(p.current, p.prevClose), history, lastDate: history[history.length - 1]?.date ?? null }; };

async function fredMetric(seriesId, apiKey) {
    const obs = await fredObservations(seriesId, apiKey, { limit: 400, revalidate: 1800 });
    const asc = [...obs].reverse().map((o) => ({ date: o.date, price: o.value }));
    return { current: obs[0].value, dailyChange: dailyChange(obs[0].value, obs[1]?.value ?? obs[0].value), history: asc.slice(-HIST), lastDate: obs[0].date };
}
const polyMetric = (ticker, key) => async () => toMetric(await polygonDaily(ticker, key, { years: 2, revalidate: 1800 }));

const getPath = (o, p) => p.split('.').reduce((a, k) => (a ? a[k] : undefined), o);
const setPath = (o, p, v) => { const ks = p.split('.'); const last = ks.pop(); let cur = o; for (const k of ks) cur = cur[k] ??= {}; cur[last] = v; };

/**
 * Build one metric from the best available direct source. Returns {metric, src} or null.
 * ctx = { apiKey, poly, er, faults, tiers } (tiers = lib/marketExtraTiers makeTiers()).
 * The keyless backup tiers each sit behind a `?_fail=` gate: gold_api, treasury, pmms,
 * cnbc (both CNBC tiers) / cnbc_cl / cnbc_dxy, dxy_computed, fawaz_bdt.
 */
async function directMetric(path, { apiKey, poly, er, faults = new Set(), tiers }) {
    const tier = (name, fn) => safe(() => gate(name, faults, fn));
    const cnbc = (name, fn) => safe(() => gate('cnbc', faults, () => gate(name, faults, fn)));
    switch (path) {
        case 'fx.usdcad': { if (poly) { const p = await safe(polyMetric('C:USDCAD', poly)); if (p) return { metric: p, src: 'Polygon' }; } return er?.CAD != null ? { metric: flat(er.CAD), src: 'ER-API' } : null; }
        case 'fx.usdinr': { if (poly) { const p = await safe(polyMetric('C:USDINR', poly)); if (p) return { metric: p, src: 'Polygon' }; } return er?.INR != null ? { metric: flat(er.INR), src: 'ER-API' } : null; }
        case 'fx.usdbdt': {
            if (er?.BDT != null) return { metric: flat(er.BDT), src: 'ER-API' };
            // Frankfurter (usdRates' 2nd tier) has no BDT, so a dead ER-API used to blank this.
            const f = await tier('fawaz_bdt', () => tiers.usdbdt());
            return f ? { metric: { ...flat(f.current), lastDate: f.asOf }, src: 'Fawaz' } : null;
        }
        case 'fx.dxy': {
            // The real ICE index first; the basket computed from FX rates only as a last resort.
            const c = await cnbc('cnbc_dxy', () => tiers.dxy()); if (c) return { metric: c, src: 'CNBC' };
            if (faults.has('dxy_computed')) return null;
            const d = dxyFromUsdRates(er); return d != null ? { metric: flat(d), src: 'computed (FX basket)' } : null;
        }
        case 'commodities.gc': { const p = poly && await safe(polyMetric('C:XAUUSD', poly)); if (p) return { metric: p, src: 'Polygon' }; const g = await tier('gold_api', () => goldApiSpot('XAU')); return g ? { metric: flat(g.current), src: 'gold-api' } : null; }  // FRED's GOLDPMGBD228NLBM is discontinued
        case 'commodities.cl': {
            const f = await safe(() => fredMetric('DCOILWTICO', apiKey)); if (f) return { metric: f, src: 'FRED' };
            const c = await cnbc('cnbc_cl', () => tiers.oil()); return c ? { metric: c, src: 'CNBC' } : null;
        }
        case 'commodities.btc': {
            const p = poly && await safe(polyMetric('X:BTCUSD', poly)); if (p) return { metric: p, src: 'Polygon' };
            if (!faults.has('coinbase')) { const cb = await safe(() => coinbaseSpot('BTC-USD', { revalidate: 300 })); if (cb) return { metric: flat(cb.current), src: 'Coinbase' }; }
            if (!faults.has('coingecko')) { const cg = await safe(() => coingeckoPrice('bitcoin', { revalidate: 300 })); if (cg) return { metric: { current: cg.current, dailyChange: dailyChange(cg.current, cg.prevClose), history: [], lastDate: null }, src: 'CoinGecko' }; }
            if (!faults.has('kraken')) { const kr = await safe(() => krakenSpot('XBTUSD', { revalidate: 120 })); if (kr) return { metric: flat(kr.current), src: 'Kraken' }; }
            return null;
        }
        case 'rates.tnx': {
            const f = await safe(() => fredMetric('DGS10', apiKey)); if (f) return { metric: f, src: 'FRED' };
            const t = await tier('treasury', () => tiers.tnx()); return t ? { metric: t, src: 'US Treasury' } : null;
        }
        case 'rates.t2y': {
            const f = await safe(() => fredMetric('DGS2', apiKey)); if (f) return { metric: f, src: 'FRED' };
            const t = await tier('treasury', () => tiers.t2y()); return t ? { metric: t, src: 'US Treasury' } : null;
        }
        case 'rates.mortgageRate': {
            const f = await safe(() => fredMetric('MORTGAGE30US', apiKey)); if (f) return { metric: f, src: 'FRED' };
            const m = await tier('pmms', () => tiers.mortgage()); return m ? { metric: m, src: 'Freddie Mac PMMS' } : null;
        }
        default: return null;
    }
}

/** Race a metric's chain against the request's fill deadline (late -> null, never throws). */
function withinDeadline(promise, deadlineAt) {
    const ms = Math.max(MIN_FILL_MS, deadlineAt - Date.now());
    let t;
    const timer = new Promise((r) => { t = setTimeout(() => r(null), ms); });
    return Promise.race([promise.catch(() => null), timer]).finally(() => clearTimeout(t));
}

const BASE_PATHS = ['fx.usdcad', 'fx.usdinr', 'fx.usdbdt', 'fx.dxy', 'commodities.gc', 'commodities.cl', 'commodities.btc', 'rates.tnx', 'rates.t2y', 'rates.mortgageRate'];
// Everything the page shows: the base metrics + the computed crosses + the Lambda-only
// real-estate prints. The per-metric last-good merge fills any of these the live build lacks.
const LG_PATHS = [...BASE_PATHS, 'fx.inrbdt', 'fx.cadinr', 'fx.cadbdt', 'realEstate.rentIndex', 'realEstate.mortgagePayment', 'realEstate.atnhpi'];
const UNAVAILABLE_RE = /unavailable:\s*\d+\s*metrics/i;

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

async function buildDirect(ctx, messages, deadlineAt) {
    const out = { fx: {}, commodities: {}, rates: {}, realEstate: {}, _meta: { source: 'Direct sources (fallback)', hasErrors: false, sourceLog: {}, messages } };
    const built = await Promise.all(BASE_PATHS.map((p) => withinDeadline(directMetric(p, ctx), deadlineAt).then((r) => [p, r])));
    for (const [p, r] of built) if (r?.metric?.current != null) { setPath(out, p, r.metric); out._meta.sourceLog[p.split('.').pop()] = r.src; }
    addCrosses(out, out._meta.sourceLog);
    // Derive health from what's actually missing (don't hardcode red): a full
    // direct build is healthy data, just from fallback sources. Hardcoding
    // hasErrors:true left market-extra perpetually degraded whenever the Lambda
    // was down (e.g. a transient Lambda 503) even though the fallback filled
    // every metric -- same reconciliation the Lambda path above already does.
    const stillNull = BASE_PATHS.filter((p) => { const m = getPath(out, p); return !m || m.current == null; });
    if (stillNull.length) out._meta.messages.push(`unavailable: ${stillNull.length} metrics`);
    out._meta.hasErrors = stillNull.length > 0;
    return out;
}

/**
 * Per-metric last-known-good: fill every metric the live build is missing from the
 * last good copy (each stamped `stale: true` + `savedAt`), then re-derive health. A
 * payload carrying ANY borrowed metric is degraded (hasErrors + stale -> red footer,
 * never edge-cached); the "unavailable" count is only what is still blank. Before
 * this, a lone gold-api print passed isGood and the page showed 12 blanks instead of
 * yesterday's numbers.
 */
async function fillFromLastGood(out, faults) {
    const filled = await mergeLastGood(out, { key: 'market-extra', paths: LG_PATHS, faults });
    if (!filled.length) return out;
    const meta = out._meta = out._meta || {};
    const stillNull = BASE_PATHS.filter((p) => { const m = getPath(out, p); return !m || m.current == null; });
    meta.messages = (meta.messages || []).filter((m) => !UNAVAILABLE_RE.test(m));
    meta.messages.push(`Stale (last-known-good): ${filled.join(', ')}`);
    if (stillNull.length) meta.messages.push(`unavailable: ${stillNull.length} metrics`);
    meta.staleMetrics = filled;
    meta.stale = true;
    meta.hasErrors = true;
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
    const startedAt = Date.now();
    const debug = new URL(request.url).searchParams.get('debug');
    const faults = faultsFrom(request);
    const apiKey = faults.has('fred') ? '' : (process.env.FRED_API_KEY || '');
    const poly = faults.has('polygon') ? '' : (process.env.POLYGON_KEY || '');
    const messages = [];
    const tiers = makeTiers();
    const deadlineAt = startedAt + FILL_DEADLINE_MS;

    const er = await usdRates(faults);
    const lam = faults.has('lambda') ? null : await lambdaExtra(messages);
    const ctx = { apiKey, poly, er, faults, tiers };

    if (debug === 'compare') {
        const direct = await buildDirect(ctx, [], deadlineAt);
        const summ = (o) => o ? {
            usdcad: o.fx?.usdcad?.current, usdinr: o.fx?.usdinr?.current, usdbdt: o.fx?.usdbdt?.current, dxy: o.fx?.dxy?.current,
            gold: o.commodities?.gc?.current, crude: o.commodities?.cl?.current, btc: o.commodities?.btc?.current,
            tnx: o.rates?.tnx?.current, t2y: o.rates?.t2y?.current, mortgage: o.rates?.mortgageRate?.current,
        } : o;
        return Response.json({ polygonKeyPresent: !!poly, sourceLog: direct._meta.sourceLog, lambda: summ(lam), direct: summ(direct), messages });
    }

    // Never-throws: Lambda(+direct null-fill) -> full direct build -> per-metric
    // last-known-good fill -> whole last-known-good -> empty shape.
    return serve('market-extra', async () => {
        if (lam) {
            const log = (lam._meta = lam._meta || {}).sourceLog = lam._meta.sourceLog || {};
            const nullPaths = BASE_PATHS.filter((p) => { const m = getPath(lam, p); return !m || m.current == null; });
            const fills = await Promise.all(nullPaths.map((p) => withinDeadline(directMetric(p, ctx), deadlineAt).then((r) => [p, r])));
            const filled = [];
            for (const [p, r] of fills) if (r?.metric?.current != null) { setPath(lam, p, r.metric); log[p.split('.').pop()] = `${r.src} (filled)`; filled.push(p.split('.').pop()); }
            if (!lam.fx) lam.fx = {};
            addCrosses(lam, log);
            // Reconcile the Lambda's _meta with what we actually serve after the
            // direct backfill: drop its now-stale "unavailable: N metrics" note and
            // recompute health from what is STILL missing. Otherwise the status
            // footer stays red for metrics we already successfully backfilled.
            const stillNull = BASE_PATHS.filter((p) => { const m = getPath(lam, p); return !m || m.current == null; });
            lam._meta = lam._meta || {};
            lam._meta.messages = (lam._meta.messages || []).filter((m) => !UNAVAILABLE_RE.test(m));
            if (filled.length) lam._meta.messages.push(`Filled from direct sources: ${filled.join(', ')}`);
            if (stillNull.length) lam._meta.messages.push(`unavailable: ${stillNull.length} metrics`);
            lam._meta.hasErrors = stillNull.length > 0;
            return fillFromLastGood(lam, faults);
        }
        const direct = await buildDirect(ctx, messages, deadlineAt);
        const ok = Object.keys(direct.fx).length + Object.keys(direct.commodities).length + Object.keys(direct.rates).length;
        if (ok === 0) throw new Error(`All market sources failed: ${messages.join(' | ')}`);
        return fillFromLastGood(direct, faults);
    }, {
        isGood: (x) => x && (Object.keys(x.fx || {}).length + Object.keys(x.commodities || {}).length + Object.keys(x.rates || {}).length) > 0,
        fallback: { fx: {}, commodities: {}, rates: {}, realEstate: {} },
        faults,
    });
}
