/**
 * /api/vol — IV rank / IV percentile / VRP for SPY, QQQ, TQQQ, SQQQ, UVXY.
 *
 * Dashboard-only (no Lambda hop). Never-throw via serve(). Per-series cascades
 * (owner's rule: 3+ sources so there is always a backup); every tier except the
 * last Yahoo one is datacenter-reachable:
 *   Vol indices: CBOE CDN daily-history CSVs (keyless, full history)
 *                → CNBC '.VIX'/'.VXN'/'.VVIX' daily bars (keyless, ~2y — plenty
 *                  for the 1y window; verified live 2026-07-05)
 *                → FRED VIXCLS/VXNCLS (key; VVIX has NO FRED series)
 *                → Yahoo ^VIX/^VXN/^VVIX (blocked from Vercel today — kept as a
 *                  self-heal tier, same design as copper/gold).
 *   ETF closes (21d realized vol): CNBC harmony daily bars (keyless)
 *                → Polygon daily aggs (POLYGON_KEY; free tier is a day behind,
 *                  which is fine for a 21-day realized-vol window)
 *                → Yahoo chart (self-heal tier).
 *   Live intraday overrides: CNBC quote endpoint '.VIX'/'.VXN'/'.VVIX'
 *                (keyless, 5-min revalidate; gated by vol_cnbc). Applied by
 *                buildVolMetrics only when strictly newer than the last EOD
 *                close — see lib/vol.js. Sources show it as e.g. 'VIX:cboe+live'.
 * Fault gates (one per SOURCE, tripping it everywhere it's used, like cg_*):
 * vol_cboe, vol_cnbc, vol_fred, vol_polygon, vol_yahoo.
 */
import { cnbcHistory, cnbcQuotes, polygonDaily, fredObservations, yahooChart } from '../../../lib/sources';
import { serve } from '../../../lib/store';
import { faultsFrom, gate } from '../../../lib/faults';
import { parseCboeCsv, buildVolMetrics, VOL_PROXIES } from '../../../lib/vol';

export const fetchCache = 'default-cache';

const CBOE_URL = (name) => `https://cdn.cboe.com/api/global/us_indices/daily_prices/${name}_History.csv`;
const FRED_FALLBACK = { VIX: 'VIXCLS', VXN: 'VXNCLS' }; // no VVIX series on FRED
const INDICES = ['VIX', 'VXN', 'VVIX'];
const TICKERS = Object.keys(VOL_PROXIES);

async function fetchCboe(name) {
    const res = await fetch(CBOE_URL(name), { next: { revalidate: 1800 } });
    if (!res.ok) throw new Error(`CBOE ${name}: HTTP ${res.status}`);
    const series = parseCboeCsv(await res.text());
    if (!series.length) throw new Error(`CBOE ${name}: empty/unparseable CSV`);
    return series;
}

const toSeries = (hist) => hist.map((h) => ({ date: h.date, value: h.price })).filter((p) => Number.isFinite(p.value));

async function fetchIndex(name, fredKey, faults, notes) {
    const fredId = FRED_FALLBACK[name];
    const chain = [
        ['cboe', 'vol_cboe', () => fetchCboe(name)],
        ['cnbc', 'vol_cnbc', async () => toSeries(await cnbcHistory(`.${name}`, { range: '3M' }))], // '3M' actually returns ~2y of daily bars
        ...(fredId && fredKey ? [['fred', 'vol_fred', async () => {
            const obs = await fredObservations(fredId, fredKey, { limit: 400, revalidate: 1800 });
            return [...obs].reverse().map((o) => ({ date: o.date, value: o.value })).filter((p) => Number.isFinite(p.value));
        }]] : []),
        ['yahoo', 'vol_yahoo', async () => toSeries((await yahooChart(`^${name}`, { range: '2y', interval: '1d', revalidate: 1800 })).history)],
    ];
    for (const [label, gateName, fn] of chain) {
        try {
            const series = await gate(gateName, faults, fn);
            if (series && series.length) return { series, source: label };
        } catch (e) {
            notes.push(`${name} ${label}: ${String(e?.message).slice(0, 80)}`);
        }
    }
    return { series: null, source: null };
}

async function fetchEtfCloses(ticker, polygonKey, faults, notes) {
    const chain = [
        ['cnbc', 'vol_cnbc', async () => (await cnbcHistory(ticker, { range: '3M' })).map((h) => h.price)],
        ...(polygonKey ? [['polygon', 'vol_polygon', async () => (await polygonDaily(ticker, polygonKey, { years: 1, revalidate: 1800 })).history.map((h) => h.price)]] : []),
        ['yahoo', 'vol_yahoo', async () => (await yahooChart(ticker, { range: '3mo', interval: '1d', revalidate: 1800 })).history.map((h) => h.price)],
    ];
    for (const [label, gateName, fn] of chain) {
        try {
            const closes = await gate(gateName, faults, fn);
            if (closes && closes.length) return { closes, source: label };
        } catch (e) {
            notes.push(`${ticker} ${label}: ${String(e?.message).slice(0, 80)}`);
        }
    }
    return { closes: null, source: null };
}

/**
 * Live intraday index levels — ONE keyless CNBC quote call for all three
 * indices, 5-min revalidate (vs 30-min for the daily histories). Gated by
 * vol_cnbc (per-SOURCE semantics, same gate as the CNBC daily bars). Any
 * failure returns {} — buildVolMetrics then serves EOD closes exactly as
 * before this tier existed. Same live-overrides-stale pattern as SPY's
 * Finnhub spot override.
 */
async function fetchLiveQuotes(faults, notes) {
    try {
        const quotes = await gate('vol_cnbc', faults, () => cnbcQuotes(INDICES.map((n) => `.${n}`), { revalidate: 300 }));
        const out = {};
        for (const n of INDICES) {
            const q = quotes[`.${n}`];
            if (q) out[n] = { value: q.price, date: q.asOf, lastTime: q.lastTime };
        }
        return out;
    } catch (e) {
        notes.push(`live quotes: ${String(e?.message).slice(0, 80)}`);
        return {};
    }
}

export async function GET(request) {
    // Touch the request so Next renders this handler dynamically (per request),
    // while the CBOE/CNBC/FRED fetches still come from the 30-min Data Cache.
    // Without this the route is STATICALLY PRERENDERED at build time (faultsFrom
    // can't mark it dynamic — its try/catch swallows Next's DynamicServerError),
    // which froze the payload and ignored ?_fail= on production (caught 2026-07-05).
    request.headers.get('user-agent');

    const faults = faultsFrom(request);
    const fredKey = process.env.FRED_API_KEY;
    const polygonKey = process.env.POLYGON_KEY;

    return serve('vol', async () => {
        const notes = [];
        const [indexResults, etfResults, liveQuotes] = await Promise.all([
            Promise.all(INDICES.map((n) => fetchIndex(n, fredKey, faults, notes))),
            Promise.all(TICKERS.map((t) => fetchEtfCloses(t, polygonKey, faults, notes))),
            fetchLiveQuotes(faults, notes),
        ]);
        const indexSeries = {};
        INDICES.forEach((n, i) => {
            indexSeries[n] = indexResults[i].series;
        });
        const etfCloses = {};
        const etfSources = [];
        TICKERS.forEach((t, i) => {
            etfCloses[t] = etfResults[i].closes;
            if (etfResults[i].source) etfSources.push(`${t}:${etfResults[i].source}`);
        });

        const payload = buildVolMetrics(indexSeries, etfCloses, liveQuotes);
        // Which indices actually got a live override (buildVolMetrics is the
        // single authority on that decision — derive, don't re-guess).
        const liveIndices = new Set(payload.tickers.filter((t) => t.live).map((t) => VOL_PROXIES[t.ticker].index));
        const indexSources = [];
        INDICES.forEach((n, i) => {
            if (indexResults[i].source) indexSources.push(`${n}:${indexResults[i].source}${liveIndices.has(n) ? '+live' : ''}`);
        });

        const anyData = payload.tickers.some((t) => t.iv != null || t.rv21 != null);
        return {
            ...payload,
            _meta: {
                source: indexSources.concat(etfSources).join(' · ') || 'none',
                hasErrors: !anyData,
                messages: notes,
            },
        };
    }, {
        faults,
        isGood: (p) => !!p && Array.isArray(p.tickers) && p.tickers.some((t) => t.iv != null),
        fallback: { updated_at: null, live_at: null, tickers: [], _meta: { source: 'Unavailable', hasErrors: true, messages: [] } },
    });
}
