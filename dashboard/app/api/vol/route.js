/**
 * /api/vol — IV rank / IV percentile / VRP for SPY, QQQ, TQQQ, SQQQ, UVXY.
 *
 * Dashboard-only (no Lambda hop). Never-throw via serve(). Sources, all
 * datacenter-reachable (Yahoo/Stooq are blocked from Vercel):
 *   Vol indices: CBOE CDN daily-history CSVs (keyless, full history)
 *                → FRED VIXCLS/VXNCLS (key) — VVIX has NO FRED series, so a
 *                  CBOE outage degrades only the UVXY row (rank/pctile null).
 *   ETF closes (21d realized vol): CNBC harmony daily bars (keyless)
 *                → Polygon daily aggs (POLYGON_KEY; free tier is a day behind,
 *                  which is fine for a 21-day realized-vol window).
 * Fault gates: vol_cboe, vol_fred, vol_cnbc, vol_polygon.
 */
import { cnbcHistory, polygonDaily, fredObservations } from '../../../lib/sources';
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

async function fetchIndex(name, fredKey, faults, notes) {
    try {
        const series = await gate('vol_cboe', faults, () => fetchCboe(name));
        return { series, source: 'cboe' };
    } catch (e) {
        notes.push(`${name} cboe: ${String(e?.message).slice(0, 80)}`);
    }
    const fredId = FRED_FALLBACK[name];
    if (fredId && fredKey) {
        try {
            const obs = await gate('vol_fred', faults, () => fredObservations(fredId, fredKey, { limit: 400, revalidate: 1800 }));
            const series = [...obs].reverse().map((o) => ({ date: o.date, value: o.value })).filter((p) => Number.isFinite(p.value));
            if (series.length) return { series, source: 'fred' };
        } catch (e) {
            notes.push(`${name} fred: ${String(e?.message).slice(0, 80)}`);
        }
    }
    return { series: null, source: null };
}

async function fetchEtfCloses(ticker, polygonKey, faults, notes) {
    try {
        const hist = await gate('vol_cnbc', faults, () => cnbcHistory(ticker, { range: '3M' }));
        return { closes: hist.map((h) => h.price), source: 'cnbc' };
    } catch (e) {
        notes.push(`${ticker} cnbc: ${String(e?.message).slice(0, 80)}`);
    }
    if (polygonKey) {
        try {
            const p = await gate('vol_polygon', faults, () => polygonDaily(ticker, polygonKey, { years: 1, revalidate: 1800 }));
            return { closes: p.history.map((h) => h.price), source: 'polygon' };
        } catch (e) {
            notes.push(`${ticker} polygon: ${String(e?.message).slice(0, 80)}`);
        }
    }
    return { closes: null, source: null };
}

export async function GET(request) {
    const faults = faultsFrom(request);
    const fredKey = process.env.FRED_API_KEY;
    const polygonKey = process.env.POLYGON_KEY;

    return serve('vol', async () => {
        const notes = [];
        const [indexResults, etfResults] = await Promise.all([
            Promise.all(INDICES.map((n) => fetchIndex(n, fredKey, faults, notes))),
            Promise.all(TICKERS.map((t) => fetchEtfCloses(t, polygonKey, faults, notes))),
        ]);
        const indexSeries = {};
        const indexSources = [];
        INDICES.forEach((n, i) => {
            indexSeries[n] = indexResults[i].series;
            if (indexResults[i].source) indexSources.push(`${n}:${indexResults[i].source}`);
        });
        const etfCloses = {};
        const etfSources = [];
        TICKERS.forEach((t, i) => {
            etfCloses[t] = etfResults[i].closes;
            if (etfResults[i].source) etfSources.push(`${t}:${etfResults[i].source}`);
        });

        const payload = buildVolMetrics(indexSeries, etfCloses);
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
        fallback: { updated_at: null, tickers: [], _meta: { source: 'Unavailable', hasErrors: true, messages: [] } },
    });
}
