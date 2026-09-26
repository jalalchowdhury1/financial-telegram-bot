/**
 * /api/factors — 🧬 the factor row: Value / Momentum / Quality / Small caps / Low vol,
 * each as a price ratio against the S&P 500 (SPY), over 8 windows (1M … 10Y).
 * Math + cascade logic live in lib/factors.js (pure, unit-tested).
 *
 * Dashboard-only (no Lambda hop). Never-throw via serve(). Layers, deepest last:
 *
 *   per ticker, RECENT daily bars (must be ≤5 days old to win outright):
 *     1. CNBC harmony '1Y'  (keyless; ~2y daily)
 *     2. Nasdaq historical  (keyless; ~10y daily — also covers the long history)
 *     3. Polygon aggs       (POLYGON_KEY; 2y daily; free tier is 5 req/min, so it is
 *                            deliberately NOT first — CNBC/Nasdaq absorb the load)
 *     4. Yahoo chart 10y/1d (raw closes; 429s from Vercel as of 2026-07 — kept as a
 *                            free best-effort tier, never counted as redundancy)
 *   per ticker, LONG history (only the part older than the recent series is used):
 *     1. CNBC harmony '5Y'  (keyless; weekly, Sunday-dated → shifted to Friday)
 *     2. Nasdaq (same call as above, memoised)
 *     3. Yahoo (same call as above, memoised)
 *     4. baked weekly closes in lib/data/factorsBaked.json (scripts/bake-factors.mjs)
 *   route level:
 *     5. /tmp last-known-good            (serve(); also beats a STALE live payload)
 *     6. Upstash KV last-known-good      (lib/factorStore.js; survives cold starts)
 *     7. the all-baked payload           (stale-flagged; never blank)
 *
 * Fault gates (?_fail=…): fx_cnbc, fx_cnbcw, fx_nasdaq, fx_polygon, fx_yahoo,
 * fx_baked, fx_kv — plus serve()'s `lastgood` (skip /tmp) and `sheetlkg` (skip
 * the last-resort hook, i.e. KV here).
 */
import { cnbcHistory, nasdaqHistory, polygonDaily, yahooChart } from '../../../lib/sources';
import { serve } from '../../../lib/store';
import { faultsFrom, gate, trip } from '../../../lib/faults';
import {
    BENCH, TICKERS, resolveTicker, buildPayload, isFreshGoodPayload, isStorablePayload, callBudget,
    weeklyToFriday, todayET,
} from '../../../lib/factors';
import { loadFactorsKV, saveFactorsKV } from '../../../lib/factorStore';
import baked from '../../../lib/data/factorsBaked.json';

export const fetchCache = 'default-cache';
export const maxDuration = 30;

// Timing budget (maxDuration 30 s). The deadline stops NEW tiers from starting; the
// hard cap bounds the tiers already in flight (a CNBC hang + a slow Nasdaq could
// otherwise run a ticker past 30 s, and a Vercel 504 skips serve() entirely). Past the
// cap produce() throws → /tmp → KV → baked, like any other failure.
const DEADLINE_MS = 18000;
const HARD_CAP_MS = 23000;
const KV_SAVE_BY_MS = 25000; // saveFactorsKV aborts after 3 s → done by 28 s
// POLYGON_KEY is shared with spy / spy-daily-move / market-extra / vol / breadth on a
// 5-requests-per-minute free tier. During a CNBC+Nasdaq outage six parallel calls here
// would starve those cards' own fallbacks, so the factor row may spend at most 2 per
// invocation, one reserved for SPY (without SPY nothing is computable).
const POLYGON_BUDGET = 2;

function capped(promise, ms) {
    return new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error(`factor sources over ${ms / 1000}s`)), ms);
        promise.then((v) => { clearTimeout(t); resolve(v); }, (e) => { clearTimeout(t); reject(e); });
    });
}

function bakedSeries(faults) {
    return (t) => {
        trip('fx_baked', faults);
        const rows = baked?.tickers?.[t];
        return Array.isArray(rows) ? rows.map(([date, price]) => ({ date, price })) : null;
    };
}

/** The floor: every ticker from the bake alone. Never throws; empty if the bake is faulted/missing. */
async function bakedPayload(faults, today) {
    try {
        const only = bakedSeries(faults);
        const resolved = {};
        for (const t of TICKERS) {
            resolved[t] = await resolveTicker(t, { recent: [], long: [], baked: only, today });
        }
        const p = buildPayload(resolved, { bakedAt: baked?.bakedAt || null });
        return { ...p, _meta: { ...p._meta, source: `baked snapshot ${baked?.bakedAt || '?'}`, stale: true, hasErrors: true } };
    } catch {
        return { factors: [], windows: [], _meta: { source: 'none', hasErrors: true, stale: true, messages: ['no factor source'] } };
    }
}

export async function GET(request) {
    // Touch the request so Next renders this handler per request (see vol/route.js):
    // without it the route is prerendered at BUILD time and ?_fail= is ignored.
    request.headers.get('user-agent');

    const faults = faultsFrom(request);
    const polygonKey = process.env.POLYGON_KEY;
    const today = todayET();
    const fallback = await bakedPayload(faults, today);

    let produced = null;
    return serve('factors', async () => {
        const started = Date.now();
        const deadline = started + DEADLINE_MS;
        const polygonMay = callBudget(POLYGON_BUDGET, BENCH);
        const polygon = (t) => {
            if (!polygonMay(t)) return Promise.reject(new Error('polygon budget spent (shared 5/min key)'));
            return gate('fx_polygon', faults, () => polygonDaily(t, polygonKey, { years: 2, tries: 1, timeout: 6000, revalidate: 21600 }))
                .then((r) => r.history);
        };
        const nasdaqMemo = new Map();
        const yahooMemo = new Map();
        const nasdaq = (t) => {
            if (!nasdaqMemo.has(t)) nasdaqMemo.set(t, gate('fx_nasdaq', faults, () => nasdaqHistory(t, { years: 10, timeout: 8000 })));
            return nasdaqMemo.get(t);
        };
        const yahoo = (t) => {
            if (!yahooMemo.has(t)) {
                yahooMemo.set(t, gate('fx_yahoo', faults, () => yahooChart(t, { range: '10y', interval: '1d', adjusted: false, tries: 1, timeout: 5000, revalidate: 1800 }))
                    .then((r) => r.history));
            }
            return yahooMemo.get(t);
        };

        const recent = [
            { name: 'cnbc', fn: (t) => gate('fx_cnbc', faults, () => cnbcHistory(t, { range: '1Y', timeout: 6000, tries: 1 })) },
            { name: 'nasdaq', fn: nasdaq },
            { name: 'polygon', fn: polygon },
            { name: 'yahoo', fn: yahoo },
        ];
        const long = [
            { name: 'cnbc-weekly', fn: (t) => gate('fx_cnbcw', faults, () => cnbcHistory(t, { range: '5Y', timeout: 6000, tries: 1 })).then((h) => weeklyToFriday(h, today)) },
            { name: 'nasdaq', fn: nasdaq },
            { name: 'yahoo', fn: yahoo },
        ];

        const results = await capped(Promise.all(TICKERS.map((t) =>
            resolveTicker(t, { recent, long, baked: bakedSeries(faults), today, deadline }))), HARD_CAP_MS);
        const resolved = Object.fromEntries(results.map((r) => [r.ticker, r]));
        const payload = buildPayload(resolved, { bakedAt: baked?.bakedAt || null });
        produced = payload;

        // Durable copy for cold instances. Skipped under fault injection (never let a
        // test write shared state), for anything stale, and when time is short.
        if (faults.size === 0 && isStorablePayload(payload) && Date.now() - started < KV_SAVE_BY_MS) {
            await saveFactorsKV(payload);
        }
        return payload;
    }, {
        faults,
        isGood: isFreshGoodPayload,
        shouldStore: isStorablePayload,
        // The KV copy only wins if it is at least as new as what live just produced.
        lastResort: async () => {
            if (faults.has('fx_kv')) return null;
            const kv = await loadFactorsKV();
            if (kv && produced?.asOf && (kv.asOf || '') < produced.asOf) return null;
            return kv;
        },
        fallback,
    });
}
