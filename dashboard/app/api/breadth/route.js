/**
 * /api/breadth — ETF breadth ratios (RSP/SPY, IWM/SPY, XLK/XLU, HYG/LQD).
 *
 * Dashboard-only (no Lambda hop). Never-throw via serve(). Per-ticker cascade:
 *   Polygon daily aggs (POLYGON_KEY; free tier is a day behind, acceptable
 *     for breadth ratios that measure multi-day trends)
 *   → CNBC harmony daily bars (keyless, '1Y' range — verified datacenter-
 *     reachable, like the vol table and copper/gold).
 *
 * A pair whose ticker histories have no overlapping dates (or one leg is
 * completely unavailable) is omitted from the payload — the rest still ship.
 *
 * Fault gates: breadth_polygon, breadth_cnbc.
 */
import { polygonDaily, cnbcHistory } from '../../../lib/sources';
import { serve } from '../../../lib/store';
import { faultsFrom, gate } from '../../../lib/faults';
import { PAIRS, ratioSeries, pairStats } from '../../../lib/breadth';

export const fetchCache = 'default-cache';

/**
 * Fetch one ticker's daily history via the cascade:
 *   1. Polygon daily aggs (requires POLYGON_KEY)
 *   2. CNBC harmony daily bars (keyless)
 * Returns { history: [{date, price}], source: 'polygon'|'cnbc' } or
 * { history: null, source: 'none' } if all tiers fail.
 */
async function fetchTickerHistory(ticker, polygonKey, faults) {
    // Tier 1: Polygon
    if (polygonKey) {
        try {
            const result = await gate('breadth_polygon', faults, () =>
                polygonDaily(ticker, polygonKey, { years: 1 }),
            );
            const hist = result.history;
            if (Array.isArray(hist) && hist.length > 0) return { history: hist, source: 'polygon' };
        } catch {
            // fall through
        }
    }

    // Tier 2: CNBC harmony
    try {
        const hist = await gate('breadth_cnbc', faults, () =>
            cnbcHistory(ticker, { range: '1Y' }),
        );
        if (Array.isArray(hist) && hist.length > 0) return { history: hist, source: 'cnbc' };
    } catch {
        // fall through
    }

    return { history: null, source: 'none' };
}

export async function GET(request) {
    // Touch the request so Next renders this handler dynamically (per request),
    // while fetches still come from the Data Cache. Without this the route is
    // STATICALLY PRERENDERED at build time — see vol/route.js commentary.
    request.headers.get('user-agent');

    const faults = faultsFrom(request);
    const polygonKey = process.env.POLYGON_KEY;

    return serve('breadth', async () => {
        const messages = [];
        const pairs = {};

        // Fetch histories for every unique ticker across all pairs
        const allTickers = [...new Set(Object.values(PAIRS).flat())];
        const histResults = {};
        await Promise.all(allTickers.map(async (ticker) => {
            const result = await fetchTickerHistory(ticker, polygonKey, faults);
            histResults[ticker] = result;
        }));

        // Build per-ticker source labels
        const tickerSources = [];
        for (const ticker of allTickers) {
            if (histResults[ticker] && histResults[ticker].source && histResults[ticker].source !== 'none') {
                tickerSources.push(`${ticker}:${histResults[ticker].source}`);
            }
        }

        // Build each pair
        for (const [key, [tickerA, tickerB]] of Object.entries(PAIRS)) {
            const histA = histResults[tickerA] ? histResults[tickerA].history : null;
            const histB = histResults[tickerB] ? histResults[tickerB].history : null;

            if (!histA || !histB) {
                messages.push(`${key}: skipped (${tickerA}: ${histA ? 'ok' : 'unavailable'}, ${tickerB}: ${histB ? 'ok' : 'unavailable'})`);
                continue;
            }

            const series = ratioSeries(histA, histB);
            if (!series.length) {
                messages.push(`${key}: no overlapping dates`);
                continue;
            }

            pairs[key] = pairStats(series);
        }

        // updated_at = most recent asOf across all computed pairs
        let updated_at = null;
        for (const stats of Object.values(pairs)) {
            if (stats.asOf && (!updated_at || stats.asOf > updated_at)) {
                updated_at = stats.asOf;
            }
        }

        const expectedPairCount = Object.keys(PAIRS).length;

        return {
            updated_at,
            pairs,
            _meta: {
                source: tickerSources.length ? tickerSources.join(' · ') : 'none',
                hasErrors: Object.keys(pairs).length < expectedPairCount,
                messages,
            },
        };
    }, {
        faults,
        isGood: (p) => !!p && typeof p === 'object' && Object.keys(p.pairs || {}).length > 0,
        fallback: { pairs: {}, _meta: { source: 'none', hasErrors: true, messages: ['no breadth source'] } },
    });
}