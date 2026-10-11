/**
 * cdn.js — the edge (Vercel CDN) cache policy for every GET data route.
 *
 * Why: every /api response used to be `no-store`, so every page load ran every
 * route cold (market-extra alone ~5 s) even when the same answer had been built
 * seconds earlier. A healthy answer now sits at Vercel's edge for a short time:
 *
 *   max-age                  serve the cached answer as-is (edge HIT, ~50 ms)
 *   stale-while-revalidate   past max-age, still answer instantly from the edge
 *                            and rebuild in the background (edge STALE)
 *
 * So the oldest answer anyone can see is max-age + swr (the last column below).
 *
 * The rules that keep this honest:
 * - Only a HEALTHY LIVE answer is cached. Degraded (`_meta.stale` / `hasErrors`),
 *   last-known-good, last-resort and fallback answers are never cached, so the very
 *   next request retries the live sources.
 * - Fault-injection requests (`?_fail=`) are never cached.
 * - The browser never caches (`cache-control: no-store` stays). Only Vercel's edge
 *   does, via `Vercel-CDN-Cache-Control`, which Vercel strips before the response
 *   leaves the edge.
 * - The edge cache key includes the query string, so `?_t=<now>` (the page's manual
 *   refresh button, scripts/health_check.py) always reaches the live route.
 */

// [max-age, stale-while-revalidate] in seconds. Prices move → short; daily data → long.
export const CDN_POLICY = {
    spy:              [60, 240],    // ≤ 5 min: live price
    'spy-daily-move': [60, 240],    // ≤ 5 min
    'fear-greed':     [120, 480],   // ≤ 10 min: CNN updates every few minutes at most
    sheets:           [120, 480],   // ≤ 10 min: his Google Sheet (also read by the bot's brief)
    'market-extra':   [120, 480],   // ≤ 10 min: the slowest route (~5 s cold)
    vol:              [120, 480],   // ≤ 10 min
    'jev-pills':      [120, 480],   // ≤ 10 min: regime pills, built from sibling routes
    polymarket:       [300, 900],   // ≤ 20 min: betting odds drift slowly
    'rubber-band':    [300, 1500],  // ≤ 30 min: written nightly by the Mac mini
    aaii:             [900, 2700],  // ≤ 1 h: weekly survey (the route also caches 3 h per instance)
    breadth:          [900, 2700],  // ≤ 1 h: daily bars
    factors:          [900, 2700],  // ≤ 1 h: daily bars
    history:          [900, 2700],  // ≤ 1 h: one row per day
    fred:             [900, 2700],  // ≤ 1 h: daily/monthly series (the page says "refreshes every 30 min")
};

// The Lambda-primary routes (spy, spy-daily-move, market-extra, polymarket) answer from
// direct sources with hasErrors:false ON PURPOSE (a full direct build is good data) and
// mark it only in the source label: "Finnhub (fallback)". scripts/health_check.py reads
// the same marker (FALLBACK_SOURCE_MARKER).
const FALLBACK_SOURCE_MARKER = '(fallback)';

/** True when a payload reports itself as degraded — such answers are never cached. */
export function isDegraded(payload) {
    if (!payload || typeof payload !== 'object') return false;
    const m = payload._meta && typeof payload._meta === 'object' ? payload._meta : null;
    if (m && (m.stale || m.hasErrors || m.fallback)) return true;
    // spy-daily-move + polymarket carry a top-level `source` (polymarket's _meta has none).
    const source = m?.source ?? payload.source;
    return typeof source === 'string' && source.toLowerCase().includes(FALLBACK_SOURCE_MARKER);
}

/**
 * Response headers for one answer. Always `cache-control: no-store` for the browser;
 * adds the edge policy only for a healthy, non-test answer from a route that has one.
 * `degraded` lets a route veto caching for a reason its payload doesn't flag.
 */
export function cacheHeaders(key, { payload, testMode = false, degraded = false } = {}) {
    const headers = { 'cache-control': 'no-store' };
    const policy = CDN_POLICY[key];
    if (!policy || testMode || degraded || isDegraded(payload)) return headers;
    const [maxAge, swr] = policy;
    headers['vercel-cdn-cache-control'] = `max-age=${maxAge}, stale-while-revalidate=${swr}`;
    return headers;
}
