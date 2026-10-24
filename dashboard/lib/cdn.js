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
 * - Only a HEALTHY LIVE answer gets the full policy. Degraded (`_meta.stale` /
 *   `hasErrors`), last-known-good, last-resort and fallback answers sit at the edge for
 *   DEGRADED_MAX_AGE only (2 min, no swr): shorter than the page's 5-min refresh, so
 *   the next refresh still retries the live sources, but the same-minute repeats
 *   (jev-pills + /api/freshness reading siblings, two monitors at once) no longer
 *   re-run a slow failing route (Vercel Active CPU cap, 24 Oct 2026).
 * - Price rows (worst case ≤ 10 min) stretch to ≤ 2 h while the US market is not
 *   open, never past the next open (lib/marketClock): nothing on them moves, and the
 *   24/7 30-min /api/freshness probe was re-running every one (24 Oct 2026).
 * - Fault-injection requests (`?_fail=`) are never cached.
 * - The browser never caches (`cache-control: no-store` stays). Only Vercel's edge
 *   does, via `Vercel-CDN-Cache-Control`, which Vercel strips before the response
 *   leaves the edge.
 * - The edge cache key includes the query string, so `?_t=<now>` (the page's manual
 *   refresh button, scripts/health_check.py) always reaches the live route.
 */
import { marketStatus } from './marketClock';

// [max-age, stale-while-revalidate] in seconds. Prices move → short; daily data → long.
// Live rows: max-age just over the page's 5-min market-hours refresh (app/page.js,
// lib/marketClock refreshMs), so the next refresh is an edge HIT, not a function run
// (Vercel Active CPU cap, 21 Oct 2026). Off-hours the page refreshes every 30 min.
export const CDN_POLICY = {
    spy:              [330, 270],   // ≤ 10 min: live price
    'spy-daily-move': [330, 270],   // ≤ 10 min
    'fear-greed':     [330, 270],   // ≤ 10 min: CNN updates every few minutes at most
    sheets:           [330, 270],   // ≤ 10 min: his Google Sheet (also read by the bot's brief)
    'market-extra':   [330, 270],   // ≤ 10 min: the slowest route (~5 s cold)
    vol:              [330, 270],   // ≤ 10 min
    'jev-pills':      [330, 270],   // ≤ 10 min: regime pills, built from sibling routes
    polymarket:       [1800, 1800], // ≤ 1 h: betting odds drift slowly
    'rubber-band':    [3600, 3600], // ≤ 2 h: written nightly by the Mac mini (lib/servedFreshness grace counts it)
    history:          [3600, 3600], // ≤ 2 h: Sheet1 gets rows at 02:00 + 14:00 UTC (freshness maxAgeH 30 h counts it)
    aaii:             [7200, 3600], // ≤ 3 h: weekly survey (the route also caches 3 h per instance)
    breadth:          [7200, 3600], // ≤ 3 h: daily bars
    factors:          [7200, 3600], // ≤ 3 h: daily bars
    fred:             [7200, 3600], // ≤ 3 h: daily/monthly series (the page says "refreshes every 3 h")
    freshness:        [3300, 300],  // ≤ 1 h: monitor probe; only a reply still green an hour on (app/api/freshness)
};

/** Degraded answers: a short edge copy, no swr (see the rules above). */
export const DEGRADED_MAX_AGE = 120;
/** Off-hours max-age ceiling for the price rows (+ their swr ≤ 2 h). */
export const OFF_HOURS_MAX_AGE = 6900;

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
 * adds the edge policy only for a non-test answer from a route that has one (a short
 * one when degraded). `degraded` flags a reason the payload itself doesn't.
 */
export function cacheHeaders(key, { payload, testMode = false, degraded = false, now = Date.now() } = {}) {
    const headers = { 'cache-control': 'no-store' };
    const policy = CDN_POLICY[key];
    if (!policy || testMode) return headers;
    const [base, swr] = policy;
    if (degraded || isDegraded(payload)) {
        headers['vercel-cdn-cache-control'] = `max-age=${Math.min(base, DEGRADED_MAX_AGE)}`;
        return headers;
    }
    const st = marketStatus(now);
    const offHours = base + swr <= 600 && st.state !== 'open';
    const maxAge = offHours ? Math.max(base, Math.min(OFF_HOURS_MAX_AGE, Math.floor(st.ms / 1000))) : base;
    headers['vercel-cdn-cache-control'] = `max-age=${maxAge}, stale-while-revalidate=${swr}`;
    return headers;
}
