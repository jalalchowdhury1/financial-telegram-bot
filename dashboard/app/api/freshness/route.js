/**
 * GET /api/freshness — fleet freshness contract v1 (ages in hours only, no data).
 *
 * Reads the two producer -> screen pipelines (/api/rubber-band, /api/history) and the
 * market routes (served:* items, lib/servedFreshness.js) through the SAME public routes the page reads, so edge cache + last-known-good fallbacks are
 * included in what is measured. Pure maths: lib/servedFreshness.js. No auth by design:
 * the body holds only pipeline names and hour counts.
 */
import { buildFreshness, isRed, SERVED_ROUTES } from '../../../lib/servedFreshness';
import { cacheHeaders } from '../../../lib/cdn';

export const dynamic = 'force-dynamic';
export const maxDuration = 60;

// Must outlast every served route's own budget (/api/fred: FRED_BUDGET_MS 40 s on a cold
// cache), or a slow-but-fine route reads as "served nothing" → red (21 Oct 2026).
export const TIMEOUT_MS = 50000;

async function sibling(origin, path) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
    try {
        const res = await fetch(`${origin}${path}`, { cache: 'no-store', signal: ctl.signal, headers: { 'user-agent': 'freshness-probe/1.0' } });
        if (!res.ok) throw new Error(`${path} HTTP ${res.status}`);
        return await res.json();
    } finally {
        clearTimeout(timer);
    }
}

/**
 * Edge-cache the reply (≤ 1 h, lib/cdn.js `freshness`) only when every item would still
 * be green an hour from now with nothing changing; anything red or close to it is
 * re-measured by the very next probe. The 24/7 30-min fleet probe re-ran every market
 * route each time (Vercel Active CPU cap, 24 Oct 2026).
 */
export function freshnessHeaders(body) {
    const inAnHour = (h) => (h == null ? h : h + 1);
    const greenLater = body.items.every((i) => !isRed(i)
        && !isRed({ ...i, inputAgeH: inAnHour(i.inputAgeH), servedAgeH: inAnHour(i.servedAgeH) }));
    return cacheHeaders('freshness', { degraded: !greenLater });
}

export async function GET(request) {
    request.headers.get('user-agent');
    const headers = { 'cache-control': 'no-store' };
    try {
        const origin = new URL(request.url).origin;
        // A market route that errors is graded (null → red), it does not 500 the whole probe.
        const soft = (p) => sibling(origin, p).catch(() => null);
        const [rubberBand, history, ...served] = await Promise.all([
            sibling(origin, '/api/rubber-band'), sibling(origin, '/api/history'),
            ...SERVED_ROUTES.map((r) => soft(`/api/${r}`)),
        ]);
        const servedMap = Object.fromEntries(SERVED_ROUTES.map((r, i) => [r, served[i]]));
        const body = buildFreshness({ rubberBand, history, served: servedMap });
        return Response.json(body, { status: 200, headers: freshnessHeaders(body) });
    } catch (e) {
        return Response.json({ error: String(e?.message || e).slice(0, 160) }, { status: 500, headers });
    }
}
