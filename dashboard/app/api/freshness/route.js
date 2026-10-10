/**
 * GET /api/freshness — fleet freshness contract v1 (ages in hours only, no data).
 *
 * Reads the two producer -> screen pipelines (/api/rubber-band, /api/history) and the
 * market routes (served:* items, lib/servedFreshness.js) through the SAME public routes the page reads, so edge cache + last-known-good fallbacks are
 * included in what is measured. Pure maths: lib/servedFreshness.js. No auth by design:
 * the body holds only pipeline names and hour counts.
 */
import { buildFreshness, SERVED_ROUTES } from '../../../lib/servedFreshness';

export const dynamic = 'force-dynamic';
export const maxDuration = 30;

const TIMEOUT_MS = 20000;

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
        return Response.json(buildFreshness({ rubberBand, history, served: servedMap }), { status: 200, headers });
    } catch (e) {
        return Response.json({ error: String(e?.message || e).slice(0, 160) }, { status: 500, headers });
    }
}
