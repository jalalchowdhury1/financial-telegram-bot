/**
 * GET /api/freshness — fleet freshness contract v1 (ages in hours only, no data).
 *
 * Reads the two producer -> screen pipelines through the SAME public routes the page
 * reads (/api/rubber-band, /api/history), so edge cache + last-known-good fallbacks are
 * included in what is measured. Pure maths: lib/servedFreshness.js. No auth by design:
 * the body holds only pipeline names and hour counts.
 */
import { buildFreshness } from '../../../lib/servedFreshness';

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
        const [rubberBand, history] = await Promise.all([sibling(origin, '/api/rubber-band'), sibling(origin, '/api/history')]);
        return Response.json(buildFreshness({ rubberBand, history }), { status: 200, headers });
    } catch (e) {
        return Response.json({ error: String(e?.message || e).slice(0, 160) }, { status: 500, headers });
    }
}
