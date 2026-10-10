/**
 * Durable last-known-good for /api/factors in Upstash KV (`ftb:factors:lg`).
 *
 * Why: serve()'s /tmp copy lives only as long as one warm Vercel instance. A cold
 * instance during a CNBC + Nasdaq + Polygon + Yahoo outage would otherwise drop
 * straight to the baked floor. KV survives cold starts.
 *
 * KV client = lib/kv.js (not lib/jevLog's) on purpose: every call there is
 * `cache: 'no-store'` with a hard timeout. The route sets fetchCache
 * 'default-cache', under which a bare fetch() would be served from Next's Data
 * Cache — a last-good READ must never be a cached read.
 *
 * Write budget: rewrite only when the payload's asOf changed or the last write
 * is older than REWRITE_MS (tracked by a /tmp marker), so a healthy day costs a
 * few SETs per warm instance, not one per page load. Nothing here ever throws.
 */
import { loadLastGood, saveLastGood } from './store';
import { isGoodPayload } from './factors';
import { defaultKv } from './kv';

export const KV_KEY = 'ftb:factors:lg';
export const REWRITE_MS = 6 * 3600e3;
export const KV_MAX_AGE_MS = 30 * 864e5;
const MARK_KEY = 'factors-kvmark';

// The KV client itself lives in lib/kv.js (shared with serve() and the legacy routes);
// re-exported here because /api/sheets + lib/aaii callers import it from this module.
export { defaultKv };

/**
 * @returns {Promise<object|null>} the stored payload relabelled as a KV copy, or null.
 */
export async function loadFactorsKV({ kv = defaultKv, now = Date.now() } = {}) {
    try {
        const raw = await kv.get(KV_KEY);
        const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
        const data = parsed?.data;
        if (!isGoodPayload(data)) return null;
        if (parsed.savedAt && now - Date.parse(parsed.savedAt) > KV_MAX_AGE_MS) return null;
        const meta = data._meta || {};
        // Relabel provenance: a cached payload must never read as live.
        return {
            ...data,
            _meta: {
                ...meta,
                source: `KV last-good (${parsed.savedAt || 'unknown'}) ← ${meta.source || 'unknown'}`,
                lastGoodAt: parsed.savedAt || null,
                messages: (meta.messages || []).map((m) => `cached: ${m}`),
            },
        };
    } catch {
        return null;
    }
}

/** @returns {Promise<boolean>} true when a KV write happened and succeeded. */
export async function saveFactorsKV(payload, { kv = defaultKv, tmp = { load: loadLastGood, save: saveLastGood }, now = Date.now() } = {}) {
    try {
        if (!isGoodPayload(payload) || payload._meta?.stale) return false;
        let mark = null;
        try { mark = tmp.load(MARK_KEY)?.data || null; } catch { /* ignore */ }
        if (mark && mark.asOf === payload.asOf && now - Date.parse(mark.at) < REWRITE_MS) return false;
        const at = new Date(now).toISOString();
        const ok = await kv.set(KV_KEY, { data: payload, savedAt: at });
        if (ok) { try { tmp.save(MARK_KEY, { asOf: payload.asOf, at }); } catch { /* ignore */ } }
        return !!ok;
    } catch {
        return false;
    }
}
