/**
 * Durable last-known-good for /api/factors in Upstash KV (`ftb:factors:lg`).
 *
 * Why: serve()'s /tmp copy lives only as long as one warm Vercel instance. A cold
 * instance during a CNBC + Nasdaq + Polygon + Yahoo outage would otherwise drop
 * straight to the baked floor. KV survives cold starts.
 *
 * Own tiny KV client (not lib/jevLog's) on purpose: every call here is
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

export const KV_KEY = 'ftb:factors:lg';
export const REWRITE_MS = 6 * 3600e3;
export const KV_MAX_AGE_MS = 30 * 864e5;
const MARK_KEY = 'factors-kvmark';

async function kvCall(path, init = {}, timeoutMs = 3000) {
    const base = process.env.KV_REST_API_URL || '';
    const token = process.env.KV_REST_API_TOKEN || '';
    if (!base || !token) return null;
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    try {
        const res = await fetch(`${base}${path}`, {
            ...init,
            cache: 'no-store',
            signal: ctl.signal,
            headers: { Authorization: `Bearer ${token}`, ...(init.headers || {}) },
        });
        if (!res.ok) return null;
        const data = await res.json();
        return data && !data.error ? data : null;
    } catch {
        return null;
    } finally {
        clearTimeout(t);
    }
}

export const defaultKv = {
    async get(key) {
        const d = await kvCall(`/get/${encodeURIComponent(key)}`);
        return d ? d.result ?? null : null;
    },
    async set(key, value) {
        const d = await kvCall(`/set/${encodeURIComponent(key)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(value),
        });
        return !!d;
    },
};

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
