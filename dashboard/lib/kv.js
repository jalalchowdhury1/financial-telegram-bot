/**
 * Tiny shared Upstash KV (REST) client for the durable last-known-good tiers.
 *
 * Extracted from lib/factorStore.js so serve() (lib/store.js), /api/sheets and
 * /api/fear-greed share ONE client with the same guarantees:
 *   - every call is `cache: 'no-store'` — routes like /api/factors set fetchCache
 *     'default-cache', under which a bare fetch() is answered from Next's Data
 *     Cache; a last-good READ must never itself be a cached read;
 *   - a hard 3 s timeout, so a slow KV can't hold a request hostage;
 *   - it NEVER throws: no env (KV_REST_API_URL / KV_REST_API_TOKEN unset, e.g.
 *     local dev and Jest) → every call silently returns null / false.
 */

export const KV_TIMEOUT_MS = 3000;

export async function kvCall(path, init = {}, timeoutMs = KV_TIMEOUT_MS) {
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

/** Parse a stored `{data, savedAt}` envelope (Upstash hands back a JSON string). Never throws. */
export function parseEnvelope(raw) {
    try {
        const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
        return parsed && typeof parsed === 'object' ? parsed : null;
    } catch {
        return null;
    }
}
