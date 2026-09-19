/**
 * jevLog — best-effort Upstash KV persistence for daily Jev pills verdicts.
 *
 * Keys:
 *   ftb:jev:<YYYY-MM-DD>    — JSON payload for that day
 *   ftb:jev:days            — LPUSH list of date strings (newest first)
 *   ftb:jev:seen:<date>     — NX guard with 40-day TTL
 *
 * Every function returns null/false on any failure (no KV env vars, network
 * error, parse error). Never throws.
 *
 * Upstash REST protocol:
 *   GET /get/<key>         — returns { result: <stored-value>, error: null }
 *   POST /set/<key>        — body is the RAW value (JSON.stringify(value))
 *   POST /set/<key>/<val>/EX/<ttl>/NX  — path-form NX guard, no body; result === 'OK' means claimed
 *   POST /lpush/<key>/<val> — path-form LPUSH, no body
 */

const BASE = () => process.env.KV_REST_API_URL || '';
const TOKEN = () => process.env.KV_REST_API_TOKEN || '';

function canLog() {
    return !!(BASE() && TOKEN());
}

/**
 * GET /get/<key> — returns the decoded value or null.
 */
async function kvGet(key) {
    if (!canLog()) return null;
    try {
        const url = `${BASE()}/get/${encodeURIComponent(key)}`;
        const res = await fetch(url, {
            headers: { Authorization: `Bearer ${TOKEN()}` },
        });
        if (!res.ok) return null;
        const data = await res.json();
        // Upstash returns { result: <value>, error: null }
        if (data.error) return null;
        return data.result ?? null;
    } catch {
        return null;
    }
}

/**
 * POST /set/<key> — body is JSON.stringify(value) (the RAW value, no wrapper).
 * Returns true on success, false on failure.
 */
async function kvSet(key, value) {
    if (!canLog()) return false;
    try {
        const url = `${BASE()}/set/${encodeURIComponent(key)}`;
        const res = await fetch(url, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${TOKEN()}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(value),
        });
        if (!res.ok) return false;
        const data = await res.json();
        return !data.error;
    } catch {
        return false;
    }
}

/**
 * POST /set/<key>/<value>/EX/<ttl>/NX — path-form NX guard (no body).
 * Returns true when the key was claimed (result === 'OK'), null when it
 * already exists (result === null), false on any error (HTTP/network).
 */
async function kvSetNx(key, value, ttl) {
    if (!canLog()) return false;
    try {
        const url = `${BASE()}/set/${encodeURIComponent(key)}/${encodeURIComponent(String(value))}/EX/${ttl}/NX`;
        const res = await fetch(url, {
            method: 'POST',
            headers: { Authorization: `Bearer ${TOKEN()}` },
        });
        if (!res.ok) return false;
        const data = await res.json();
        if (data.error) return false;
        // Upstash returns result: 'OK' when claimed, result: null when already exists
        if (data.result === 'OK') return true;
        return null;
    } catch {
        return false;
    }
}

/**
 * POST /lpush/<key>/<element> — path-form LPUSH (no body).
 * Returns true on success, false on failure.
 */
async function kvLpush(key, element) {
    if (!canLog()) return false;
    try {
        const url = `${BASE()}/lpush/${encodeURIComponent(key)}/${encodeURIComponent(String(element))}`;
        const res = await fetch(url, {
            method: 'POST',
            headers: { Authorization: `Bearer ${TOKEN()}` },
        });
        if (!res.ok) return false;
        const data = await res.json();
        return !data.error;
    } catch {
        return false;
    }
}

/**
 * GET /lrange/<key>/0/-1 — returns array of elements or null.
 */
async function kvLrange(key) {
    if (!canLog()) return null;
    try {
        const url = `${BASE()}/lrange/${encodeURIComponent(key)}/0/-1`;
        const res = await fetch(url, {
            headers: { Authorization: `Bearer ${TOKEN()}` },
        });
        if (!res.ok) return null;
        const data = await res.json();
        if (data.error) return null;
        return Array.isArray(data.result) ? data.result : null;
    } catch {
        return null;
    }
}

/**
 * Read a day's stored verdict payload.
 * @param {string} dateStr — 'YYYY-MM-DD'
 * @returns {object|null} the payload or null
 */
export async function readDay(dateStr) {
    const key = `ftb:jev:${dateStr}`;
    const raw = await kvGet(key);
    if (raw == null) return null;
    // Upstash get returns the stored value as a string (we store JSON.stringify'd objects)
    // JSON.parse to hydrate it back
    if (typeof raw === 'string') {
        try { return JSON.parse(raw); } catch { return null; }
    }
    // Already an object (legacy or other format)
    if (raw && typeof raw === 'object') return raw;
    return null;
}

/**
 * Get the newest logged day's data before today.
 * @returns {object|null} payload with `.date` or null
 */
export async function yesterday() {
    try {
        const days = await kvLrange('ftb:jev:days');
        if (!days || !days.length) return null;
        const todayStr = new Date().toISOString().slice(0, 10);
        const sorted = [...days].sort().reverse();
        for (const d of sorted) {
            if (d < todayStr) {
                const payload = await readDay(d);
                if (payload) {
                    return { ...payload, date: d };
                }
            }
        }
        return null;
    } catch {
        return null;
    }
}

/**
 * List all logged date strings (newest first).
 * @returns {string[]|null}
 */
export async function listDays() {
    try {
        const days = await kvLrange('ftb:jev:days');
        if (!days) return null;
        return [...days].sort().reverse();
    } catch {
        return null;
    }
}

/**
 * Log today's verdict payload — writes at most once per day.
 *
 * 1. Claim the NX guard FIRST. If already claimed (already logged today),
 *    return true without any further writes.
 * 2. If claimed, SET ftb:jev:<date> to the payload, then LPUSH the date,
 *    and return the SET result.
 *
 * @param {string} dateStr — 'YYYY-MM-DD'
 * @param {object} payload — verdicts + state hash + spy price + pills
 * @returns {boolean} true if logged or already logged, false on failure
 */
export async function logVerdicts(dateStr, payload) {
    if (!canLog()) return false;
    try {
        // Step 1: claim the NX guard (null = already exists, false = error)
        const seenKey = `ftb:jev:seen:${dateStr}`;
        const claimed = await kvSetNx(seenKey, '1', 3456000);
        if (claimed === null) {
            // Already logged today — no further writes
            return true;
        }
        if (!claimed) {
            // KV server error — log failed
            return false;
        }

        // Step 2: claimed — write the payload and push date
        const dateKey = `ftb:jev:${dateStr}`;
        const stored = await kvSet(dateKey, payload);
        if (stored) {
            await kvLpush('ftb:jev:days', dateStr);
        }
        return stored;
    } catch {
        return false;
    }
}

/**
 * Check if KV is configured.
 * @returns {boolean}
 */
export function isConfigured() {
    return canLog();
}