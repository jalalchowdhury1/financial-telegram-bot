/**
 * Durable "last-known-good" store + a never-throws route wrapper.
 *
 * Goal: a dashboard data route can NEVER crash, 500, or return garbage. The
 * worst case is serving the last value we ever successfully fetched (flagged
 * stale), and if even that is absent, a caller-provided safe default.
 *
 * Storage is best-effort and layered, each wrapped so it can never throw:
 *   1. /tmp file (survives for the life of a warm serverless instance)
 *   2. (optional) Redis via REDIS_URL — only if a client is available; we never
 *      hard-depend on it, so a missing/!broken Redis cannot break a request.
 * Reads/writes are always guarded; any failure degrades silently.
 */

import fs from 'fs';

const tmpPath = (key) => `/tmp/lg-${key.replace(/[^a-z0-9_-]/gi, '_')}.json`;

export function saveLastGood(key, data) {
    try {
        fs.writeFileSync(tmpPath(key), JSON.stringify({ data, savedAt: new Date().toISOString() }));
    } catch { /* ignore — storage is best-effort */ }
}

/**
 * @param {string} key
 * @param {number} [maxAgeMs] if set, ignore copies older than this
 * @returns {{data:any, savedAt:string}|null}
 */
export function loadLastGood(key, maxAgeMs) {
    try {
        const p = tmpPath(key);
        if (!fs.existsSync(p)) return null;
        const parsed = JSON.parse(fs.readFileSync(p, 'utf8'));
        if (maxAgeMs && parsed.savedAt && Date.now() - new Date(parsed.savedAt).getTime() > maxAgeMs) return null;
        return parsed;
    } catch { return null; }
}

const json = (body, status = 200) => Response.json(body, { status, headers: { 'cache-control': 'no-store' } });

/**
 * Wrap a route producer so it can never throw.
 *
 * @param {string} key                  last-good cache key (per route)
 * @param {() => Promise<any>} produce   returns the live/fallback payload, or throws
 * @param {object} [opts]
 * @param {(payload:any)=>boolean} [opts.isGood]  payload is servable as-is (default: truthy)
 * @param {(payload:any)=>boolean} [opts.shouldStore]  payload is worth SAVING as the new
 *        last-known-good (default: same as isGood). These differ when a route can build a
 *        payload that is better than the cache but partly derived FROM it — /api/fred does
 *        this when live FRED is dead but the Four Horsemen resolved from Treasury/BLS. Such
 *        a payload must be served, but storing it would refresh the cache's `savedAt` on
 *        largely stale content and let it outlive the 7-day window indefinitely.
 * @param {any} [opts.fallback]          safe default if there is no last-good either
 * @param {number} [opts.maxStaleMs]     max age of a last-good copy to serve (default 7 days)
 */
export async function serve(key, produce, opts = {}) {
    const { isGood = (x) => !!x, fallback = null, maxStaleMs = 7 * 864e5, faults = null, lastResort = null } = opts;
    const shouldStore = opts.shouldStore || isGood;
    // In fault-injection test mode we never write last-good (don't pollute real
    // data) and `?_fail=lastgood` also disables READING it (to reach the default).
    const testMode = faults && faults.size > 0;
    const readLG = (testMode && faults.has('lastgood')) ? () => null : (k, m) => loadLastGood(k, m);
    const writeLG = testMode ? () => {} : saveLastGood;
    // Last-resort durable fallback (e.g. the Google-Sheet helper tab), tried ONLY
    // after both live + /tmp last-good are unavailable. Never-throws; `?_fail=sheetlkg`
    // disables it so a fault test can reach the bare default.
    const runLastResort = (lastResort && !(testMode && faults.has('sheetlkg')))
        ? async () => { try { return await lastResort(); } catch { return null; } }
        : async () => null;
    try {
        const payload = await produce();
        if (isGood(payload)) {
            if (shouldStore(payload)) writeLG(key, payload);
            return json(payload);
        }
        const lg = readLG(key, maxStaleMs);
        if (lg) return json(withStale(lg, 'produced empty; serving last-known-good'));
        const lr = await runLastResort();
        if (lr) return json(asStale(lr, 'live produced empty + no cache; serving last-resort fallback'));
        return json(payload ?? fallback);
    } catch (e) {
        const lg = readLG(key, maxStaleMs);
        if (lg) return json(withStale(lg, `live sources failed (${String(e?.message).slice(0, 120)}); serving last-known-good`));
        const lr = await runLastResort();
        if (lr) return json(asStale(lr, `live sources failed (${String(e?.message).slice(0, 120)}); serving last-resort fallback`));
        return json(addMeta(fallback, { source: 'Unavailable', hasErrors: true, messages: [String(e?.message).slice(0, 160)] }));
    }
}

// Stamp a last-resort payload as stale (keeping its own _meta.source) so the UI and
// health check both treat it as degraded data, not fresh.
function asStale(payload, message) {
    const meta = (payload && payload._meta) || {};
    return {
        ...payload,
        _meta: { ...meta, stale: true, hasErrors: true, messages: [...(meta.messages || []), message] },
    };
}

function withStale(lg, message) {
    const d = lg.data || {};
    return addMeta(d, {
        ...(d._meta || {}),
        source: `${d._meta?.source || 'cache'} (last-known-good ${lg.savedAt})`,
        hasErrors: true,
        stale: true,
        lastGoodAt: lg.savedAt,
        messages: [...(d._meta?.messages || []), message],
    });
}

function addMeta(obj, meta) {
    if (obj && typeof obj === 'object' && !Array.isArray(obj)) return { ...obj, _meta: meta };
    return { data: obj, _meta: meta };
}
