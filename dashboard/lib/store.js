/**
 * Durable "last-known-good" store + a never-throws route wrapper.
 *
 * Goal: a dashboard data route can NEVER crash, 500, or return garbage. The
 * worst case is serving the last value we ever successfully fetched (flagged
 * stale), and if even that is absent, a caller-provided safe default.
 *
 * Storage is best-effort and layered, each wrapped so it can never throw:
 *   1. /tmp file (survives for the life of a warm serverless instance)
 *   2. Upstash KV `ftb:lg:<key>` (lib/kv.js; survives cold starts). Read only after
 *      the /tmp copy misses; written on a healthy answer, throttled (see saveLastGoodKV).
 *      No KV env → silently skipped, so a missing/broken KV cannot break a request.
 * Reads/writes are always guarded; any failure degrades silently.
 */

import fs from 'fs';
import crypto from 'crypto';
import { cacheHeaders } from './cdn';
import { defaultKv, parseEnvelope } from './kv';

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

// ---- Durable KV last-known-good (`ftb:lg:<key>`) --------------------------------------
export const kvKeyFor = (key) => `ftb:lg:${key}`;
export const KV_REWRITE_MS = 30 * 60e3;   // unchanged payload: refresh the KV copy at most every 30 min
export const KV_MIN_GAP_MS = 60e3;        // changed payload: still at most one SET a minute per key per instance
export const KV_MAX_BYTES = 900 * 1024;   // Upstash caps a value at 1 MB; never try to write near it

const markKey = (key) => `kvmark-${key}`;
const hashOf = (str) => crypto.createHash('sha1').update(str).digest('hex');

/**
 * Write a healthy payload to KV — throttled by a /tmp marker {hash, at}: only when its
 * JSON changed since the last KV write (and ≥ KV_MIN_GAP_MS passed), or the last write
 * is older than KV_REWRITE_MS. Never writes a payload already flagged stale (a durable
 * copy must only ever hold real live data) or one over KV_MAX_BYTES.
 * @returns {Promise<boolean>} true when a KV write happened and succeeded. Never throws.
 */
export async function saveLastGoodKV(key, data, { kv = defaultKv, now = Date.now() } = {}) {
    try {
        if (!data || data?._meta?.stale) return false;
        const body = JSON.stringify(data);
        if (body.length > KV_MAX_BYTES) return false;
        const hash = hashOf(body);
        const mark = loadLastGood(markKey(key))?.data || null;
        const age = mark ? now - Date.parse(mark.at) : Infinity;
        if (mark && (mark.hash === hash ? age < KV_REWRITE_MS : age < KV_MIN_GAP_MS)) return false;
        const at = new Date(now).toISOString();
        const ok = await kv.set(kvKeyFor(key), { data, savedAt: at });
        if (ok) saveLastGood(markKey(key), { hash, at });
        return !!ok;
    } catch { return false; }
}

/**
 * @returns {Promise<{data:any, savedAt:string}|null>} the KV copy (same shape as
 * loadLastGood), or null when absent / older than maxAgeMs / KV unreachable. Never throws.
 */
export async function loadLastGoodKV(key, maxAgeMs, { kv = defaultKv, now = Date.now() } = {}) {
    try {
        const env = parseEnvelope(await kv.get(kvKeyFor(key)));
        if (!env || env.data == null || !env.savedAt) return null;
        if (maxAgeMs && now - Date.parse(env.savedAt) > maxAgeMs) return null;
        return { data: env.data, savedAt: env.savedAt };
    } catch { return null; }
}

// Every answer is `no-store` for the browser. Only serve()'s healthy live path adds an
// edge-cache policy (lib/cdn.js) — degraded / cached / fallback answers never do.
const json = (body, status = 200, headers = { 'cache-control': 'no-store' }) => Response.json(body, { status, headers });

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
    const { isGood = (x) => !!x, fallback = null, maxStaleMs = 7 * 864e5, faults = null, lastResort = null, kv = defaultKv } = opts;
    const shouldStore = opts.shouldStore || isGood;
    // In fault-injection test mode we never write last-good (/tmp OR KV — don't pollute
    // real data) and `?_fail=lastgood` also disables READING both (to reach the default).
    // `?_fail=kvlg` disables only the KV read, so a test can prove the KV tier alone.
    const testMode = faults && faults.size > 0;
    const readLG = (testMode && faults.has('lastgood')) ? () => null : (k, m) => loadLastGood(k, m);
    const readKV = (testMode && (faults.has('lastgood') || faults.has('kvlg')))
        ? async () => null
        : (k, m) => loadLastGoodKV(k, m, { kv });
    const writeLG = testMode ? async () => {} : async (k, p) => {
        saveLastGood(k, p);
        await saveLastGoodKV(k, p, { kv });
    };
    // Last-resort durable fallback (e.g. the Google-Sheet helper tab), tried ONLY
    // after live + /tmp + KV last-good are all unavailable. Never-throws; `?_fail=sheetlkg`
    // disables it so a fault test can reach the bare default.
    const runLastResort = (lastResort && !(testMode && faults.has('sheetlkg')))
        ? async () => { try { return await lastResort(); } catch { return null; } }
        : async () => null;
    // Order after live: /tmp → KV → lastResort. Returns a Response, or null if all miss.
    const degraded = async (why) => {
        const lg = readLG(key, maxStaleMs);
        if (lg) return json(withStale(lg, `${why}; serving last-known-good`));
        const kvlg = await readKV(key, maxStaleMs);
        if (kvlg) return json(withStaleKV(kvlg, `${why}; /tmp copy missing; serving KV last-known-good`));
        const lr = await runLastResort();
        if (lr) return json(asStale(lr, `${why} + no cache; serving last-resort fallback`));
        return null;
    };
    try {
        const payload = await produce();
        if (isGood(payload)) {
            if (shouldStore(payload)) await writeLG(key, payload);
            return json(payload, 200, cacheHeaders(key, { payload, testMode }));
        }
        return (await degraded('live produced empty')) || json(payload ?? fallback);
    } catch (e) {
        const why = `live sources failed (${String(e?.message).slice(0, 120)})`;
        try {
            const res = await degraded(why);
            if (res) return res;
        } catch { /* fall through to the safe default */ }
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

// A KV copy is relabelled so it can never read as live: `KV last-good (<savedAt>) ← <orig>`.
function withStaleKV(lg, message) {
    const d = lg.data || {};
    return addMeta(d, {
        ...(d._meta || {}),
        source: `KV last-good (${lg.savedAt}) ← ${d._meta?.source || 'unknown'}`,
        hasErrors: true,
        stale: true,
        lastGoodAt: lg.savedAt,
        messages: [...(d._meta?.messages || []).map((m) => `cached: ${m}`), message],
    });
}

function addMeta(obj, meta) {
    if (obj && typeof obj === 'object' && !Array.isArray(obj)) return { ...obj, _meta: meta };
    return { data: obj, _meta: meta };
}
