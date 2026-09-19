/**
 * Last-known-good store for the Jev pill INPUTS (T10Y3M, NFCI, claims, Sahm).
 *
 * Why a second store: lib/store.js writes /tmp, which on Vercel lives only as
 * long as one warm instance. The pills route already has Upstash KV creds
 * (lib/jevLog.js), so a pill input's last-good goes to BOTH: /tmp for speed,
 * KV (`ftb:jev:lg:<key>`) so a cold instance during a FRED outage still finds
 * it. Every call is best-effort and never throws.
 *
 * KV write budget: a key is rewritten only when its value/asOf changed or the
 * local copy is older than REWRITE_MS, so a healthy day costs a handful of
 * SETs, not one per dashboard load.
 */

import { loadLastGood, saveLastGood } from './store';
import { kvGet, kvSet } from './jevLog';

export const KV_PREFIX = 'ftb:jev:lg:';
export const REWRITE_MS = 6 * 3600e3;

function tooOld(savedAt, maxAgeMs) {
    if (!maxAgeMs || !savedAt) return false;
    return Date.now() - new Date(savedAt).getTime() > maxAgeMs;
}

/**
 * @param {object} [deps] injectable for tests
 * @param {{get:(k:string)=>Promise<any>, set:(k:string,v:any)=>Promise<boolean>}} [deps.kv]
 * @param {{load:(k:string,m?:number)=>any, save:(k:string,d:any)=>void}} [deps.tmp]
 * @returns {{load:(key:string,maxAgeMs?:number)=>Promise<{data:any,savedAt:string}|null>, save:(key:string,data:any)=>Promise<boolean>}}
 */
export function makePillStore(deps = {}) {
    const kv = deps.kv || { get: kvGet, set: kvSet };
    const tmp = deps.tmp || { load: loadLastGood, save: saveLastGood };
    return {
        async load(key, maxAgeMs) {
            try {
                const local = tmp.load(key, maxAgeMs);
                if (local && local.data) return local;
            } catch { /* fall through to KV */ }
            try {
                const remote = await kv.get(KV_PREFIX + key);
                const parsed = typeof remote === 'string' ? JSON.parse(remote) : remote;
                if (!parsed || typeof parsed !== 'object' || !parsed.data) return null;
                if (tooOld(parsed.savedAt, maxAgeMs)) return null;
                try { tmp.save(key, parsed.data); } catch { /* best effort */ }
                return parsed;
            } catch { return null; }
        },
        async save(key, data) {
            let prev = null;
            try { prev = tmp.load(key); } catch { /* ignore */ }
            try { tmp.save(key, data); } catch { /* ignore */ }
            const unchanged = !!(prev && prev.data
                && prev.data.value === data?.value
                && prev.data.asOf === data?.asOf
                && !tooOld(prev.savedAt, REWRITE_MS));
            if (unchanged) return false;
            try {
                return !!(await kv.set(KV_PREFIX + key, { data, savedAt: new Date().toISOString() }));
            } catch { return false; }
        },
    };
}
