/**
 * snapshot.js — ⚡ instant open. Each feed's last good answer is kept on THIS device
 * (localStorage), so the next visit paints real numbers in ~0.2 s instead of skeletons,
 * then swaps to live ones as each feed lands.
 *
 * A saved copy must never pass for live data: every card showing one carries a
 * "🕐 10:42" tag (see `.card[data-cached]` in globals.css) until its live answer
 * arrives, and fresh-print marks stay off while any of their inputs is saved.
 *
 * Storage can be blocked (private mode, full quota, Safari ITP) — every access is
 * wrapped, and a failure just means the page loads the old way, from skeletons.
 * ~1 MB in all (/api/fred is most of it), well under the ~5 MB localStorage quota.
 */
const SNAP_ROOT = 'fd:snap:';
/**
 * Keyed to the deploy: a payload's shape can change between deploys, and an old-shape copy
 * rendered by new code could throw. The first open after a deploy loads from skeletons,
 * once. NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA is inlined at build time by Vercel; 'dev' locally.
 */
const BUILD = (process.env.NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA || 'dev').slice(0, 7);
export const SNAP_PREFIX = `${SNAP_ROOT}v1:${BUILD}:`;
/** Older than this, skeletons are more honest than a saved copy. */
export const SNAP_MAX_AGE_MS = 3 * 24 * 3600 * 1000;

function storage() {
    try { return typeof window !== 'undefined' ? window.localStorage : null; } catch { return null; }
}

/** @returns {{savedAt:number, data:any}|null} */
export function readSnap(key, now = Date.now(), store = storage()) {
    try {
        const raw = store?.getItem(SNAP_PREFIX + key);
        if (!raw) return null;
        const s = JSON.parse(raw);
        if (!s || !Number.isFinite(s.savedAt) || s.data == null) return null;
        // too old, or stamped in the future (a clock change) — do not show it
        if (now - s.savedAt > SNAP_MAX_AGE_MS || s.savedAt - now > 60e3) return null;
        return s;
    } catch { return null; }
}

/** Save a feed's answer. Error payloads are never saved. Returns true on success. */
export function writeSnap(key, data, now = Date.now(), store = storage()) {
    if (data == null || typeof data !== 'object' || data.error) return false;
    try {
        store.setItem(SNAP_PREFIX + key, JSON.stringify({ savedAt: now, data }));
        return true;
    } catch { return false; }
}

const hasKeys = (o) => !!o && typeof o === 'object' && Object.keys(o).length > 0;
const real = (v) => typeof v === 'string' ? v.trim() !== '' && v !== 'N/A' : v != null;
/**
 * Is this live answer real data? Every route answers 200 even when it fails, with its own
 * fallback body — often no `error` field ({fx:{}…}, tickers:[], value:null, all 'N/A').
 * Such an answer must never replace a saved copy on screen, nor overwrite it in storage.
 * Unknown feeds fall back to "no `error` field".
 */
const LIVE_TESTS = {
    spy: (d) => Number.isFinite(d.current),
    sheets: (d) => ['NotSoBoring', 'FrontRunner', 'AAIIDiff'].some((k) => real(d[k])) || real(d.VIX?.current),
    spyDailyMove: (d) => real(d.value),
    fg: (d) => Number.isFinite(d.score) && d._meta?.source !== 'Failed',
    fred: (d) => hasKeys(d.yieldCurve) || hasKeys(d.indicators),
    extra: (d) => ['fx', 'commodities', 'rates'].some((k) => hasKeys(d[k])),
    history: (d) => hasKeys(d.metrics),
    jev: (d) => d.enabled === false || Array.isArray(d.pills),
    vol: (d) => Array.isArray(d.tickers) && d.tickers.length > 0,
};
export function isLiveAnswer(key, d) {
    if (d == null || typeof d !== 'object' || d.error) return false;
    try { return LIVE_TESTS[key] ? !!LIVE_TESTS[key](d) : true; } catch { return false; }
}

/** Drop copies saved by other deploys (~1 MB each) so they never fill the quota. */
export function purgeOldSnaps(store = storage()) {
    let n = 0;
    try {
        const old = [];
        for (let i = 0; i < store.length; i++) {
            const k = store.key(i);
            if (k && k.startsWith(SNAP_ROOT) && !k.startsWith(SNAP_PREFIX)) old.push(k);
        }
        for (const k of old) { store.removeItem(k); n++; }
    } catch { /* blocked storage: nothing to purge */ }
    return n;
}

/** "10:42" when saved today, "Thu 10:42" otherwise — device-local time. */
export function savedLabel(savedAt, now = Date.now()) {
    if (!Number.isFinite(savedAt)) return '';
    const d = new Date(savedAt);
    const hm = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    if (d.toDateString() === new Date(now).toDateString()) return hm;
    // A week or more ago, a bare weekday reads as this week: add the date ("Thu Sep 17 09:05").
    if (now - savedAt > 6 * 864e5) return `${d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }).replace(',', '')} ${hm}`;
    return `${d.toLocaleDateString('en-US', { weekday: 'short' })} ${hm}`;
}
