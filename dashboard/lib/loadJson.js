/**
 * loadJson.js — how the page reads its /api routes. Never throws: null = no answer,
 * and the caller keeps whatever it was already showing.
 *
 * - Timeout (60 s — double the 30 s that factors + jev-pills cap themselves at, and
 *   far past any route's normal cold time): a stalled phone connection used to leave
 *   the refresh spinner turning forever, with the refresh button disabled the whole time.
 * - One retry, 1.5 s later, for a network error, an unreadable body, or a 5xx (the
 *   routes themselves answer 200 even when degraded, so a 5xx is Vercel: a cold-start
 *   crash or a 504). A TIMEOUT is not retried — the route already had its full time.
 * - `bust` adds `?_t=<now>` so the request skips Vercel's edge cache (lib/cdn.js).
 *   Automatic loads leave it off and may be answered from the edge in ~50 ms.
 */
export const LOAD_TIMEOUT_MS = 60000;
export const RETRY_DELAY_MS = 1500;

export function withBust(path, bust, now = Date.now()) {
    if (!bust) return path;
    return `${path}${path.includes('?') ? '&' : '?'}_t=${now}`;
}

export async function getJson(path, {
    bust = false,
    timeoutMs = LOAD_TIMEOUT_MS,
    retries = 1,
    retryDelayMs = RETRY_DELAY_MS,
    fetchImpl = (...a) => fetch(...a),
    sleep = (ms) => new Promise((r) => setTimeout(r, ms)),
} = {}) {
    for (let attempt = 0; attempt <= retries; attempt++) {
        const last = attempt === retries;
        const ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
        let timedOut = false;
        const timer = ctrl ? setTimeout(() => { timedOut = true; ctrl.abort(); }, timeoutMs) : null;
        try {
            const res = await fetchImpl(withBust(path, bust), ctrl ? { signal: ctrl.signal } : {});
            if (res.status >= 500 && !last) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch {
            if (timedOut || last) return null;
            await sleep(retryDelayMs);
        } finally {
            if (timer) clearTimeout(timer);
        }
    }
    return null;
}
