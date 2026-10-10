/**
 * Run a best-effort task (e.g. a durable KV last-good SET) WITHOUT holding the response.
 *
 * On Vercel the function may be frozen as soon as the response is sent, so a bare
 * fire-and-forget promise can be lost. Vercel's runtime exposes the request context
 * (the same hook `@vercel/functions`' `waitUntil` reads) on
 * `globalThis[Symbol.for('@vercel/request-context')]`; when present we hand it the
 * promise so the instance stays alive until it settles. Elsewhere (local dev, Jest, a
 * runtime without the hook) it degrades to fire-and-forget. Next 13.5 has no `after()`,
 * and this avoids adding `@vercel/functions` as a dependency.
 *
 * Never throws; the returned promise never rejects (tests may await it).
 */
const CTX = Symbol.for('@vercel/request-context');

export function runInBackground(task) {
    let p;
    try {
        p = Promise.resolve(typeof task === 'function' ? task() : task).catch(() => undefined);
    } catch {
        return Promise.resolve(undefined);
    }
    try {
        const ctx = globalThis[CTX]?.get?.();
        if (ctx && typeof ctx.waitUntil === 'function') ctx.waitUntil(p);
    } catch { /* no request context: fire-and-forget */ }
    return p;
}
