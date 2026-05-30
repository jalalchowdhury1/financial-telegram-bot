/**
 * Fault-injection for testing the fallback waterfalls. Activated only by the
 * `?_fail=<comma,list>` query param, e.g. ?_fail=lambda,polygon,yahoo.
 *
 * It can ONLY degrade the caller's own response — it never writes to caches or
 * the last-known-good store (serve() skips writes when faults are present), so
 * it cannot poison data for anyone else. Safe to ship; meaningless without the
 * explicit param.
 */
export function faultsFrom(request) {
    try {
        const v = new URL(request.url).searchParams.get('_fail') || '';
        return new Set(v.split(',').map((s) => s.trim().toLowerCase()).filter(Boolean));
    } catch { return new Set(); }
}

/** Throw if this named source is being artificially failed. */
export function trip(name, faults) {
    if (faults && faults.has(name)) throw new Error(`[injected fault: ${name}]`);
}

/** Run fn unless `name` is in the fault set (then throw). */
export async function gate(name, faults, fn) {
    trip(name, faults);
    return fn();
}
