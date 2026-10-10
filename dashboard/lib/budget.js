/**
 * A per-request time budget: `race(promise, onTimeout)` resolves with the promise's value,
 * or with `onTimeout` once the budget (measured from `startedAt`) is spent — so a slow
 * optional tier degrades to "unavailable" instead of running the route into a 504.
 * Every stage still gets at least `minMs` (a late stage is not starved to zero).
 * A rejected promise resolves to `onTimeout` too. Never throws.
 *
 * A raced-out promise keeps running in the background; callers must only use the value
 * `race` returns (never let the late promise write into an already-served payload).
 */
export const TIMED_OUT = Symbol('budget-timed-out');

export function makeBudget(totalMs, { startedAt = Date.now(), minMs = 1000 } = {}) {
    const deadlineAt = startedAt + totalMs;
    const remaining = () => Math.max(minMs, deadlineAt - Date.now());
    const race = (promise, onTimeout = TIMED_OUT) => {
        let t;
        const timer = new Promise((r) => { t = setTimeout(() => r(onTimeout), remaining()); });
        return Promise.race([Promise.resolve(promise).catch(() => onTimeout), timer]).finally(() => clearTimeout(t));
    };
    return { race, remaining, deadlineAt };
}
