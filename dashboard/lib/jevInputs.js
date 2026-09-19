/**
 * Pure helpers for repairing pill inputs that have incomplete data from the
 * sibling /api/fred route.
 *
 * Every function here is unit-testable with no network or side effects.
 *
 * Exports:
 *   FRESH             — freshness deadlines per input (days)
 *   claims4wkFromHistory — 4-week average of weekly ICSA in thousands
 *   sahmFromHistory       — Sahm rule from monthly UNRATE history
 *   resolvePillInput      — generic cascade over sources → { value, asOf, source, tried }
 */

import { isStale } from './freshness';
import { loadLastGood, saveLastGood } from './store';

export const FRESH = { T10Y3M: 7, NFCI: 14, ICSA: 14, UNRATE: 45 };

/**
 * 4-week average of weekly ICSA in THOUSANDS, from an ascending history.
 * Each value in the history is a raw number (e.g. 231000 for 231k).
 * Returns the mean ÷ 1000, not rounded. null if < 4 points.
 */
export function claims4wkFromHistory(history) {
    if (!Array.isArray(history) || history.length < 4) return null;
    const last4 = history.slice(-4);
    const total = last4.reduce((sum, p) => sum + (p.value ?? 0), 0);
    return total / 4000; // ÷1000 for thousands, ÷4 for the average
}

/**
 * Sahm: mean of the latest 3 months minus the min of the latest 12 months,
 * from an ascending monthly UNRATE history (each point has a date and value
 * as a decimal fraction, e.g. 4.1 for 4.1%).
 * Returns the raw difference (not a percentage subtraction). null if < 12 points.
 */
export function sahmFromHistory(history) {
    if (!Array.isArray(history) || history.length < 12) return null;
    const last12 = history.slice(-12);
    const last3 = last12.slice(-3);
    const mean3 = last3.reduce((s, p) => s + (p.value ?? 0), 0) / 3;
    const min12 = Math.min(...last12.map((p) => p.value ?? Infinity));
    return mean3 - min12;
}

/**
 * Generic cascade for one pill input.
 *
 * sources: [{ name, freshnessDays, fetch, derive? }] — same contract as
 *   resolveHorseman. fetch() must return an ascending [{date, value}] array or
 *   throw. Stale sources (newest point older than freshnessDays) are skipped.
 *   derive(history) (optional) turns the history into the value we keep —
 *   e.g. a 4-week average — instead of the newest point; null → source skipped.
 * faults: Set (see lib/faults.js) — when present, skip named sources.
 * now: Date — injectable "current" time for staleness tests.
 * lastGoodKey: string — store key for /tmp last-known-good, e.g. 'jev-t10y3m'.
 * maxStaleMs: number — max age in ms for a last-good entry (default 7 days).
 *
 * Returns { value: number|null, asOf: string|null, source: string|null, tried: string[] }.
 * Never throws. Writes last-good on any live success (unless faults.size > 0).
 */
export async function resolvePillInput({ sources, faults = new Set(), now = new Date(), lastGoodKey, maxStaleMs = 7 * 864e5 }) {
    const tried = [];

    for (const s of sources) {
        if (faults.has(`hm_${s.name}`)) {
            tried.push(`${s.name}:off`);
            continue;
        }
        try {
            const history = await s.fetch();
            if (!Array.isArray(history) || history.length < 2) {
                tried.push(`${s.name}:empty`);
                continue;
            }
            const last = history[history.length - 1];
            if (!last || !Number.isFinite(last.value) || !last.date) {
                tried.push(`${s.name}:empty`);
                continue;
            }
            if (isStale(last.date, s.freshnessDays, now)) {
                tried.push(`${s.name}:stale(${last.date})`);
                continue;
            }
            const value = typeof s.derive === 'function' ? s.derive(history) : last.value;
            if (!Number.isFinite(value)) {
                tried.push(`${s.name}:underived`);
                continue;
            }
            tried.push(`${s.name}:ok`);
            const result = { value, asOf: last.date, source: s.name, tried };
            if (lastGoodKey && faults.size === 0) {
                saveLastGood(lastGoodKey, result);
            }
            return result;
        } catch (e) {
            tried.push(`${s.name}:err`);
        }
    }

    // All sources failed — try last-known-good
    if (lastGoodKey && !(faults.has('lastgood'))) {
        const lg = loadLastGood(lastGoodKey, maxStaleMs);
        if (lg && lg.data && lg.data.value != null) {
            tried.push(`lastgood:ok(${lg.savedAt})`);
            return { ...lg.data, tried, source: 'lastgood' };
        }
    }

    tried.push('lastgood:none');
    return { value: null, asOf: null, source: null, tried };
}