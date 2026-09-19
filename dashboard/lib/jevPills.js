/**
 * Jev pills assembly logic — pure function that takes already-fetched data
 * and produces the route payload. Kept separate from route.js so the assembly
 * can be unit-tested without importing next/server or hitting the network.
 *
 * Exports:
 *   assemblePills({ raw, jevAnswers, yesterday, mode }) → payload object
 */

import { JEV_P_FLOOR } from './jev';
import {
    toData,
    buildState,
    ruleVerdicts,
    conflictPairs,
    mergeVerdicts,
    diffSinceYesterday,
} from './jevBrief';

/**
 * @param {object} opts
 * @param {object} opts.raw — raw route payloads { spy, fg, vol, fred, breadth, sheets, t10y3m }
 * @param {object|null} opts.jevAnswers — from judgeMany(), or null
 * @param {object|null} opts.yesterday — logged day payload (with .date), or null
 * @param {string} opts.mode — 'on' | 'rules'
 * @returns {object} the full jev-pills payload (less _meta.logged, which the route appends)
 */
export function assemblePills({ raw, jevAnswers, yesterday, mode }) {
    const data = toData(raw);
    const state = buildState(data);
    const rule = ruleVerdicts(data);
    const merged = mergeVerdicts(rule, jevAnswers, JEV_P_FLOOR);
    const cps = conflictPairs(data);
    const since = diffSinceYesterday(merged, yesterday ? yesterday.pills || yesterday : null);

    const asOf = new Date().toISOString();

    // Collect source labels from each raw route's _meta
    const sources = {};
    if (raw && typeof raw === 'object') {
        for (const key of ['spy', 'fg', 'vol', 'fred', 'breadth', 'sheets']) {
            const src = raw[key]?._meta?.source || raw[key]?.source || null;
            if (src) sources[key] = src;
        }
    }

    const jevStatus = mode === 'rules'
        ? 'rules'
        : (jevAnswers ? 'ok' : (process.env.TYPESAFE_API_KEY ? 'error: no answers' : 'off'));

    return {
        enabled: true,
        mode,
        asOf,
        state,
        pills: merged,
        conflictPairs: cps,
        since: since.noBaseline
            ? { ...since, date: null }
            : { ...since, date: yesterday?.date || null },
        _meta: {
            jev: jevStatus,
            sources,
        },
    };
}