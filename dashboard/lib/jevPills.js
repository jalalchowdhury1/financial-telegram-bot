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
    pillFactors,
    PILLS,
} from './jevBrief';

/**
 * @param {object} opts
 * @param {object} opts.raw — raw route payloads { spy, fg, vol, fred, breadth, sheets, t10y3m }
 * @param {object|null} opts.jevAnswers — from judgeMany(), or null
 * @param {object|null} opts.yesterday — logged day payload (with .date), or null
 * @param {string} opts.mode — 'on' | 'rules'
 * @param {object} [opts.inputSources] — per-input source labels from repairPillInputs, default {}
 * @returns {object} the full jev-pills payload (less _meta.logged, which the route appends)
 */
export function assemblePills({ raw, jevAnswers, yesterday, mode, inputSources = {} }) {
    const data = toData(raw);
    const state = buildState(data);
    const rule = ruleVerdicts(data);
    const merged = mergeVerdicts(rule, jevAnswers, JEV_P_FLOOR);

    // Attach raw Jev answer to every pill (null when Jev gave no answer)
    for (const pill of PILLS) {
        merged[pill].jev = jevAnswers?.[pill]
            ? { verdict: jevAnswers[pill].verdict, p: jevAnswers[pill].p }
            : null;
    }

    const factors = pillFactors(data);
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

    // Freshness per feed, for the popup's "data through" line (null when a feed has no date)
    const dataAsOf = {
        breadth: raw?.breadth?.updated_at ?? null,
        vol: raw?.vol?.updated_at ?? null,
        fred: raw?.fred?.yieldCurve?.asOf ?? raw?.fred?.yieldCurve?.date ?? null,
    };

    const jevStatus = mode === 'rules'
        ? 'rules'
        : (jevAnswers ? 'ok' : (process.env.TYPESAFE_API_KEY ? 'error: no answers' : 'off'));

    return {
        enabled: true,
        mode,
        asOf,
        state,
        pills: merged,
        factors,
        conflictPairs: cps,
        since: since.noBaseline
            ? { ...since, date: null }
            : { ...since, date: yesterday?.date || null },
        _meta: {
            jev: jevStatus,
            sources,
            dataAsOf,
            inputSources,
        },
    };
}