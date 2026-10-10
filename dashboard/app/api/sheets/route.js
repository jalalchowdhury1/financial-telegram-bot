import { fetchText, fetchJson } from '../../../lib/fetcher';
import { faultsFrom } from '../../../lib/faults';
import { cacheHeaders } from '../../../lib/cdn';
import { resolveVixFearGreedTag } from '../../../lib/vixFearGreed';
import { loadLastGood, saveLastGood } from '../../../lib/store';
import { resolveAaii } from '../../../lib/aaii';
import { defaultKv } from '../../../lib/kv';
import { resolveLiveFields, fillFromLastGood, persistLiveFields, summarize, toResults } from '../../../lib/sheetsCascade';
import bakedAaii from '../../../lib/data/aaiiNewest.json';

export const dynamic = 'force-dynamic';

// One fetch per URL per request: the VIX tag and the VIX-level CBOE tier both read
// VIX_History.csv, and a second download of ~470 KB buys nothing.
function memoFetch(fn) {
    const seen = new Map();
    return (url, opts) => {
        if (!seen.has(url)) seen.set(url, fn(url, opts));
        return seen.get(url);
    };
}

/**
 * NotSoBoring / FrontRunner / VIX pill, field by field (lib/sheetsCascade.js), plus
 * AAII straight from AAII (lib/aaii.js) and the VIX fear/greed tag computed from
 * CBOE → FRED (lib/vixFearGreed.js). Never throws: any failure degrades to a
 * flagged-stale last-good copy, then 'N/A' — never an unflagged old value.
 */
export async function GET(request) {
    request.headers.get('user-agent'); // touch the request: keeps the route dynamic (AGENTS.md §3)
    const faults = faultsFrom(request);
    const store = { load: loadLastGood, save: saveLastGood };
    const text = memoFetch(fetchText);
    try {
        // AAII never throws out of here: any failure = 'N/A' + hasErrors, never an old sheet value.
        const aaiiPromise = resolveAaii({ fetchText, store, kv: defaultKv, baked: bakedAaii, faults })
            .catch((e) => ({ payload: null, messages: [`aaii resolver threw: ${String(e?.message).slice(0, 120)}`] }));
        const tagPromise = resolveVixFearGreedTag({ fredApiKey: process.env.FRED_API_KEY, fetchJson, fetchText: text, faults })
            .catch((e) => ({ tag: 'N/A', tier: 'none', fallback: true, message: `VIX fear/greed: resolver threw (${String(e?.message).slice(0, 120)})` }));
        const livePromise = resolveLiveFields({ faults, fetchText: text, fetchJson, fredApiKey: process.env.FRED_API_KEY });
        const [{ fields, messages: liveMessages }, aaii, tag] = await Promise.all([livePromise, aaiiPromise, tagPromise]);

        if (tag.tag && tag.tag !== 'N/A') {
            fields.vixFearGreed = { value: tag.tag, source: tag.tier === 'cboe' ? 'CBOE-computed' : 'FRED-computed (lags a trading day)', tier: tag.tier, live: tag.tier === 'cboe', stale: !!tag.stale };
        } else {
            fields.vixFearGreed = null;
        }
        const filled = await fillFromLastGood(fields, { faults, store, kv: defaultKv });
        await persistLiveFields(fields, { faults, store, kv: defaultKv, kvRecord: filled.kvRecord });

        const summary = summarize(fields);
        const messages = [...liveMessages, ...filled.messages];
        if (!summary.hasErrors) messages.unshift('Live data loaded');
        messages.push(tag.message);
        let { hasErrors } = summary;

        const a = aaii.payload;
        const results = {
            ...toResults(fields),
            AAIIDiff: a ? a.diff : 'N/A',
            AAII: a
                ? { bull: a.bull, neutral: a.neutral, bear: a.bear, as_of: a.as_of, source: a.source, stale: a.stale, lastGood: !!aaii.lastGood }
                : null,
        };
        if (!a) {
            hasErrors = true;
            messages.push(`AAII unavailable: ${aaii.messages.join(' | ')}`);
        } else {
            messages.push(`AAII ${a.as_of} via ${a.source}${aaii.cachedAt ? ` (cached ${aaii.cachedAt})` : ''}`);
            if (a.stale) { hasErrors = true; messages.push(`AAII is STALE: survey ${a.as_of} is more than 9 days old`); }
            if (aaii.lastGood) { hasErrors = true; messages.push('AAII from the last good copy: every live tier failed'); }
        }

        const body = {
            ...results,
            _meta: { source: summary.source, hasErrors, stale: summary.stale, staleFields: summary.staleFields, fields: summary.fields, messages },
        };
        // Edge-cached only when every field is live from the primary sheet and the tag is
        // CBOE-computed, and never on a fault test (lib/cdn.js).
        const degraded = summary.hasErrors || !!tag.fallback;
        return Response.json(body, { headers: cacheHeaders('sheets', { payload: body, testMode: faults.size > 0, degraded }) });
    } catch (e) {
        // Belt and braces: nothing above should throw, but the route must never 500.
        return Response.json({
            ...toResults({}),
            AAIIDiff: 'N/A',
            AAII: null,
            _meta: { source: 'Static Defaults', hasErrors: true, stale: false, staleFields: [], messages: [`sheets route failed: ${String(e?.message).slice(0, 160)}`] },
        }, { headers: { 'cache-control': 'no-store' } });
    }
}
