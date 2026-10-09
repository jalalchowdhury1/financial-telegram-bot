/**
 * /api/aaii — AAII Sentiment Survey straight from AAII (no Google Sheet).
 *
 * Contract (read by the Telegram bot and other repos — do not rename fields):
 *   200 { bull, neutral, bear, diff, as_of, source, stale, _meta }
 *       diff = bear − bull as "15.40%" (the string the site has always shown)
 *       as_of = survey week date YYYY-MM-DD; source = 'aaii.com' | 'substack'
 *       stale = as_of more than 9 days old (a missed week)
 *   503 { error } when every tier AND the last good copy are unavailable —
 *       never made-up numbers.
 *
 * Tiers + caching: lib/aaii.js (3 h per-instance cache, then aaii.com → Substack
 * API → Substack RSS → last good ≤ 21 d). Fault gates: aaii_http, aaii_substack,
 * aaii_rss, aaii_lastgood (e.g. ?_fail=aaii_http proves the Substack tier).
 */
import { fetchText } from '../../../lib/fetcher';
import { faultsFrom } from '../../../lib/faults';
import { loadLastGood, saveLastGood } from '../../../lib/store';
import { cacheHeaders } from '../../../lib/cdn';
import { resolveAaii } from '../../../lib/aaii';
import { defaultKv } from '../../../lib/factorStore';
import bakedAaii from '../../../lib/data/aaiiNewest.json';

export const dynamic = 'force-dynamic';

export async function GET(request) {
    request.headers.get('user-agent');
    let faults = new Set();
    try {
        faults = faultsFrom(request);
        const { payload, cachedAt, lastGood, messages } = await resolveAaii({
            fetchText,
            store: { load: loadLastGood, save: saveLastGood },
            kv: defaultKv,
            baked: bakedAaii,
            faults,
        });
        if (!payload) {
            return Response.json(
                { error: `AAII unavailable: ${messages.join(' | ')}`.slice(0, 600) },
                { status: 503, headers: { 'cache-control': 'no-store' } },
            );
        }
        const body = {
            ...payload,
            _meta: {
                source: payload.source,
                stale: payload.stale,
                hasErrors: !!(payload.stale || lastGood),
                lastGood: !!lastGood,
                cachedAt,
                messages,
            },
        };
        return Response.json(body, { headers: cacheHeaders('aaii', { payload: body, testMode: faults.size > 0 }) });
    } catch (e) {
        return Response.json(
            { error: `AAII route failed: ${String(e?.message).slice(0, 200)}` },
            { status: 503, headers: { 'cache-control': 'no-store' } },
        );
    }
}
