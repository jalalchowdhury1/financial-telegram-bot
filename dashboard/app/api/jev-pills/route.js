/**
 * /api/jev-pills — Jev regime pills endpoint.
 *
 * Thin wrapper: fetches sibling routes and T10Y3M, then delegates assembly to
 * assemblePills() from lib/jevPills.js. Never-throw via serve().
 *
 * Env: JEV_PILLS=off → { enabled: false } immediately (before serve).
 *      JEV_PILLS=rules → rules only, no Jev call.
 *      JEV_PILLS=on or unset → Jev with rule backup.
 */
import { serve } from '../../../lib/store';
import { faultsFrom } from '../../../lib/faults';
import { fredObservations } from '../../../lib/sources';
import { judgeMany } from '../../../lib/jev';
import { JEV_QUESTIONS, toData, buildState } from '../../../lib/jevBrief';
import { assemblePills } from '../../../lib/jevPills';
import { logVerdicts, yesterday as loadYesterday } from '../../../lib/jevLog';

export const fetchCache = 'default-cache';

const SIBLING_ROUTES = ['/api/spy', '/api/fear-greed', '/api/vol', '/api/fred', '/api/breadth', '/api/sheets'];
const FETCH_TIMEOUT_MS = 10_000;

/**
 * Fetch one sibling route on the same origin.
 * Returns the parsed JSON body, or null on any failure.
 */
async function fetchSibling(baseOrigin, path, faults) {
    const url = `${baseOrigin}${path}`;
    // Append fault params if present
    const qs = [];
    if (faults && faults.size > 0) {
        qs.push(`_fail=${[...faults].join(',')}`);
    }
    const fullUrl = qs.length ? `${url}?${qs.join('&')}` : url;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
        const res = await fetch(fullUrl, {
            headers: { 'user-agent': 'jev-pills-route/1.0' },
            signal: controller.signal,
            // Siblings already cache their own upstreams; the pills must see their current payloads
            cache: 'no-store',
        });
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    } finally {
        clearTimeout(timer);
    }
}

/**
 * Fetch T10Y3M from FRED (5 observations is plenty to get the latest).
 */
async function fetchT10y3m(fredKey) {
    if (!fredKey) return null;
    try {
        const obs = await fredObservations('T10Y3M', fredKey, { limit: 5 });
        if (!obs || !obs.length) return null;
        return obs[0].value ?? null;
    } catch {
        return null;
    }
}

export async function GET(request) {
    // Touch the request so this handler runs per-request (dynamic, not static).
    request.headers.get('user-agent');

    // Kill switch: if env is 'off', return immediately (no serve wrapper).
    const jevPillsEnv = (process.env.JEV_PILLS || '').toLowerCase();
    if (jevPillsEnv === 'off') {
        return Response.json({ enabled: false }, {
            status: 200,
            headers: { 'cache-control': 'no-store' },
        });
    }

    const faults = faultsFrom(request);
    const mode = jevPillsEnv === 'rules' ? 'rules' : 'on';

    return serve('jev-pills', async () => {
        const origin = new URL(request.url).origin;

        // Fetch all sibling routes and T10Y3M in parallel
        const fetches = [
            ...SIBLING_ROUTES.map((path) => fetchSibling(origin, path, faults)),
            fetchT10y3m(process.env.FRED_API_KEY),
        ];
        const results = await Promise.all(fetches);

        const raw = {
            spy: results[0],
            fg: results[1],
            vol: results[2],
            fred: results[3],
            breadth: results[4],
            sheets: results[5],
            t10y3m: results[6],
        };

        // Build state text for Jev
        const dataObj = toData(raw);
        const state = buildState(dataObj);

        // Call Jev unless mode is 'rules'
        let jevAnswers = null;
        if (mode === 'on') {
            jevAnswers = await judgeMany(state, JEV_QUESTIONS);
        }

        // Load yesterday's verdicts from KV
        const yesterdayData = await loadYesterday();

        // Assemble the payload
        const payload = assemblePills({ raw, jevAnswers, yesterday: yesterdayData, mode });

        // Log today's verdicts (best effort, never throw)
        const todayStr = new Date().toISOString().slice(0, 10);
        try {
            const logged = await logVerdicts(todayStr, {
                date: todayStr,
                pills: payload.pills,
                state: payload.state,
                spy: { price: raw.spy?.current ?? raw.spy?.price ?? null },
            });
            payload._meta.logged = logged;
        } catch {
            payload._meta.logged = false;
        }

        return payload;
    }, {
        faults,
        maxStaleMs: 6 * 3600e3, // 6 hours
        isGood: (p) => !!p && p.enabled === true && !!p.pills,
        fallback: {
            enabled: true,
            mode: 'rules',
            pills: null,
            _meta: { jev: 'error: no data' },
        },
    });
}