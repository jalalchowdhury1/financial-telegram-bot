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
import { fredObservations, fredGraphCsv } from '../../../lib/sources';
import { judgeMany } from '../../../lib/jev';
import { JEV_QUESTIONS, toData, buildState } from '../../../lib/jevBrief';
import { assemblePills } from '../../../lib/jevPills';
import { logVerdicts, yesterday as loadYesterday } from '../../../lib/jevLog';
import { FRESH, claims4wkFromHistory, sahmFromHistory, resolvePillInput } from '../../../lib/jevInputs';
import { parseFredGraphCsv } from '../../../lib/horsemen';

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
 * repairPillInputs — run AFTER sibling fetches, BEFORE toData.
 *
 * Mutates `raw` to repair four pill inputs that the sibling /api/fred route
 * may have served as null (no data available from its own cascade). Each only
 * fires when the sibling's value is absent. Never throws.
 *
 * Returns inputSources for _meta: a map of input name → source label.
 */
export async function repairPillInputs(raw, { fredKey, faults = new Set(), now = new Date() }) {
    const inputSources = { t10y3m: 'fred-route', nfci: 'fred-route', claims: 'fred-route', sahm: 'fred-route' };
    if (!raw) return inputSources;

    // -- 1. T10Y3M (always resolved here, replaces fetchT10y3m) --
    const t10y3mSources = [];
    if (fredKey) {
        t10y3mSources.push({
            name: 'fred',
            freshnessDays: FRESH.T10Y3M,
            fetch: async () => {
                const obs = await fredObservations('T10Y3M', fredKey, { limit: 30 });
                // fredObservations returns DESCENDING (newest first); reverse for resolvePillInput
                return (obs || []).slice().reverse();
            },
        });
    }
    t10y3mSources.push({
        name: 'fredcsv',
        freshnessDays: FRESH.T10Y3M,
        fetch: async () => parseFredGraphCsv(await fredGraphCsv('T10Y3M')),
    });
    const t10y3mResult = await resolvePillInput({ sources: t10y3mSources, faults, now, lastGoodKey: 'jev-t10y3m' });
    raw.t10y3m = t10y3mResult.value; // number or null — same contract fetchT10y3m had
    inputSources.t10y3m = t10y3mResult.source; // 'fred' is this input's primary

    // Helper to set a fred.checklist or fred.indicators value
    const setFredField = (path, valueObj) => {
        const parts = path.split('.');
        let obj = raw;
        if (!obj.fred) obj.fred = {};
        if (parts[0] === 'fred') parts.shift();
        // Start from raw.fred (the stripped path is e.g. ['checklist', 'nfci'])
        obj = obj.fred;
        for (let i = 0; i < parts.length - 1; i++) {
            if (!obj[parts[i]] || typeof obj[parts[i]] !== 'object') obj[parts[i]] = {};
            obj = obj[parts[i]];
        }
        obj[parts[parts.length - 1]] = valueObj;
    };

    // -- 2. NFCI (C) — only when raw.fred?.checklist?.nfci?.value is not a finite number --
    const nfciValue = raw?.fred?.checklist?.nfci?.value;
    if (!(nfciValue != null && Number.isFinite(nfciValue))) {
        const nfciResult = await resolvePillInput({
            sources: [{ name: 'fredcsv', freshnessDays: FRESH.NFCI, fetch: async () => parseFredGraphCsv(await fredGraphCsv('NFCI')) }],
            faults, now, lastGoodKey: 'jev-nfci',
        });
        if (nfciResult.value != null) {
            setFredField('fred.checklist.nfci', {
                value: nfciResult.value,
                asOf: nfciResult.asOf,
                stale: false,
                unavailable: false,
                source: nfciResult.source,
            });
        }
        inputSources.nfci = nfciResult.source;
    }

    // -- 3. Claims (A) — only when raw.fred?.indicators?.claims?.value is not finite --
    // Tier 1: the sibling's already-repaired horsemen.claims history (bls/fredcsv
    // inside /api/fred). Tier 2: keyless FRED CSV. Tier 3: last-good. Every tier
    // is reduced to the 4-week average in thousands, exactly like indicators.claims.
    const claimsValue = raw?.fred?.indicators?.claims?.value;
    if (!(claimsValue != null && Number.isFinite(claimsValue))) {
        const horsemenHist = raw?.fred?.horsemen?.claims?.history;
        const claimsResult = await resolvePillInput({
            sources: [
                { name: 'horsemen', freshnessDays: FRESH.ICSA, derive: claims4wkFromHistory,
                    fetch: async () => (Array.isArray(horsemenHist) ? horsemenHist : []) },
                { name: 'fredcsv', freshnessDays: FRESH.ICSA, derive: claims4wkFromHistory,
                    fetch: async () => parseFredGraphCsv(await fredGraphCsv('ICSA')) },
            ],
            faults, now, lastGoodKey: 'jev-claims',
        });
        if (claimsResult.value != null) {
            setFredField('fred.indicators.claims', {
                value: claimsResult.value, asOf: claimsResult.asOf,
                stale: false, unavailable: false, source: claimsResult.source,
            });
        }
        inputSources.claims = claimsResult.source;
    }

    // -- 4. Sahm (B) — same shape over monthly UNRATE --
    const sahmValue = raw?.fred?.indicators?.sahmRule?.value;
    if (!(sahmValue != null && Number.isFinite(sahmValue))) {
        const unHist = raw?.fred?.horsemen?.unemployment?.history;
        const sahmResult = await resolvePillInput({
            sources: [
                { name: 'horsemen', freshnessDays: FRESH.UNRATE, derive: sahmFromHistory,
                    fetch: async () => (Array.isArray(unHist) ? unHist : []) },
                { name: 'fredcsv', freshnessDays: FRESH.UNRATE, derive: sahmFromHistory,
                    fetch: async () => parseFredGraphCsv(await fredGraphCsv('UNRATE')) },
            ],
            faults, now, lastGoodKey: 'jev-sahm',
        });
        if (sahmResult.value != null) {
            setFredField('fred.indicators.sahmRule', {
                value: sahmResult.value, asOf: sahmResult.asOf,
                stale: false, unavailable: false, source: sahmResult.source,
            });
        }
        inputSources.sahm = sahmResult.source;
    }

    return inputSources;
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

        // Fetch all sibling routes in parallel (no longer fetches T10Y3M here)
        const fetches = SIBLING_ROUTES.map((path) => fetchSibling(origin, path, faults));
        const results = await Promise.all(fetches);

        const raw = {
            spy: results[0],
            fg: results[1],
            vol: results[2],
            fred: results[3],
            breadth: results[4],
            sheets: results[5],
            t10y3m: null, // placeholder — repairPillInputs will set it
        };

        // Repair pill inputs that sibling routes couldn't serve
        const inputSources = await repairPillInputs(raw, {
            fredKey: process.env.FRED_API_KEY,
            faults,
        });

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
        const payload = assemblePills({ raw, jevAnswers, yesterday: yesterdayData, mode, inputSources });

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