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
import { fredObservations, fredGraphCsv, treasuryYieldCurveCsv } from '../../../lib/sources';
import { judgeMany } from '../../../lib/jev';
import { JEV_QUESTIONS, toData, buildState } from '../../../lib/jevBrief';
import { assemblePills } from '../../../lib/jevPills';
import { logVerdicts, yesterday as loadYesterday } from '../../../lib/jevLog';
import { FRESH, claims4wkFromHistory, sahmFromHistory, parseTreasurySpreadCsv, resolvePillInput } from '../../../lib/jevInputs';
import { makePillStore } from '../../../lib/jevStore';
import { fetchSheetLkg } from '../../../lib/sheetLkg';
import { parseFredGraphCsv } from '../../../lib/horsemen';

export const fetchCache = 'default-cache';
// Outage path: sibling wait (10 s) + backup tiers with their own timeouts. Valid on every Vercel plan.
export const maxDuration = 30;

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
 * Mutates `raw` to repair four pill inputs the sibling /api/fred route may
 * have served as null (or not at all — a FRED outage makes that route slow and
 * fetchSibling gives up at 10 s). Tiers, in order, per input:
 *
 *   t10y3m  fred API → treasury (10 Yr − 3 Mo, origin publisher) → fredcsv → last-good
 *   nfci    sheet (dashboard_lkg snapshot)                        → fredcsv → last-good
 *   claims  horsemen (sibling's repaired ICSA history, 4-wk avg) → sheet → fredcsv → last-good
 *   sahm    horsemen (sibling's repaired UNRATE history, Sahm)    → sheet → fredcsv → last-good
 *
 * `fredcsv` is FRED's keyless CSV — a documented phantom on Vercel (AGENTS.md),
 * kept as a harmless 5 s last attempt. Last-good = /tmp + KV (lib/jevStore.js),
 * SEEDED from the sibling's own value on every healthy call, so the tier exists
 * before the outage that needs it. Faults: `hm_fred`, `hm_treasury`, `hm_sheet`
 * (or the sibling's `sheetlkg`), `hm_horsemen`, `hm_fredcsv`, `lastgood`.
 *
 * Never throws. Returns inputSources for _meta; `opts.diag` (if given) receives
 * `{ tried: { input: [...] }, seeded: [...] }` for `_meta.inputTried`.
 */
export async function repairPillInputs(raw, { fredKey, faults = new Set(), now = new Date(), store = null, diag = null } = {}) {
    const inputSources = { t10y3m: 'fred-route', nfci: 'fred-route', claims: 'fred-route', sahm: 'fred-route' };
    if (!raw) return inputSources;
    const lg = store || makePillStore();
    const tried = {};
    const seeded = [];
    const LG_MAX_MS = 14 * 864e5; // weekly / monthly series: a two-week-old copy is still the print

    // One Sheet snapshot shared by every tier that wants it (lazy: only fetched on a miss).
    let sheetPromise = null;
    const sheet = () => {
        if (faults.has('sheetlkg')) throw new Error('[injected fault: sheetlkg]');
        if (!sheetPromise) sheetPromise = fetchSheetLkg(now).catch(() => null);
        return sheetPromise;
    };
    const readSheet = (pick) => async () => {
        const snap = await sheet();
        const v = snap ? pick(snap) : null;
        return v && Number.isFinite(v.value) ? { value: v.value, asOf: v.asOf ?? null } : null;
    };
    const lastDate = (hist) => (Array.isArray(hist) && hist.length ? hist[hist.length - 1]?.date ?? null : null);

    const fredcsv = (id, freshnessDays, derive) => ({
        name: 'fredcsv', freshnessDays, ...(derive ? { derive } : {}),
        fetch: async () => parseFredGraphCsv(await fredGraphCsv(id, { timeout: 4000 })),
    });

    // Helper to set a fred.checklist or fred.indicators value
    const setFredField = (path, valueObj) => {
        const parts = path.split('.');
        if (!raw.fred || typeof raw.fred !== 'object') raw.fred = {};
        if (parts[0] === 'fred') parts.shift();
        let obj = raw.fred;
        for (let i = 0; i < parts.length - 1; i++) {
            if (!obj[parts[i]] || typeof obj[parts[i]] !== 'object') obj[parts[i]] = {};
            obj = obj[parts[i]];
        }
        obj[parts[parts.length - 1]] = valueObj;
    };
    const isFinite_ = (v) => v != null && Number.isFinite(v);

    // -- 1. T10Y3M — always resolved here (the sibling never carries it) --
    const t10y3m = async () => {
        const sources = [];
        if (fredKey) {
            sources.push({
                name: 'fred', freshnessDays: FRESH.T10Y3M,
                fetch: async () => {
                    const obs = await fredObservations('T10Y3M', fredKey, { limit: 30 });
                    return (obs || []).slice().reverse(); // fredObservations is DESCENDING
                },
            });
        }
        sources.push({
            name: 'treasury', freshnessDays: FRESH.T10Y3M,
            fetch: async () => {
                const year = now.getUTCFullYear();
                let rows = parseTreasurySpreadCsv(await treasuryYieldCurveCsv(year, { timeout: 5000 }), '3 mo', '10 yr');
                if (rows.length < 2) { // first days of January: the current-year file is near-empty
                    const prior = parseTreasurySpreadCsv(await treasuryYieldCurveCsv(year - 1, { timeout: 5000 }), '3 mo', '10 yr');
                    rows = [...prior, ...rows];
                }
                return rows;
            },
        });
        sources.push(fredcsv('T10Y3M', FRESH.T10Y3M));
        const r = await resolvePillInput({ sources, faults, now, lastGoodKey: 'jev-t10y3m', maxStaleMs: LG_MAX_MS, store: lg });
        raw.t10y3m = r.value; // number or null — the contract toData expects
        inputSources.t10y3m = r.source; // 'fred' is this input's primary
        tried.t10y3m = r.tried;
    };

    // -- 2. NFCI — only when the sibling has no finite value --
    const nfci = async () => {
        const cur = raw?.fred?.checklist?.nfci;
        if (isFinite_(cur?.value)) { seeded.push('nfci'); await seed('jev-nfci', cur); return; }
        const r = await resolvePillInput({
            sources: [
                { name: 'sheet', freshnessDays: FRESH.NFCI, read: readSheet((s) => s.checklist?.nfci) },
                fredcsv('NFCI', FRESH.NFCI),
            ],
            faults, now, lastGoodKey: 'jev-nfci', maxStaleMs: LG_MAX_MS, store: lg,
        });
        if (r.value != null) setFredField('fred.checklist.nfci', { value: r.value, asOf: r.asOf, stale: false, unavailable: false, source: r.source });
        inputSources.nfci = r.source;
        tried.nfci = r.tried;
    };

    // -- 3. Claims — 4-week average in thousands, exactly like indicators.claims --
    const claims = async () => {
        const cur = raw?.fred?.indicators?.claims;
        if (isFinite_(cur?.value)) { seeded.push('claims'); await seed('jev-claims', cur); return; }
        const horsemenHist = raw?.fred?.horsemen?.claims?.history;
        const r = await resolvePillInput({
            sources: [
                { name: 'horsemen', freshnessDays: FRESH.ICSA, derive: claims4wkFromHistory,
                    fetch: async () => (Array.isArray(horsemenHist) ? horsemenHist : []) },
                { name: 'sheet', freshnessDays: FRESH.ICSA,
                    read: readSheet((s) => {
                        const v = s.indicators?.claims;
                        return v ? { value: v.value, asOf: v.asOf ?? lastDate(s.horsemen?.claims?.history) } : null;
                    }) },
                fredcsv('ICSA', FRESH.ICSA, claims4wkFromHistory),
            ],
            faults, now, lastGoodKey: 'jev-claims', maxStaleMs: LG_MAX_MS, store: lg,
        });
        if (r.value != null) setFredField('fred.indicators.claims', { value: r.value, asOf: r.asOf, stale: false, unavailable: false, source: r.source });
        inputSources.claims = r.source;
        tried.claims = r.tried;
    };

    // -- 4. Sahm — same shape over monthly UNRATE --
    const sahm = async () => {
        const cur = raw?.fred?.indicators?.sahmRule;
        if (isFinite_(cur?.value)) { seeded.push('sahm'); await seed('jev-sahm', cur); return; }
        const unHist = raw?.fred?.horsemen?.unemployment?.history;
        const r = await resolvePillInput({
            sources: [
                { name: 'horsemen', freshnessDays: FRESH.UNRATE, derive: sahmFromHistory,
                    fetch: async () => (Array.isArray(unHist) ? unHist : []) },
                { name: 'sheet', freshnessDays: FRESH.UNRATE,
                    read: readSheet((s) => {
                        const v = s.indicators?.sahmRule;
                        return v ? { value: v.value, asOf: v.asOf ?? lastDate(s.horsemen?.unemployment?.history) } : null;
                    }) },
                fredcsv('UNRATE', FRESH.UNRATE, sahmFromHistory),
            ],
            faults, now, lastGoodKey: 'jev-sahm', maxStaleMs: LG_MAX_MS, store: lg,
        });
        if (r.value != null) setFredField('fred.indicators.sahmRule', { value: r.value, asOf: r.asOf, stale: false, unavailable: false, source: r.source });
        inputSources.sahm = r.source;
        tried.sahm = r.tried;
    };

    // Seed the last-good tier from the sibling's own healthy value (never in fault mode).
    async function seed(key, cur) {
        if (faults.size > 0) return;
        try { await lg.save(key, { value: cur.value, asOf: cur.asOf ?? null, source: 'fred-route' }); } catch { /* best effort */ }
    }

    // Independent inputs → run together so the worst outage path is the slowest chain, not their sum.
    await Promise.all([t10y3m, nfci, claims, sahm].map((fn) => fn().catch(() => {})));

    if (diag && typeof diag === 'object') { diag.tried = tried; diag.seeded = seeded; }
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
        const diag = {};
        const inputSources = await repairPillInputs(raw, {
            fredKey: process.env.FRED_API_KEY,
            faults,
            diag,
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
        payload._meta.inputTried = diag.tried || {};

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