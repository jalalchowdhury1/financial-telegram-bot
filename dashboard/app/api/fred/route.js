import { FRED_SERIES, FRED_FRESHNESS, EXTERNAL_URLS } from '../../../lib/constants';
import { fetchJson, proxyFetch } from '../../../lib/fetcher';
import { withFreshness } from '../../../lib/freshness';
import { serve } from '../../../lib/store';
import { fetchSheetLkg } from '../../../lib/sheetLkg';
import { faultsFrom } from '../../../lib/faults';
import { resolveLeg, buildCopperGold } from '../../../lib/copperGold';
import { resolveSpEps, parseMultplEps, parseShillerCsv, toMonthlyHistory } from '../../../lib/spEps';
import { resolveBankruptcies } from '../../../lib/bankruptcies';
import bakedBankruptcies from '../../../lib/data/bankruptciesBaked.json';
import { cnbcQuotes, cnbcHistory, goldApiSpot, polygonDaily, fredObservations, yahooChart } from '../../../lib/sources';

// In Next 13.5 the route's default fetchCache is 'only-no-store', which ERRORS
// on cached fetches. 'default-cache' permits caching (and never errors), so the
// FRED calls below can use the 30-min Data Cache. Reading the request in GET()
// keeps the handler running per request (fresh fetchedAt, no build-time bake).
export const fetchCache = 'default-cache';

const REVALIDATE_SECONDS = 1800; // 30 minutes
const RETRY_DELAYS_MS = [400, 900, 1800]; // back-off on 429

// Hide the API key if it ever ends up in an error message / URL.
const maskKey = (s) => (typeof s === 'string' ? s.replace(/api_key=[^&\s]+/g, 'api_key=***') : s);

// ─────────────────────────────────────────────────────────────────────────────
// FRED series fetch — 429 back-off, wrapped in a 30-min server cache.
//
// unstable_cache stores the *result* for 30 min regardless of force-dynamic, and
// serves it on a cache hit WITHOUT calling FRED again. That is both the fix for
// the 429 rate-limiting AND the "remember each number for 30 minutes" behavior:
// a throttled load reuses the cached copy instead of blanking out. A thrown error
// (final 429) is NOT cached, so the next load retries.
// ─────────────────────────────────────────────────────────────────────────────

async function fetchSeriesRaw(seriesId, apiKey, limit) {
    const url = `${EXTERNAL_URLS.FRED_BASE}?series_id=${seriesId}&api_key=${apiKey}&file_type=json&sort_order=desc&limit=${limit}`;
    let lastErr;
    for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
        try {
            // Cached in the Data Cache for 30 min (fetchCache='default-cache').
            const data = await fetchJson(url, { revalidate: REVALIDATE_SECONDS });
            return data.observations
                .filter(o => o.value !== '.')
                .map(o => ({ date: o.date, value: parseFloat(o.value) }));
        } catch (e) {
            lastErr = e;
            const is429 = /\b429\b/.test(e?.message || '');
            if (!is429 || attempt === RETRY_DELAYS_MS.length) break;
            await new Promise(r => setTimeout(r, RETRY_DELAYS_MS[attempt]));
        }
    }
    // Re-throw with the key masked so it never leaks into _meta.messages.
    throw new Error(maskKey(lastErr?.message || `Failed to fetch ${seriesId}`));
}

function fetchSeries(seriesId, apiKey, limit = 15) {
    return fetchSeriesRaw(seriesId, apiKey, limit);
}

// Cache the slow HTML P/E scrapes the same way (30 min).
const cachedText = async (key, url, ms) => {
    const res = await proxyFetch(url, { revalidate: REVALIDATE_SECONDS, timeout: ms });
    return res.text();
};

function findByMonthOffset(arr, nMonths) {
    if (!arr?.length) return undefined;
    const target = new Date(arr[0].date);
    target.setMonth(target.getMonth() - nMonths);
    return arr.reduce((best, obs) => {
        const d = Math.abs(new Date(obs.date) - target);
        return d < Math.abs(new Date(best.date) - target) ? obs : best;
    });
}

const dateOf = (arr) => arr?.[0]?.date ?? null;

// ─────────────────────────────────────────────────────────────────────────────
// Copper/Gold ratio — replaces the discontinued LEI. A leading growth/rates gauge.
//
// This route runs on Vercel's serverless functions (datacenter IPs), where Yahoo
// and Stooq are blocked/JS-walled — which is why the old 2-source (Yahoo→Stooq)
// version went permanently N/A. Each leg now cascades through SEVERAL independent,
// datacenter-reachable sources, all normalized to the same unit (copper $/lb, gold
// $/oz) so the ratio (~1.4) is consistent regardless of which source answered:
//
//   COPPER $/lb : CNBC @HG.1 (keyless, daily history) → FRED PCOPPUSDM (key)
//                 → gold-api.com HG (keyless spot) → Yahoo HG=F (self-heal)
//   GOLD   $/oz : CNBC @GC.1 (keyless, daily history) → Polygon C:XAUUSD (key)
//                 → FRED GOLDPMGBD228NLBM (key) → gold-api.com XAU → Yahoo GC=F
//
// Each source can be fault-injected for testing the cascade end-to-end, e.g.
// `?_fail=cg_cnbc` forces both legs past CNBC; `?_fail=cg_cnbc,cg_fred,cg_polygon`
// forces them all the way down to the keyless gold-api spot tier.
//
// Sources that expose history give a real 1-month AND 3-month change (the "trend"
// the owner cares about, not just the level). Spot-only sources still give the
// current ratio. Never throws; a miss → unavailable (the daily health-check flags
// an unexpected N/A here).
// ─────────────────────────────────────────────────────────────────────────────
const LB_PER_TONNE = 2204.6226;     // FRED PCOPPUSDM is USD per metric ton → ÷ this = $/lb

// A "source" knows its fault name, how stale its newest point may be, and how to
// fetch { current, currentDate, historyAsc:[{date,price}] } in the canonical unit.
function copperSources(fredKey) {
    return [
        { name: 'cnbc', freshnessDays: 7, fetch: async () => {
            // Quote is required (gives the current price); history is best-effort (gives the delta).
            const [q, hist] = await Promise.all([cnbcQuotes(['@HG.1']), cnbcHistory('@HG.1').catch((e) => { console.warn(`[CNBC history @HG.1] ${e.message}`); return []; })]);
            const cur = q['@HG.1'];
            const last = hist[hist.length - 1];
            return { current: cur?.price ?? last?.price, currentDate: cur?.asOf ?? last?.date, historyAsc: hist };
        } },
        { name: 'fred', freshnessDays: 80, fetch: async () => {  // monthly series, reported weeks late
            if (!fredKey) throw new Error('no FRED key');
            const obs = await fredObservations('PCOPPUSDM', fredKey, { limit: 60 }); // newest-first $/tonne
            const historyAsc = [...obs].reverse().map((o) => ({ date: o.date, price: o.value / LB_PER_TONNE }));
            const last = historyAsc[historyAsc.length - 1];
            if (!last) throw new Error('FRED PCOPPUSDM: no observations');
            return { current: last.price, currentDate: last.date, historyAsc };
        } },
        { name: 'goldapi', freshnessDays: 7, fetch: async () => {
            const s = await goldApiSpot('HG');               // copper, $/lb (spot, no history)
            return { current: s.current, currentDate: s.asOf, historyAsc: [] };
        } },
        { name: 'yahoo', freshnessDays: 7, fetch: async () => {
            const y = await yahooChart('HG=F', { range: '3mo', revalidate: REVALIDATE_SECONDS });
            return { current: y.current, currentDate: y.history[y.history.length - 1]?.date, historyAsc: y.history };
        } },
    ];
}

function goldSources(fredKey, polyKey) {
    return [
        { name: 'cnbc', freshnessDays: 7, fetch: async () => {
            const [q, hist] = await Promise.all([cnbcQuotes(['@GC.1']), cnbcHistory('@GC.1').catch((e) => { console.warn(`[CNBC history @GC.1] ${e.message}`); return []; })]);
            const cur = q['@GC.1'];
            const last = hist[hist.length - 1];
            return { current: cur?.price ?? last?.price, currentDate: cur?.asOf ?? last?.date, historyAsc: hist };
        } },
        { name: 'polygon', freshnessDays: 12, fetch: async () => {
            if (!polyKey) throw new Error('no Polygon key');
            const p = await polygonDaily('C:XAUUSD', polyKey, { years: 1, revalidate: REVALIDATE_SECONDS });
            const last = p.history[p.history.length - 1];
            if (!last) throw new Error('Polygon C:XAUUSD: no history');
            return { current: p.current, currentDate: last.date, historyAsc: p.history };
        } },
        { name: 'fred', freshnessDays: 12, fetch: async () => {  // daily LBMA fix
            if (!fredKey) throw new Error('no FRED key');
            const obs = await fredObservations('GOLDPMGBD228NLBM', fredKey, { limit: 200 });
            const historyAsc = [...obs].reverse().map((o) => ({ date: o.date, price: o.value }));
            const last = historyAsc[historyAsc.length - 1];
            if (!last) throw new Error('FRED GOLDPMGBD228NLBM: no observations');
            return { current: last.price, currentDate: last.date, historyAsc };
        } },
        { name: 'goldapi', freshnessDays: 7, fetch: async () => {
            const s = await goldApiSpot('XAU');              // gold, $/oz (spot, no history)
            return { current: s.current, currentDate: s.asOf, historyAsc: [] };
        } },
        { name: 'yahoo', freshnessDays: 7, fetch: async () => {
            const y = await yahooChart('GC=F', { range: '3mo', revalidate: REVALIDATE_SECONDS });
            return { current: y.current, currentDate: y.history[y.history.length - 1]?.date, historyAsc: y.history };
        } },
    ];
}

// ─────────────────────────────────────────────────────────────────────────────
// S&P 500 EPS (TTM, as-reported — the E in P/E). Three independent sources so
// the card always has a backup (same philosophy as copper/gold above):
//
//   1. multpl  — the by-month earnings table (level + monthly history to 1871,
//                inflation-adjusted to current dollars; lags ~2-3 quarters like
//                all as-reported TTM series, so its freshness window is 400d)
//   2. derived — FRED SP500 index level ÷ the live TTM P/E scraped above.
//                Spot-only but updates daily; skipped when the P/E fell back to
//                CAPE (10-yr smoothed earnings → dividing by it isn't TTM EPS)
//   3. datahub — GitHub-raw mirror of Shiller's dataset ('Real Earnings' column,
//                matching multpl's units). Ultra-reliable CDN, but its earnings
//                run years behind → in practice the graceful-staleness fallback
//                that keeps the CHART rendering when multpl is down.
//
// Fault gates for testing: ?_fail=eps_multpl / eps_derived / eps_datahub.
// resolveSpEps is pure (lib/spEps.js) and never throws.
// ─────────────────────────────────────────────────────────────────────────────
function spEpsSources({ fredKey, pe, peSource }) {
    return [
        { name: 'multpl', freshnessDays: 400, fetch: async () => {
            const html = await cachedText('multpl-eps', EXTERNAL_URLS.MULTPL_EPS, 8000);
            const hist = parseMultplEps(html);
            const last = hist[hist.length - 1];
            if (!last) throw new Error('multpl EPS: no rows parsed');
            return { current: last.value, currentDate: last.date, historyAsc: hist };
        } },
        { name: 'derived', freshnessDays: 7, fetch: async () => {
            if (!pe || peSource === 'cape') throw new Error('derived EPS: no TTM P/E to divide by');
            if (!fredKey) throw new Error('derived EPS: no FRED key');
            const spx = await fredObservations('SP500', fredKey, { limit: 5 }); // newest-first daily closes
            return { current: spx[0].value / pe, currentDate: spx[0].date, historyAsc: [] };
        } },
        { name: 'datahub', freshnessDays: 400, fetch: async () => {
            const csv = await cachedText('datahub-eps', EXTERNAL_URLS.DATAHUB_SHILLER, 8000);
            const hist = parseShillerCsv(csv);
            const last = hist[hist.length - 1];
            if (!last) throw new Error('datahub EPS: no rows parsed');
            return { current: last.value, currentDate: last.date, historyAsc: hist };
        } },
    ];
}

// Resolve both legs (each cascades through its sources via resolveLeg) and build
// the ratio + 1mo/3mo trend. resolveLeg + buildCopperGold live in lib/copperGold
// (pure, unit-tested); here we only wire the network-bound sources + keys.
async function fetchCopperGold(now, { fredKey, polyKey, faults }) {
    const [copper, gold] = await Promise.all([
        resolveLeg(copperSources(fredKey), faults, now),
        resolveLeg(goldSources(fredKey, polyKey), faults, now),
    ]);
    return buildCopperGold(copper, gold);
}

// ─────────────────────────────────────────────────────────────────────────────
// Build the dashboard payload, stamping each metric with asOf + staleness.
// A value older than its series' freshness deadline (or missing) becomes null,
// so the UI shows N/A — never a misleadingly old number.
// ─────────────────────────────────────────────────────────────────────────────

function buildResponse(series, peRatio, now) {
    const {
        T10Y2Y: t10y2y, UNRATE: unrate, UMCSENT: umcsent, ICSA: icsa, BAMLC0A4CBBB: bbb,
        DFII10: dfii10, NFCI: nfci, M2SL: m2sl, RSXFS: rsxfs,
        HOUST: houst, INDPRO: indpro, JTSJOL: jtsjol, DGORDER: dgorder, PSAVERT: psavert,
        A053RC1Q027SBEA: corpProfits, GDP: gdpData, USREC: usrec,
    } = series;

    const F = (value, seriesId, asOf, extra = {}) => ({
        ...withFreshness(value, asOf, FRED_FRESHNESS[seriesId], now),
        ...extra,
    });

    // Recession shading
    const recessionPeriods = [];
    const recSorted = [...usrec].reverse();
    let recStart = null;
    for (let i = 0; i < recSorted.length; i++) {
        if (recSorted[i].value === 1 && recStart === null) recStart = recSorted[i].date;
        else if (recSorted[i].value === 0 && recStart !== null) {
            recessionPeriods.push({ start: recStart, end: recSorted[i].date });
            recStart = null;
        }
    }
    if (recStart !== null) recessionPeriods.push({ start: recStart, end: recSorted[recSorted.length - 1].date });

    // ── Top cards ──
    const yc = withFreshness(t10y2y[0]?.value, dateOf(t10y2y), FRED_FRESHNESS.T10Y2Y, now);
    const yieldCurve = { current: yc.value, asOf: yc.asOf, stale: yc.stale, date: t10y2y[0]?.date, history: [...t10y2y].reverse() };

    const gdpMap = new Map();
    for (const gd of gdpData) gdpMap.set(gd.date, gd.value);
    const profitMarginHistory = [];
    for (const cp of corpProfits) {
        const gdpValue = gdpMap.get(cp.date);
        if (gdpValue && gdpValue !== 0) profitMarginHistory.push({ date: cp.date, value: (cp.value / gdpValue) * 100 });
    }
    const pm = withFreshness(profitMarginHistory[0]?.value, profitMarginHistory[0]?.date, FRED_FRESHNESS.A053RC1Q027SBEA, now);
    const profitMargin = { current: pm.value, asOf: pm.asOf, stale: pm.stale, date: profitMarginHistory[0]?.date || '', history: [...profitMarginHistory].reverse() };

    // ── Economic indicators ──
    const unrate3mo = unrate.length >= 3 ? unrate.slice(0, 3).reduce((s, v) => s + v.value, 0) / 3 : null;
    // 12-month low MUST slice: unrate now carries FULL history for the Horsemen
    // chart, and a min over all of it would break the Sahm rule.
    const unrate12moLow = unrate.length > 0 ? Math.min(...unrate.slice(0, 12).map(u => u.value)) : null;
    const sahmRule = unrate3mo !== null && unrate12moLow !== null ? unrate3mo - unrate12moLow : undefined;

    const sentimentCurrent = umcsent[0]?.value;
    const sentimentPrev = umcsent[1]?.value;
    const claims4wk = icsa.length >= 4 ? icsa.slice(0, 4).reduce((s, v) => s + v.value, 0) / 4 : undefined;
    const bbbCurrent = bbb[0]?.value;
    const tipsCurrent = dfii10[0]?.value;
    const tipsPrev = dfii10[1]?.value;
    // (LEI/USSLIND removed — discontinued since 2020. Replaced by the Copper/Gold ratio,
    //  fetched from multiple price sources and merged into `indicators` in GET().)

    // ── Bull checklist ──
    const nfciCurrent = nfci[0]?.value;
    const m2Current = m2sl[0]?.value;
    const m2YearAgo = findByMonthOffset(m2sl, 12)?.value;
    const m2Growth = m2YearAgo ? ((m2Current - m2YearAgo) / m2YearAgo) * 100 : undefined;
    const retailCurrent = rsxfs[0]?.value;
    const retail3mo = findByMonthOffset(rsxfs, 3)?.value;
    const retailGrowth = retail3mo ? ((retailCurrent - retail3mo) / retail3mo) * 100 : undefined;
    const housingCurrent = houst[0]?.value;
    const housing6moAvg = houst.length >= 6 ? houst.slice(0, 6).reduce((s, v) => s + v.value, 0) / 6 : 0;
    const indproCurrent = indpro[0]?.value;
    const indpro6mo = findByMonthOffset(indpro, 6)?.value;
    const indproChange = indpro6mo ? ((indproCurrent - indpro6mo) / indpro6mo) * 100 : undefined;
    const joltsCurrent = jtsjol[0]?.value;
    const durableCurrent = dgorder[0]?.value;
    const durable3mo = findByMonthOffset(dgorder, 3)?.value;
    const durableChange = durable3mo ? ((durableCurrent - durable3mo) / durable3mo) * 100 : undefined;
    const savingsCurrent = psavert[0]?.value;

    // ── Four Horsemen (recession watch card) ──
    // Claims + unemployment ride along from the already-fetched FRED series with
    // full ascending histories; the yield-curve panel reuses `yieldCurve` above;
    // bankruptcies (non-FRED) is resolved in GET() and overwrites its placeholder,
    // exactly like copperGold.
    const claimsFresh = withFreshness(icsa[0]?.value, dateOf(icsa), FRED_FRESHNESS.ICSA, now);
    const unrateFresh = withFreshness(unrate[0]?.value, dateOf(unrate), FRED_FRESHNESS.UNRATE, now);
    const horsemen = {
        claims: { current: claimsFresh.value, asOf: claimsFresh.asOf, stale: claimsFresh.stale, staleDays: claimsFresh.staleDays, unavailable: claimsFresh.unavailable, history: [...icsa].reverse() },
        unemployment: { current: unrateFresh.value, asOf: unrateFresh.asOf, stale: unrateFresh.stale, staleDays: unrateFresh.staleDays, unavailable: unrateFresh.unavailable, history: [...unrate].reverse() },
        bankruptcies: { current: null, total: null, asOf: null, stale: false, staleDays: 0, unavailable: true, changePct: null, status: 'unknown', source: null, history: [], tried: [] },
    };

    return {
        yieldCurve,
        profitMargin,
        horsemen,
        peRatio,
        peRatioAsOf: now.toISOString(), // scraped live each cache cycle
        recessions: recessionPeriods,
        indicators: {
            sahmRule: F(sahmRule, 'UNRATE', dateOf(unrate), { status: sahmRule >= 0.5 ? 'danger' : 'safe' }),
            sentiment: F(sentimentCurrent, 'UMCSENT', dateOf(umcsent), { change: sentimentCurrent - sentimentPrev, status: sentimentCurrent > 80 ? 'strong' : sentimentCurrent > 60 ? 'neutral' : 'weak' }),
            claims: F(claims4wk !== undefined ? claims4wk / 1000 : undefined, 'ICSA', dateOf(icsa), { status: claims4wk < 250000 ? 'healthy' : claims4wk < 350000 ? 'elevated' : 'weak' }),
            creditSpread: F(bbbCurrent, 'BAMLC0A4CBBB', dateOf(bbb), { status: bbbCurrent < 1.5 ? 'tight' : bbbCurrent < 2.5 ? 'normal' : 'stressed' }),
            realYields: F(tipsCurrent, 'DFII10', dateOf(dfii10), { change: tipsCurrent - tipsPrev, status: tipsCurrent > 2.0 ? 'restrictive' : tipsCurrent > 0 ? 'neutral' : 'easy' }),
            // copperGold is added in GET() after buildResponse (it's fetched from price
            // sources, not FRED). Placeholder keeps the key order stable; GET overwrites it.
            copperGold: { value: null, asOf: null, stale: false, unavailable: true, status: 'unknown', change: null, changePct: null, change3mo: null, changePct3mo: null, copper: null, gold: null, source: null },
        },
        checklist: {
            nfci: F(nfciCurrent, 'NFCI', dateOf(nfci), { bullish: nfciCurrent < 0, status: nfciCurrent < -0.5 ? 'strong' : nfciCurrent < 0 ? 'good' : 'weak', label: 'Financial Conditions' }),
            m2: F(m2Growth, 'M2SL', dateOf(m2sl), { bullish: m2Growth > 2.0, status: m2Growth > 4.0 ? 'strong' : m2Growth > 2.0 ? 'good' : 'weak', label: 'M2 Money Supply' }),
            retail: F(retailGrowth, 'RSXFS', dateOf(rsxfs), { bullish: retailGrowth > 0, status: retailGrowth > 1.0 ? 'strong' : retailGrowth > 0 ? 'good' : 'weak', label: 'Retail Sales (3mo)' }),
            housing: F(housingCurrent, 'HOUST', dateOf(houst), { bullish: housingCurrent > housing6moAvg && housingCurrent > 1300, status: housingCurrent > 1400 ? 'strong' : (housingCurrent > housing6moAvg && housingCurrent > 1300) ? 'good' : 'weak', label: 'Housing Starts' }),
            indpro: F(indproChange, 'INDPRO', dateOf(indpro), { bullish: indproChange > 0, status: indproChange > 1.0 ? 'strong' : indproChange > 0 ? 'good' : 'weak', label: 'Industrial Production' }),
            jolts: F(joltsCurrent, 'JTSJOL', dateOf(jtsjol), { bullish: joltsCurrent > 6000, status: joltsCurrent > 7000 ? 'strong' : joltsCurrent > 6000 ? 'good' : 'weak', label: 'Job Openings (JOLTS)' }),
            durable: F(durableChange, 'DGORDER', dateOf(dgorder), { bullish: durableChange > 0, status: durableChange > 2.0 ? 'strong' : durableChange > 0 ? 'good' : 'weak', label: 'Durable Goods Orders' }),
            savings: F(savingsCurrent, 'PSAVERT', dateOf(psavert), { bullish: savingsCurrent >= 3.5, status: savingsCurrent >= 5.0 ? 'strong' : savingsCurrent >= 3.5 ? 'good' : 'weak', label: 'Savings Rate' }),
        },
    };
}

// ─────────────────────────────────────────────────────────────────────────────

const FRED_REQUESTS = [
    [FRED_SERIES.YIELD_CURVE, 100000],
    // UNRATE + ICSA fetch FULL history (not just the newest points) because the
    // Four Horsemen card charts them; the Sahm/4-wk math below slices what it needs.
    [FRED_SERIES.UNEMPLOYMENT, 100000],
    [FRED_SERIES.SENTIMENT, 5],
    [FRED_SERIES.CLAIMS, 100000],
    [FRED_SERIES.CREDIT_SPREAD, 252],
    [FRED_SERIES.REAL_YIELDS, 5],
    [FRED_SERIES.NFCI, 5],
    [FRED_SERIES.M2_MONEY, 15],
    [FRED_SERIES.RETAIL_SALES, 5],
    [FRED_SERIES.HOUSING_STARTS, 10],
    [FRED_SERIES.INDUSTRIAL_PROD, 10],
    [FRED_SERIES.JOLTS, 5],
    [FRED_SERIES.DURABLE_GOODS, 5],
    [FRED_SERIES.SAVINGS_RATE, 5],
    [FRED_SERIES.CORP_PROFITS, 100000],
    [FRED_SERIES.GDP, 100000],
    [FRED_SERIES.RECESSIONS, 100000],
];

export async function GET(request) {
    // Touch the request so Next renders this handler dynamically (per request),
    // while individual FRED fetches still come from the 30-min Data Cache.
    request.headers.get('user-agent');

    const faults = faultsFrom(request);
    const apiKey = process.env.FRED_API_KEY;
    // `?_fail=fred` forces a bad key so every series fails (tests last-good path).
    const liveKey = faults.has('fred') ? 'INVALID_INJECTED' : apiKey;

    // EVERYTHING runs inside serve(): any throw (missing key, all series failing,
    // buildResponse error, etc.) is caught -> last-known-good -> safe default.
    // The route can never 500 or return an empty body.
    return serve('fred', async () => {
        if (!apiKey) throw new Error('FRED_API_KEY not configured');
        const now = new Date();

        // Fetch in small batches with a short stagger to be polite to FRED on a
        // cold cache. Cache hits resolve instantly.
        const BATCH_SIZE = 4;
        const STAGGER_MS = 150;
        const settled = [];
        for (let i = 0; i < FRED_REQUESTS.length; i += BATCH_SIZE) {
            const batch = FRED_REQUESTS.slice(i, i + BATCH_SIZE).map(([id, limit]) =>
                fetchSeries(id, liveKey, limit)
                    .then(value => ({ id, status: 'fulfilled', value }))
                    .catch(e => ({ id, status: 'rejected', reason: e })),
            );
            settled.push(...(await Promise.all(batch)));
            if (i + BATCH_SIZE < FRED_REQUESTS.length) await new Promise(r => setTimeout(r, STAGGER_MS));
        }

        const series = {};
        const failed = [];
        for (const s of settled) {
            series[s.id] = s.status === 'fulfilled' && s.value?.length ? s.value : [];
            if (s.status === 'rejected' || !s.value?.length) failed.push(s.id);
        }

        const messages = [`Loaded ${FRED_REQUESTS.length - failed.length}/${FRED_REQUESTS.length} series`];
        for (const s of settled) {
            if (s.status === 'rejected') messages.push(`Series failed: ${maskKey(s.reason?.message || s.id)}`);
        }
        if (failed.length) console.warn(`[FRED] ${failed.length} series unavailable this load:`, failed.join(', '));

        // P/E ratio — layered, cached scrapes. peSource records which layer won so
        // the derived-EPS leg below can refuse CAPE (a 10-yr smoothed P/E).
        let peRatio = null;
        let peSource = null;
        try {
            const peHtml = await cachedText('multpl', EXTERNAL_URLS.MULTPL_PE, 8000);
            const m = peHtml.match(/Current S&P 500 PE Ratio[^\d]*(\d+\.\d+)/);
            if (m) { peRatio = parseFloat(m[1]); peSource = 'multpl'; }
        } catch (e) { messages.push(`P/E multpl failed: ${maskKey(e.message)}`); }
        if (!peRatio) {
            try {
                const yHtml = await cachedText('yahoo-pe', EXTERNAL_URLS.YAHOO_PE, 8000);
                const m = yHtml.match(/PE Ratio \(TTM\)[\s\S]*?(\d+\.\d+)/i);
                if (m) { peRatio = parseFloat(m[1]) * 1.07; peSource = 'yahoo'; }
            } catch (e) { messages.push(`P/E Yahoo failed: ${maskKey(e.message)}`); }
        }
        if (!peRatio) {
            try {
                const cape = await fetchSeries('PE10', liveKey, 3);
                if (cape.length > 0) { peRatio = cape[0].value; peSource = 'cape'; }
            } catch (e) { messages.push(`P/E CAPE failed: ${maskKey(e.message)}`); }
        }

        const responseData = buildResponse(series, peRatio, now);

        // Copper/Gold ratio comes from price sources (not FRED series). Guarded so it
        // can never break the FRED data: a failure just leaves it unavailable (which
        // the daily health-check then flags as an unexpected N/A). The leg cascade gates
        // Polygon on `cg_polygon` internally, so pass the real key here.
        const polyKey = process.env.POLYGON_KEY || '';
        try {
            const cg = await fetchCopperGold(now, { fredKey: apiKey, polyKey, faults });
            responseData.indicators.copperGold = cg;
            messages.push(`Copper/Gold: ${cg.source || 'unavailable'}`);
        } catch (e) {
            messages.push(`Copper/Gold failed: ${maskKey(e.message)}`);
        }

        // S&P 500 EPS (TTM) — guarded the same way: a failure leaves the card N/A
        // but can never break the FRED data. resolveSpEps itself never throws; the
        // try/catch is belt-and-braces around the source construction.
        responseData.spEps = { current: null, asOf: null, stale: false, unavailable: true, source: null, historySource: null, history: [], tried: [] };
        try {
            const eps = await resolveSpEps(spEpsSources({ fredKey: apiKey, pe: peRatio, peSource }), faults, now);
            responseData.spEps = { ...eps, history: toMonthlyHistory(eps.history) };
            messages.push(`S&P EPS: ${eps.source || 'unavailable'}`);
        } catch (e) {
            messages.push(`S&P EPS failed: ${maskKey(e.message)}`);
        }

        // US Bankruptcies (Four Horsemen card) — AOUSC F-2, not a FRED series.
        // Guarded like copperGold/spEps: live uscourts XLSX → baked history JSON
        // (lib/data/bankruptciesBaked.json); a total failure leaves the placeholder
        // (N/A panel) and can never break the FRED payload. Gates: bk_uscourts, bk_baked.
        try {
            const bk = await resolveBankruptcies({
                now,
                faults,
                fetchBuffer: async (url) => Buffer.from(await (await proxyFetch(url, { revalidate: REVALIDATE_SECONDS, timeout: 7000 })).arrayBuffer()),
                fetchText: (url) => cachedText(`uscourts-${url.slice(-24)}`, url, 7000),
                baked: bakedBankruptcies,
            });
            responseData.horsemen.bankruptcies = bk;
            messages.push(`Bankruptcies: ${bk.source || 'unavailable'}`);
        } catch (e) {
            messages.push(`Bankruptcies failed: ${maskKey(e.message)}`);
        }

        return {
            ...responseData,
            _meta: {
                source: 'St. Louis Fed',
                hasErrors: failed.length > 0,
                fetchedAt: now.toISOString(),
                messages,
                loadedCount: FRED_REQUESTS.length - failed.length,
            },
        };
    }, {
        // Good only if at least one series loaded; a total outage (0/18) falls
        // through to last-known-good instead of N/A everywhere.
        isGood: (p) => p && p._meta && p._meta.loadedCount > 0,
        fallback: { error: 'FRED temporarily unavailable' },
        // Last resort (only if live FRED AND the /tmp last-good are both gone): the
        // financial-dashboard-history sheet's `dashboard_lkg` helper tab. See lib/sheetLkg.js.
        lastResort: () => fetchSheetLkg(new Date()),
        faults,
    });
}
