/**
 * Independent keyless backup tiers + the per-metric last-known-good merge for
 * /api/market-extra (route: app/api/market-extra/route.js).
 *
 * Why: below the Lambda, oil / 10Y / 2Y / 30Y mortgage had ONE backup (FRED), USD/BDT
 * had one (ER-API — Frankfurter has no BDT), DXY was only ever COMPUTED from an FX
 * basket, and when every live tier died a lone gold-api print passed `isGood`, so the
 * full 13-metric last-known-good was never served (12 blanks instead of yesterday).
 *
 * Tiers here (each behind a `?_fail=` switch, checked by the route):
 *   10Y / 2Y      US Treasury daily yield-curve CSV (the ORIGIN of FRED's DGS10/DGS2)
 *   30Y mortgage  Freddie Mac PMMS_history.csv (the ORIGIN of FRED's MORTGAGE30US)
 *   WTI oil       CNBC quote + daily history @CL.1 (front-month future)
 *   DXY           CNBC .DXY (the real ICE index) — the computed basket stays LAST
 *   USD/BDT       Fawaz currency-api (jsDelivr → Cloudflare Pages mirror)
 *
 * Wrong data is worse than blank: every tier rejects a print older than its
 * freshness window instead of passing it off as today's, and the last-good merge
 * stamps each borrowed metric `stale: true` with the time it was really fetched.
 * Nothing here throws to the caller except via a rejected promise the route catches.
 */

import { cnbcQuotes, cnbcHistory, treasuryYieldCurveCsv, dailyChange } from './sources';
import { fetchJson, fetchText } from './fetcher';
import { splitCsvLine } from './horsemen';
import { loadLastGood } from './store';

export const HIST = 260;
const DAY_MS = 864e5;

// How old the newest print may be before a tier refuses it (calendar days). Daily
// series allow a long weekend + a holiday; PMMS is weekly (Thursdays); Fawaz is daily.
export const FRESH_DAYS = { treasury: 6, pmms: 13, cnbc: 6, fawaz: 4 };

const finite = (n) => Number.isFinite(n);
const ageDays = (date, now) => (now - Date.parse(`${date}T12:00:00Z`)) / DAY_MS;

/** Ascending [{date, price}] → a market-extra metric, or null if empty / too old. */
export function seriesMetric(asc, { freshDays, now = Date.now(), hist = HIST } = {}) {
    if (!Array.isArray(asc) || !asc.length) return null;
    const last = asc[asc.length - 1];
    if (!finite(last?.price) || !last.date) return null;
    if (freshDays != null && ageDays(last.date, now) > freshDays) return null;
    const prev = asc[asc.length - 2]?.price ?? last.price;
    const history = asc.slice(-hist);
    return { current: last.price, dailyChange: dailyChange(last.price, prev), history, lastDate: last.date };
}

/** Sort ascending by ISO date, last write wins on a duplicate date. */
function ascUnique(points) {
    const m = new Map();
    for (const p of points) m.set(p.date, p);
    return [...m.values()].sort((a, b) => (a.date < b.date ? -1 : 1));
}

const usDate = (s) => {
    const m = String(s || '').trim().match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
    return m ? `${m[3]}-${m[1].padStart(2, '0')}-${m[2].padStart(2, '0')}` : null;
};

/**
 * Treasury daily yield-curve CSV(s) → { tnx, t2y } ascending [{date, price}].
 * Columns found BY HEADER NAME ("2 Yr" / "10 Yr" exactly — never "20 Yr"), as in
 * lib/horsemen.js parseTreasuryCsv; values are Treasury's own 2dp = FRED's DGS10/DGS2.
 */
export function parseTreasuryTenors(...csvs) {
    const tnx = [], t2y = [];
    for (const csv of csvs) {
        if (!csv || typeof csv !== 'string') continue;
        const lines = csv.trim().split(/\r?\n/).filter((l) => l.trim());
        if (lines.length < 2) continue;
        const header = splitCsvLine(lines[0]).map((h) => h.trim().toLowerCase());
        const di = header.findIndex((h) => h === 'date');
        const idx = (n) => header.findIndex((h) => new RegExp(`^${n}\\s*yr$`).test(h));
        const i2 = idx(2), i10 = idx(10);
        if (di < 0 || i2 < 0 || i10 < 0) continue;
        for (const line of lines.slice(1)) {
            const f = splitCsvLine(line);
            const date = usDate(f[di]);
            if (!date) continue;
            const y2 = parseFloat(f[i2]), y10 = parseFloat(f[i10]);
            if (finite(y10)) tnx.push({ date, price: y10 });
            if (finite(y2)) t2y.push({ date, price: y2 });
        }
    }
    return { tnx: ascUnique(tnx), t2y: ascUnique(t2y) };
}

/** Freddie Mac PMMS_history.csv → ascending weekly 30Y fixed [{date, price}]. */
export function parsePmmsCsv(csv) {
    if (!csv || typeof csv !== 'string') return [];
    const lines = csv.trim().split(/\r?\n/).filter((l) => l.trim());
    if (lines.length < 2) return [];
    const header = splitCsvLine(lines[0]).map((h) => h.trim().toLowerCase());
    const di = header.indexOf('date'), ci = header.indexOf('pmms30');
    if (di < 0 || ci < 0) return [];
    const out = [];
    for (const line of lines.slice(1)) {
        const f = splitCsvLine(line);
        const date = usDate(f[di]);
        const v = parseFloat(f[ci]);
        if (date && finite(v)) out.push({ date, price: v });
    }
    return ascUnique(out);
}

/**
 * CNBC quote (today's price) + CNBC daily bars (history, lags a session) → metric.
 * The quote point is appended/replaces its date so `current` IS the last history bar
 * (marketWindow reads the change from the last two bars). The change is ALWAYS taken
 * from the prior bar — CNBC's own .DXY `change` field reads 0.00 all day.
 */
export function cnbcMetricFrom(quote, bars, { freshDays = FRESH_DAYS.cnbc, now = Date.now() } = {}) {
    let asc = Array.isArray(bars) ? bars.filter((b) => b?.date && finite(b.price)) : [];
    if (quote && finite(quote.price) && quote.asOf) asc = ascUnique([...asc, { date: quote.asOf, price: quote.price }]);
    return seriesMetric(asc, { freshDays, now });
}

/** Fawaz currency-api payload → { current, asOf } for one USD pair, or throws. */
export function fawazPair(json, code, { freshDays = FRESH_DAYS.fawaz, now = Date.now() } = {}) {
    const v = Number(json?.usd?.[code]);
    const asOf = typeof json?.date === 'string' ? json.date.slice(0, 10) : null;
    if (!finite(v) || v <= 0) throw new Error(`Fawaz: no usd/${code}`);
    if (!asOf || ageDays(asOf, now) > freshDays) throw new Error(`Fawaz: stale (${asOf})`);
    return { current: v, asOf };
}

// The CORRECT Cloudflare mirror is latest.currency-api.pages.dev (lib/sources.js
// fawazRates uses `<base>.currency-api.pages.dev`, which serves an HTML error page).
export const FAWAZ_URLS = [
    'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json',
    'https://latest.currency-api.pages.dev/v1/currencies/usd.json',
];

/**
 * Per-request lazy fetchers: each network call runs at most once and only if some
 * metric actually falls through to it (10Y + 2Y share one Treasury download, oil +
 * DXY one CNBC quote call). Short timeouts, one try — these sit beneath FRED /
 * ER-API inside the route's fill deadline.
 */
export function makeTiers({ now = () => Date.now() } = {}) {
    const once = (fn) => { let p; return () => (p ??= Promise.resolve().then(fn)); };
    const treasury = once(async () => {
        const y = new Date(now()).getUTCFullYear();
        // Two calendar years: early January the current file is nearly empty.
        const [cur, prev] = await Promise.allSettled([
            treasuryYieldCurveCsv(y, { revalidate: 1800, timeout: 6000 }),
            treasuryYieldCurveCsv(y - 1, { revalidate: 86400, timeout: 6000 }),
        ]);
        const t = parseTreasuryTenors(prev.status === 'fulfilled' ? prev.value : '', cur.status === 'fulfilled' ? cur.value : '');
        if (!t.tnx.length && !t.t2y.length) throw new Error('Treasury: no yield rows');
        return t;
    });
    const pmms = once(async () => {
        const csv = await fetchText('https://www.freddiemac.com/pmms/docs/PMMS_history.csv', { revalidate: 21600, timeout: 6000 });
        const s = parsePmmsCsv(csv);
        if (!s.length) throw new Error('PMMS: no rows');
        return s;
    });
    const quotes = once(() => cnbcQuotes(['@CL.1', '.DXY'], { revalidate: 300, timeout: 6000, tries: 1 }));
    const bars = {};
    const history = (symbol) => (bars[symbol] ??= cnbcHistory(symbol, { revalidate: 1800, timeout: 6000, tries: 1 }).catch(() => []));
    const cnbc = async (symbol) => {
        const [q, h] = await Promise.all([quotes().then((m) => m[symbol] || null).catch(() => null), history(symbol)]);
        const m = cnbcMetricFrom(q, h, { now: now() });
        if (!m) throw new Error(`CNBC: no fresh ${symbol}`);
        return m;
    };
    const fawaz = once(async () => {
        let last;
        for (const u of FAWAZ_URLS) {
            try { return await fetchJson(u, { revalidate: 600, timeout: 5000 }); } catch (e) { last = e; }
        }
        throw new Error(`Fawaz: ${last?.message || 'unavailable'}`);
    });
    return {
        tnx: async () => seriesMetric((await treasury()).tnx, { freshDays: FRESH_DAYS.treasury, now: now() }),
        t2y: async () => seriesMetric((await treasury()).t2y, { freshDays: FRESH_DAYS.treasury, now: now() }),
        mortgage: async () => seriesMetric(await pmms(), { freshDays: FRESH_DAYS.pmms, now: now() }),
        oil: () => cnbc('@CL.1'),
        dxy: () => cnbc('.DXY'),
        usdbdt: async () => fawazPair(await fawaz(), 'bdt', { now: now() }),
    };
}

// ── Per-metric last-known-good merge ────────────────────────────────────────────

const getPath = (o, p) => p.split('.').reduce((a, k) => (a ? a[k] : undefined), o);
const setPath = (o, p, v) => { const ks = p.split('.'); const last = ks.pop(); let cur = o; for (const k of ks) cur = cur[k] ??= {}; cur[last] = v; };
const has = (o, p) => { const m = getPath(o, p); return !!m && m.current != null; };

/**
 * Fill each metric the live build is missing from the last-known-good copy (read
 * through lib/store.js loadLastGood, sync or async — so any durable tier added there
 * benefits). Each borrowed metric is stamped `stale: true` + `savedAt` (when it was
 * REALLY fetched: a metric that was itself borrowed keeps its original savedAt, so a
 * re-save can't launder it fresh) and is skipped past `maxStaleMs`.
 *
 * Never fills in fault-test mode with `lastgood` in the fault set. Never throws.
 * Mutates `out`; returns the filled metric names.
 */
export async function mergeLastGood(out, { key, paths, faults = new Set(), maxStaleMs = 7 * DAY_MS, now = Date.now() } = {}) {
    try {
        if (!out || !paths?.length || (faults && faults.has('lastgood'))) return [];
        const missing = paths.filter((p) => !has(out, p));
        if (!missing.length) return [];
        const lg = await loadLastGood(key);
        if (!lg?.data) return [];
        const filled = [];
        const log = ((out._meta ??= {}).sourceLog ??= {});
        for (const p of missing) {
            const m = getPath(lg.data, p);
            if (!m || m.current == null || !finite(Number(m.current))) continue;
            const savedAt = (m.stale && m.savedAt) || lg.savedAt;
            const t = Date.parse(savedAt);
            if (!finite(t) || now - t > maxStaleMs) continue;
            setPath(out, p, { ...m, stale: true, savedAt });
            const name = p.split('.').pop();
            log[name] = `last-known-good ${savedAt}`;
            filled.push(name);
        }
        return filled;
    } catch { return []; }
}
