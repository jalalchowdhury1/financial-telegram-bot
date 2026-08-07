/**
 * Volatility metrics (IV rank / IV percentile / VRP) for the vol table.
 *
 * IV is proxied by CBOE index levels — no option-chain math is possible from
 * Vercel (Yahoo blocks datacenter IPs), and this is the same proxy method the
 * owner's hedgelab tool uses:
 *   SPY → VIX ·  QQQ → VXN ·  TQQQ/SQQQ → 3×VXN (leverage scales IV ~linearly
 *   by arbitrage; Leung & Sircar) ·  UVXY → VVIX (vol-of-vol).
 * Rank/percentile are computed on the UNSCALED index series — a constant
 * multiplier changes neither rank nor percentile — while the displayed IV level
 * is scaled. VRP = proxied IV − 21d realized vol of the ETF itself (annualized,
 * vol points). All of it is an approximation and the UI says so.
 */

import { isStale } from './freshness';

export const VOL_PROXIES = {
    SPY: { index: 'VIX', mult: 1 },
    QQQ: { index: 'VXN', mult: 1 },
    TQQQ: { index: 'VXN', mult: 3 },
    SQQQ: { index: 'VXN', mult: 3 },
    UVXY: { index: 'VVIX', mult: 1 },
};

const ONE_YEAR = 252; // trading days

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/; // guard: lexicographic date compare is only safe on this shape

/**
 * Default staleness deadline for every vol source, in days.
 *
 * Everything this route reads is a DAILY series (CBOE/CNBC/FRED index closes,
 * ETF daily bars), so the only legitimate gap is a weekend plus holidays — the
 * same reasoning behind FRED_FRESHNESS.T10Y2Y = 7. A source still serving the
 * same last point a week later is frozen, not quiet.
 */
export const VOL_FRESHNESS_DAYS = 7;

/**
 * Resolve one vol series through a source cascade.
 *
 * SAME CONTRACT AS copperGold's `resolveLeg` (and the horsemen cascades) —
 * deliberately not a new pattern: skip fault-injected sources, reject results
 * that are empty, and REJECT any series whose newest point is staler than the
 * source allows, falling through to the next tier. Returns the first healthy
 * series with a `tried` trail, or a null series if every source failed.
 *
 * WHY THE STALENESS GATE EXISTS (added 2026-08-06): this cascade previously
 * accepted any array with `length > 0`. A CBOE CDN CSV that stops updating —
 * or a CNBC endpoint that starts replaying an old window — would win tier 1
 * forever and the table would present months-old vol as today's number, with
 * `_meta.hasErrors:false`. Every sibling cascade in this repo already rejected
 * stale sources; this one was the exception.
 *
 * A source descriptor is: { name, gate, freshnessDays?, fetch: async () =>
 *   ascending [{date:'YYYY-MM-DD', value:number}] }.
 *
 * @returns {{points: Array|null, source: string|null, asOf: string|null, tried: string[]}}
 */
export async function resolveVolSeries(sources, faults, now = new Date()) {
    const tried = [];
    for (const s of sources) {
        if (faults && faults.has(s.gate)) { tried.push(`${s.name}:off`); continue; }
        try {
            const points = await s.fetch();
            const last = Array.isArray(points) && points.length ? points[points.length - 1] : null;
            if (!last || !last.date || !Number.isFinite(last.value)) { tried.push(`${s.name}:empty`); continue; }
            if (isStale(last.date, s.freshnessDays ?? VOL_FRESHNESS_DAYS, now)) {
                tried.push(`${s.name}:stale(${last.date})`);
                continue;
            }
            tried.push(`${s.name}:ok`);
            return { points, source: s.name, asOf: last.date, tried };
        } catch (e) { tried.push(`${s.name}:err`); }
    }
    return { points: null, source: null, asOf: null, tried };
}

/**
 * Tickers whose row is not actually usable — `iv` or `rv21` is null.
 *
 * This is what `_meta.hasErrors` must be derived from. The old test was
 * `hasErrors: !anyData`, i.e. true ONLY when every cell of every row was null:
 * a permanently dead VVIX cascade (VVIX has NO FRED tier, so it is the most
 * fragile of the three indices) nulled UVXY's entire row while the endpoint
 * still reported itself green — and /api/vol gets no equivalent of
 * check_indicators_na, so its own `_meta` is the only signal there is.
 *
 * A served value is never stale: `resolveVolSeries` rejects stale sources, so
 * null is the only way a cell can be wrong.
 */
export function volIncompleteTickers(tickers) {
    return (tickers || []).filter((t) => t && (t.iv == null || t.rv21 == null)).map((t) => t.ticker);
}

/**
 * Parse a CBOE daily-prices CSV. Two live schemas:
 *   VIX/VXN:  DATE,OPEN,HIGH,LOW,CLOSE   (take CLOSE)
 *   VVIX:     DATE,VVIX                  (take column 1)
 * Dates are MM/DD/YYYY → ISO. Returns ascending [{date, value}].
 */
export function parseCboeCsv(text) {
    const lines = String(text || '').trim().split(/\r?\n/);
    if (lines.length < 2) return [];
    const header = lines[0].split(',').map((h) => h.trim().toUpperCase());
    let col = header.indexOf('CLOSE');
    if (col === -1) col = 1; // two-column schema (DATE,<INDEX>)
    const out = [];
    for (let i = 1; i < lines.length; i++) {
        const cells = lines[i].split(',');
        const value = parseFloat(cells[col]);
        const m = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec((cells[0] || '').trim());
        if (!m || !Number.isFinite(value)) continue;
        out.push({ date: `${m[3]}-${m[1]}-${m[2]}`, value });
    }
    out.sort((a, b) => (a.date < b.date ? -1 : 1));
    return out;
}

/** Annualized realized vol (vol points, e.g. 16.4) from ascending closes, last `days` daily log returns. */
export function realizedVol(closes, days = 21) {
    const px = (closes || []).filter((p) => Number.isFinite(p) && p > 0).slice(-(days + 1));
    if (px.length < days + 1) return null;
    const rets = [];
    for (let i = 1; i < px.length; i++) rets.push(Math.log(px[i] / px[i - 1]));
    const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
    const varSum = rets.reduce((a, r) => a + (r - mean) ** 2, 0) / (rets.length - 1);
    const rv = Math.sqrt(varSum) * Math.sqrt(252) * 100;
    return Number.isFinite(rv) ? rv : null;
}

/** IV rank over a window: where current sits between the window min and max, 0–100. */
export function ivRank(values, current) {
    const v = (values || []).filter(Number.isFinite);
    if (!v.length || !Number.isFinite(current)) return null;
    const min = Math.min(...v);
    const max = Math.max(...v);
    if (max === min) return null;
    return Math.min(100, Math.max(0, ((current - min) / (max - min)) * 100));
}

/** IV percentile over a window: share of days at or below current, 0–100. */
export function ivPercentile(values, current) {
    const v = (values || []).filter(Number.isFinite);
    if (!v.length || !Number.isFinite(current)) return null;
    return (v.filter((x) => x <= current).length / v.length) * 100;
}

/**
 * Assemble the payload for /api/vol.
 *
 * `liveQuotes` (optional) carries intraday index levels from CNBC's quote
 * endpoint, keyed by index name: { VIX: { value, date, lastTime }, … }. A live
 * level REPLACES the last EOD close as "current" ONLY when it is finite, > 0,
 * and its date is a well-formed YYYY-MM-DD STRICTLY newer than the last EOD
 * point — evenings, weekends
 * and a lagging quote all fall back to plain EOD behavior. The 1y
 * rank/percentile window stays EOD-only, and RV 21d never sees a partial day.
 * `live_at` is the newest applied quote's full timestamp (only ever a full
 * ISO string with a 'T'; a date-only lastTime is withheld so the UI can't
 * misparse it as UTC midnight).
 *
 * @param {object} indexSeries  {VIX|VXN|VVIX: ascending [{date, value}] | null}
 * @param {object} etfCloses    {SPY|…: ascending array of closes | null}
 * @param {object} [liveQuotes] {VIX|VXN|VVIX: {value, date, lastTime}} (optional)
 * @returns {{updated_at: string|null, live_at: string|null, tickers: Array}}
 */
export function buildVolMetrics(indexSeries = {}, etfCloses = {}, liveQuotes = {}) {
    let updated = null;
    let liveAt = null;
    const tickers = Object.entries(VOL_PROXIES).map(([ticker, { index, mult }]) => {
        const series = indexSeries[index] || null;
        const last = series && series.length ? series[series.length - 1] : null;
        const window = series ? series.slice(-ONE_YEAR).map((p) => p.value) : [];
        const quote = liveQuotes ? liveQuotes[index] : null;
        const live = !!(last && quote && Number.isFinite(quote.value) && quote.value > 0
            && ISO_DATE.test(quote.date) && quote.date > last.date);
        const current = live ? quote.value : (last ? last.value : null);
        const asOf = live ? quote.date : (last ? last.date : null);
        const iv = current != null ? mult * current : null;
        const rank = current != null ? ivRank(window, current) : null;
        const pctile = current != null ? ivPercentile(window, current) : null;
        const rv21 = realizedVol(etfCloses[ticker] || []);
        const vrp = iv != null && rv21 != null ? iv - rv21 : null;
        if (asOf && (!updated || asOf > updated)) updated = asOf;
        if (live && typeof quote.lastTime === 'string' && quote.lastTime.includes('T')
            && (!liveAt || quote.lastTime > liveAt)) liveAt = quote.lastTime;
        return { ticker, proxy: mult === 1 ? index : `${mult}×${index}`, iv, ivRank1y: rank, ivPctile1y: pctile, rv21, vrp, live, asOf };
    });
    return { updated_at: updated, live_at: liveAt, tickers };
}
