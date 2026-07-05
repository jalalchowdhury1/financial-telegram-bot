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

export const VOL_PROXIES = {
    SPY: { index: 'VIX', mult: 1 },
    QQQ: { index: 'VXN', mult: 1 },
    TQQQ: { index: 'VXN', mult: 3 },
    SQQQ: { index: 'VXN', mult: 3 },
    UVXY: { index: 'VVIX', mult: 1 },
};

const ONE_YEAR = 252; // trading days

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
 * @param {object} indexSeries  {VIX|VXN|VVIX: ascending [{date, value}] | null}
 * @param {object} etfCloses    {SPY|…: ascending array of closes | null}
 * @returns {{updated_at: string|null, tickers: Array}}
 */
export function buildVolMetrics(indexSeries = {}, etfCloses = {}) {
    let updated = null;
    const tickers = Object.entries(VOL_PROXIES).map(([ticker, { index, mult }]) => {
        const series = indexSeries[index] || null;
        const last = series && series.length ? series[series.length - 1] : null;
        const window = series ? series.slice(-ONE_YEAR).map((p) => p.value) : [];
        const iv = last ? mult * last.value : null;
        const rank = last ? ivRank(window, last.value) : null;
        const pctile = last ? ivPercentile(window, last.value) : null;
        const rv21 = realizedVol(etfCloses[ticker] || []);
        const vrp = iv != null && rv21 != null ? iv - rv21 : null;
        if (last && (!updated || last.date > updated)) updated = last.date;
        return { ticker, proxy: mult === 1 ? index : `${mult}×${index}`, iv, ivRank1y: rank, ivPctile1y: pctile, rv21, vrp, asOf: last ? last.date : null };
    });
    return { updated_at: updated, tickers };
}
