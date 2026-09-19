/**
 * Breadth analysis: ratio calculation and statistics for ETF pairs.
 *
 * Pure math — no side effects, no fetches. All numbers may be null/undefined
 * (source down); functions must cope and return nulls.
 */

export const PAIRS = {
    rspSpy: ['RSP', 'SPY'],
    iwmSpy: ['IWM', 'SPY'],
    xlkXlu: ['XLK', 'XLU'],
    hygLqd: ['HYG', 'LQD'],
};

/**
 * Align two ascending price histories on common dates and compute the ratio
 * (priceA / priceB). Only dates present in BOTH histories are included.
 *
 * @param {Array<{date:string, price:number}>} histA ascending
 * @param {Array<{date:string, price:number}>} histB ascending
 * @returns {Array<{date:string, ratio:number}>} ascending, empty on mismatch
 */
export function ratioSeries(histA, histB) {
    if (!Array.isArray(histA) || !Array.isArray(histB) || !histA.length || !histB.length) {
        return [];
    }

    // Build a lookup from B for O(1) date lookups
    const pricesB = {};
    for (const entry of histB) {
        if (entry.price != null && Number.isFinite(entry.price)) {
            pricesB[entry.date] = entry.price;
        }
    }

    const result = [];
    for (const entry of histA) {
        const pb = pricesB[entry.date];
        if (pb != null && pb !== 0 && entry.price != null && Number.isFinite(entry.price) && entry.price !== 0) {
            result.push({ date: entry.date, ratio: entry.price / pb });
        }
    }

    result.sort((a, b) => (a.date < b.date ? -1 : 1));

    return result;
}

/**
 * Compute statistics from a ratio series.
 *
 * @param {Array<{date:string, ratio:number}>} series ascending
 * @returns {{ ratio: number|null, chg20Pct: number|null, chg60Pct: number|null, vs50dPct: number|null, asOf: string|null }}
 */
export function pairStats(series) {
    if (!Array.isArray(series) || series.length === 0) {
        return { ratio: null, chg20Pct: null, chg60Pct: null, vs50dPct: null, asOf: null };
    }

    const last = series[series.length - 1];
    if (last.ratio == null || !Number.isFinite(last.ratio)) {
        return { ratio: null, chg20Pct: null, chg60Pct: null, vs50dPct: null, asOf: last.date || null };
    }

    const ratio = last.ratio;
    const asOf = last.date;

    // chg over 20 trading rows: need current + 20 rows back
    let chg20Pct = null;
    if (series.length >= 21) {
        const idx = series.length - 21;
        if (Number.isFinite(series[idx].ratio) && series[idx].ratio !== 0) {
            chg20Pct = ((ratio / series[idx].ratio) - 1) * 100;
        }
    }

    // chg over 60 trading rows: need current + 60 rows back
    let chg60Pct = null;
    if (series.length >= 61) {
        const idx = series.length - 61;
        if (Number.isFinite(series[idx].ratio) && series[idx].ratio !== 0) {
            chg60Pct = ((ratio / series[idx].ratio) - 1) * 100;
        }
    }

    // vs50dPct: ratio vs mean of last 50 rows, %
    let vs50dPct = null;
    if (series.length >= 50) {
        const last50 = series.slice(-50);
        const sum = last50.reduce((acc, s) => acc + s.ratio, 0);
        const mean50 = sum / 50;
        if (Number.isFinite(mean50) && mean50 !== 0) {
            vs50dPct = ((ratio / mean50) - 1) * 100;
        }
    }

    return { ratio, chg20Pct, chg60Pct, vs50dPct, asOf };
}