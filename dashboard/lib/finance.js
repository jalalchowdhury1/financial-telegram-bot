/**
 * Financial math and business logic library.
 * Consolidates technical indicators and data processing calculations.
 */

/**
 * Copper/Gold ratio — a leading macro gauge (cyclical industrial demand vs. safe haven).
 * Copper in $/lb, gold in $/oz; scaled ×1000 so it reads as a friendly number (~1.4).
 * Rising = growth/risk-on, falling = risk-off. Returns null if either leg is missing.
 * @param {number} copperUsdPerLb
 * @param {number} goldUsdPerOz
 * @returns {number|null}
 */
export function copperGoldRatio(copperUsdPerLb, goldUsdPerOz) {
    // Reject non-finite (NaN/Infinity) and non-positive prices so a bad leg can
    // never produce Infinity/0 garbage (a leg's `current` is already finite-checked
    // upstream, but be defensive — this is the one place the division happens).
    if (!Number.isFinite(copperUsdPerLb) || !Number.isFinite(goldUsdPerOz) || copperUsdPerLb <= 0 || goldUsdPerOz <= 0) return null;
    return (copperUsdPerLb / goldUsdPerOz) * 1000;
}

const MS_DAY = 86400000;

/**
 * Pick the price ~windowDays before the latest observation, for computing a
 * trailing change. `historyAsc` is oldest→newest [{date:'YYYY-MM-DD', price}].
 * Returns the price of the observation whose date is CLOSEST to (latestDate −
 * windowDays), but only if that closest point is within `tolDays` of the target
 * (so a too-short history yields null rather than a bogus "change"). This makes
 * the delta robust across mixed frequencies: a monthly copper series snaps the
 * 30-day target to the prior month; a daily gold series snaps to the exact day.
 *
 * @param {Array<{date:string, price:number}>} historyAsc
 * @param {string|Date} latestDate  date of the newest point
 * @param {number} windowDays       e.g. 30 (≈1mo) or 90 (≈3mo)
 * @param {number} [tolDays]        max allowed gap from the target date
 * @returns {number|null}
 */
// tolDays default: 60% of the window, floored at 18 days. The 18-day floor lets a
// 30-day window snap to a MONTHLY copper series (1 obs/~30d) without rejecting the
// prior month. Callers here only use 30/90-day windows.
export function priceAtAgo(historyAsc, latestDate, windowDays, tolDays = Math.max(windowDays * 0.6, 18)) {
    if (!Array.isArray(historyAsc) || historyAsc.length < 2) return null;
    const latest = new Date(latestDate);
    if (Number.isNaN(latest.getTime())) return null;
    const target = latest.getTime() - windowDays * MS_DAY;
    let best = null, bestGap = Infinity;
    for (const h of historyAsc) {
        if (h == null || h.price == null || Number.isNaN(Number(h.price))) continue;
        const t = new Date(h.date).getTime();
        if (Number.isNaN(t)) continue;
        // Only look backwards from the latest point (ignore points newer than ~latest).
        if (t > latest.getTime() + MS_DAY) continue;
        const gap = Math.abs(t - target);
        if (gap < bestGap) { bestGap = gap; best = Number(h.price); }
    }
    if (best == null) return null;
    return bestGap <= tolDays * MS_DAY ? best : null;
}

/**
 * Change of the ratio between two points: {value, pct} or null if either side
 * is missing/zero. `value` is in ratio points; `pct` is percent.
 * @returns {{value:number, pct:number}|null}
 */
export function ratioChange(ratioNow, ratioAgo) {
    if (ratioNow == null || ratioAgo == null || ratioAgo === 0) return null;
    const value = ratioNow - ratioAgo;
    return { value, pct: (value / ratioAgo) * 100 };
}

/**
 * Calculate Relative Strength Index (RSI) using Wilder's Smoothing
 * @param {Array} prices - Array of price objects or numbers
 * @param {Number} period - RSI period (default 9)
 */
export function calculateRSI(prices, period = 9) {
    if (!prices || prices.length < period) return 50;

    // Extract close prices if objects
    const values = typeof prices[0] === 'object' ? prices.map(p => p.close || p.price || p.value) : prices;

    // Calculate price changes (deltas)
    const deltas = [];
    for (let i = 1; i < values.length; i++) {
        deltas.push(values[i] - values[i - 1]);
    }

    if (deltas.length < period) return 50;

    // Separate gains and losses
    const gains = deltas.map(d => Math.max(d, 0));
    const losses = deltas.map(d => Math.max(-d, 0));

    // Initialize with SMA for first period
    let avgGain = gains.slice(0, period).reduce((s, v) => s + v, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((s, v) => s + v, 0) / period;

    // Apply Wilder's smoothing (EMA with alpha = 1/period)
    const alpha = 1 / period;
    for (let i = period; i < gains.length; i++) {
        avgGain = alpha * gains[i] + (1 - alpha) * avgGain;
        avgLoss = alpha * losses[i] + (1 - alpha) * avgLoss;
    }

    // Handle edge case where avgLoss is 0 (RSI = 100)
    if (avgLoss === 0) return 100;

    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
}

/**
 * Calculate Moving Average
 * @param {Array} prices - Array of price objects or numbers
 * @param {Number} period - MA period
 */
export function calculateMA(prices, period) {
    if (!prices || prices.length < period) return null;

    const values = typeof prices[0] === 'object' ? prices.map(p => p.close || p.price || p.value) : prices;
    const slice = values.slice(-period);
    return slice.reduce((s, v) => s + v, 0) / period;
}

/**
 * Calculate percentage change between two values
 */
export function calculatePctChange(current, previous) {
    if (!previous || previous === 0) return 0;
    return ((current - previous) / previous) * 100;
}
