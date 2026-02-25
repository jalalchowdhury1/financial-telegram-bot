/**
 * Financial math and business logic library.
 * Consolidates technical indicators and data processing calculations.
 */

/**
 * Calculate Relative Strength Index (RSI)
 * @param {Array} prices - Array of price objects or numbers
 * @param {Number} period - RSI period (default 9)
 */
export function calculateRSI(prices, period = 9) {
    if (!prices || prices.length <= period) return 50;

    // Extract close prices if objects
    const values = typeof prices[0] === 'object' ? prices.map(p => p.close || p.price || p.value) : prices;

    const recentValues = values.slice(-(period + 1));
    const deltas = [];
    for (let i = 1; i < recentValues.length; i++) {
        deltas.push(recentValues[i] - recentValues[i - 1]);
    }

    let avgGain = 0, avgLoss = 0;
    for (let i = 0; i < deltas.length; i++) {
        if (deltas[i] > 0) avgGain += deltas[i];
        else avgLoss += Math.abs(deltas[i]);
    }
    avgGain /= period;
    avgLoss /= period;

    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
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
