/**
 * Financial math and business logic library.
 * Consolidates technical indicators and data processing calculations.
 */

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
