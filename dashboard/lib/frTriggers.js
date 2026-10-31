/**
 * FrontRunner's RSI triggers, as data for the v2 Signals card (lib/signals.js:
 * frontRunnerDecision is the sheet's verbatim tree; signals.test.js pins these to it).
 * OVER: RSI above it -> VIX hedge (LABU -> LABD). UNDER: RSI below it -> buy that dip.
 */
export const FR_OVER = { QQQ: 79, SPY: 79, IOO: 80, XLP: 77, VTV: 79, XLF: 81, VOX: 84, CURE: 82, RETL: 82, LABU: 87 };
export const FR_UNDER = { SOXL: 25, FNGU: 25, TQQQ: 28, TECL: 25, UPRO: 25 };
