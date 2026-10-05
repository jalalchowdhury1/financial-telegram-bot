/**
 * Fear & Greed card history cells ("Prev Close", "1 Week", "1 Month", "1 Year").
 * The route's backup tiers leave cells empty: the Yahoo VIX proxy sends previousYear 'N/A',
 * the FRED tier sends null when a past observation is missing. Math.round('N/A') printed
 * "NaN ▼NaN", and Math.round(null) would print a made-up 0 — a missing cell is null here
 * and the card shows "—" with no arrow.
 */
export const FG_HISTORY = [
    ['Prev Close', 'previousClose'],
    ['1 Week', 'previousWeek'],
    ['1 Month', 'previousMonth'],
    ['1 Year', 'previousYear'],
];

const toNum = (v) => (typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN);

/** [{label, val: whole number | null, diff: today − val | null}] — never NaN, never a fake 0. */
export function fgHistoryCells(fg) {
    const cur = toNum(fg?.score);
    return FG_HISTORY.map(([label, key]) => {
        const n = toNum(fg?.[key]);
        const val = Number.isFinite(n) ? Math.round(n) : null;
        const diff = val != null && Number.isFinite(cur) ? Math.round(cur) - val : null;
        return { label, val, diff };
    });
}
