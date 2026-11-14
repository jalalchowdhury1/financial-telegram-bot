/**
 * Main / Shartino split — helper for /api/split and the Indicators tile.
 *
 * The maths runs on the Mac mini (scripts/split.py): his last two totals, rolled forward
 * with the night shift's replay of each algo. Only percentages are published.
 */

/** True for a snapshot whose numbers the tile can draw. */
export function validateSplit(s) {
    return !!s && typeof s.asOf === 'string' && [s.main, s.target, s.band].every(Number.isFinite)
        && s.main >= 0 && s.main <= 100;
}
