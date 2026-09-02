/**
 * Rubber Band Radar — shared helpers for /api/rubber-band and <RubberBandRadar/>.
 *
 * The maths does NOT run here. The Mac mini computes the five dials nightly
 * (scripts/rubber_band.py) and publishes one JSON snapshot to a secret gist; the
 * dashboard only validates, freshness-stamps and displays it. docs/rubber-band.md
 * is the research record behind every threshold.
 */

export const DIAL_ORDER = ['slow', 'fast', 'age', 'rip', 'machines'];

const LABELS = {
    slow: 'Dip pays?',
    fast: 'Dip pays? (fast)',
    age: 'Evidence age',
    rip: 'Rips fade?',
    machines: 'Machine health',
};

export const COLOURS = new Set(['green', 'amber', 'red', 'grey']);

export function dialLabel(key) {
    return LABELS[key] || key;
}

/** True only for a snapshot with all five dials carrying a known colour and a verdict. */
export function validateSnapshot(s) {
    if (!s || typeof s !== 'object' || !s.dials || typeof s.dials !== 'object') return false;
    if (!s.verdict || typeof s.verdict !== 'object' || !COLOURS.has(s.verdict.colour)) return false;
    if (typeof s.asOf !== 'string') return false;
    return DIAL_ORDER.every((k) => s.dials[k] && COLOURS.has(s.dials[k].colour));
}

const MS_PER_DAY = 86400000;
// The engine runs each weekday evening. A Friday snapshot is legitimately the newest
// thing until Monday night (3 days); anything older than 4 days means a run was missed.
export const STALE_AFTER_DAYS = 4;

/** @returns {{ageDays:number|null, stale:boolean}} */
export function snapshotAge(asOf, now = new Date()) {
    if (!asOf) return { ageDays: null, stale: true };
    const then = new Date(`${asOf}T00:00:00Z`);
    if (Number.isNaN(then.getTime())) return { ageDays: null, stale: true };
    const ageDays = Math.floor((now.getTime() - then.getTime()) / MS_PER_DAY);
    return { ageDays, stale: ageDays > STALE_AFTER_DAYS };
}
