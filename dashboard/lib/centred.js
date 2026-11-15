/**
 * Centred — helper for /api/centred and its /v2 tile.
 *
 * Each night the Mac (~/.local/bin/centred-publish.py) checks whether every threshold of the live algo sits in
 * the middle of a flat zone. This PUBLIC feed carries counts, state, dates and whole-algo deltas only; the
 * conditions themselves live on the nuts-radar Centred tab.
 */
const STATES = ['off', 'centred', 'unknown'];
const VERDICTS = ['no harm', 'harm', 'not judged'];
const count = (n) => Number.isInteger(n) && n >= 0;
const isDate = (s) => typeof s === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(s);

/** True for a feed the tile can draw (the route's no-feed fallback has no stamp, so it is not one). */
export function validateCentred(s) {
    return !!s && s.v === 1 && STATES.includes(s.state) && count(s.checked) && count(s.off) && s.off <= s.checked
        && (s.worlds == null || count(s.worlds))
        && (s.edge == null || (count(s.edge) && s.off + s.edge <= s.checked)) // at an edge: not off, not in the middle either
        && typeof s.generated_at === 'string' && Number.isFinite(Date.parse(s.generated_at))
        && (s.verdict == null || VERDICTS.includes(s.verdict))
        && (s.delta == null || ['cagr', 'maxdd', 'sharpe'].every((k) => Number.isFinite(s.delta[k])))
        && (s.history == null || (Array.isArray(s.history) && s.history.every((r) => r && isDate(r.date) && count(r.off) && count(r.checked))));
}
