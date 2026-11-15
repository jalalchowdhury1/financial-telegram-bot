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

const AHEAD_MS = 3600e3;                             // a stamp further ahead than this is a broken clock: it would never go stale

/**
 * True for a feed the tile can draw (the route's no-feed fallback has no stamp, so it is not one). A judged night
 * (state off / centred) must have settings, and its state must agree with its off count: "all 0 in the middle" or a
 * green tile over 3 off-centre settings is never drawn (red team 15 Nov).
 */
export function validateCentred(s, now = Date.now()) {
    return !!s && s.v === 1 && STATES.includes(s.state) && count(s.checked) && count(s.off) && s.off <= s.checked
        && (s.state === 'unknown' || (s.checked > 0 && (s.state === 'off') === (s.off > 0)))
        && (s.base == null || typeof s.base === 'string')
        && (s.worlds == null || count(s.worlds))
        && (s.edge == null || (count(s.edge) && s.off + s.edge <= s.checked)) // at an edge: not off, not in the middle either
        && typeof s.generated_at === 'string' && Number.isFinite(Date.parse(s.generated_at)) && Date.parse(s.generated_at) <= now + AHEAD_MS
        && (s.verdict == null || VERDICTS.includes(s.verdict))
        && (s.delta == null || ['cagr', 'maxdd', 'sharpe'].every((k) => Number.isFinite(s.delta[k])))
        && (s.history == null || (Array.isArray(s.history) && s.history.every((r) => r && isDate(r.date) && count(r.off) && count(r.checked) && r.off <= r.checked)));
}

/**
 * What /api/centred relays: the known fields only, rebuilt one by one, so nothing extra in the gist can ride along.
 * The note is a fixed line from the publisher; one that looks like a condition (a digit, a bracket, < or >) is dropped.
 */
export function publicCentred(s) {
    const note = typeof s.note === 'string' && s.note.length <= 80 && !/[\d()<>]/.test(s.note) ? s.note : '';
    return {
        v: 1, generated_at: s.generated_at, prices_to: isDate(s.prices_to) ? s.prices_to : null, state: s.state, note,
        checked: s.checked, off: s.off, edge: s.edge ?? null, base: s.base === 'champion' ? 'champion' : 'live',
        since: isDate(s.since) ? s.since : null, verdict: s.verdict ?? 'not judged', worlds: s.worlds ?? null,
        delta: s.delta ? { cagr: s.delta.cagr, maxdd: s.delta.maxdd, sharpe: s.delta.sharpe } : null,
        history: (s.history || []).map((r) => ({ date: r.date, off: r.off, checked: r.checked })),
    };
}

/** The strip's bar: one tick per setting, off ones first; past 40 settings the ticks stand for shares, an off one never lost. */
export const cbar = (c) => {
    const n = Math.min(c.checked, 40), off = c.off ? Math.max(1, Math.round((c.off * n) / c.checked)) : 0;
    return Array.from({ length: n }, (_, i) => i < off);
};
