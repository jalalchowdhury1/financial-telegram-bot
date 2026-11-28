/**
 * Own watch: helper for /api/own and the /v2 Today card.
 *
 * Each morning (06:25) the Mac mini (~/.local/bin/own-watch.py) writes `own.json` to the split's secret gist: the
 * alarms on what he owns and a race of named runners against Main. This repo is public, so the relay rebuilds the
 * answer from an allow-list, field by field: nothing else in the gist file can ever reach the site.
 */
const STATES = ['ok', 'tripped', 'unread'];
const STAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:\d{2})$/;
const AHEAD_MS = 3600e3;                                 // a stamp further ahead than this is a broken clock: it would never go stale
const isDate = (s) => typeof s === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(s);
const isId = (s) => typeof s === 'string' && /^[a-z0-9_-]{1,16}$/.test(s);
// Plain words only: a ticker-like word in capitals, a bracket, < > or $ never passes (conditions and dollars stay on the Mac).
const isText = (s, max) => typeof s === 'string' && s.length > 0 && s.length <= max && !/[A-Z]{2}|[()<>$[\]{}]/.test(s);
const pts = (n) => Number.isFinite(n) && Math.abs(n) < 1000;

function alarm(a) {
    if (!a || !isId(a.id) || !isText(a.name, 28) || !STATES.includes(a.state)) return null;
    return { id: a.id, name: a.name, state: a.state, note: isText(a.note, 28) ? a.note : '' };
}

function race(r) {
    if (!r || !isDate(r.start) || !Number.isInteger(r.day) || r.day < 0 || r.day > 5000 || !Array.isArray(r.rows) || !r.rows.length || r.rows.length > 6) return null;
    const rows = r.rows.map((x) => (x && isId(x.id) && isText(x.name, 20) && pts(x.vs) && pts(x.fall)
        ? { id: x.id, name: x.name, vs: Math.round(x.vs * 10) / 10, fall: Math.round(x.fall * 10) / 10 } : null));
    return rows.every(Boolean) ? { start: r.start, day: r.day, rows } : null;
}

/** Main's nearest rule at the last close: points from its line (0 = on) and which side; anything else is null. */
function near(n) {
    if (!n || !Number.isFinite(n.gap) || n.gap < 0 || n.gap > 100 || !['dip', 'hot'].includes(n.side)) return null;
    return { gap: Math.round(n.gap * 10) / 10, side: n.side };
}

const KINDS = ['stocks', 'bonds', 'cash', 'fear', 'short', 'other'];
/** What Main held at the last close, as plain kinds (no tickers): 1 to 6 of { kind, pct }; anything else is null. */
function holds(h) {
    if (!Array.isArray(h) || !h.length || h.length > 6) return null;
    const out = h.map((x) => (x && KINDS.includes(x.kind) && Number.isInteger(x.pct) && x.pct >= 0 && x.pct <= 100 ? { kind: x.kind, pct: x.pct } : null));
    return out.every(Boolean) ? out : null;
}

/**
 * What /api/own relays, rebuilt from the allow-list (v, made, asof, holds[kind,pct], near{gap,side}, alarms[id,name,state,note], race{start,day,rows[id,name,vs,fall]}),
 * or null when the feed is not one the card can trust. A race, holds or near that fails its checks is dropped alone (null); a bad alarm fails the feed.
 */
export function publicOwn(s, now = Date.now()) {
    if (!s || s.v !== 1 || typeof s.made !== 'string' || !STAMP.test(s.made) || !(Date.parse(s.made) <= now + AHEAD_MS)) return null;
    if (!isDate(s.asof) || !Array.isArray(s.alarms) || !s.alarms.length || s.alarms.length > 8) return null;
    const alarms = s.alarms.map(alarm);
    if (!alarms.every(Boolean)) return null;
    return { v: 1, made: s.made, asof: s.asof, holds: holds(s.holds), near: near(s.near), alarms, race: race(s.race) };
}

export const validateOwn = (s, now = Date.now()) => publicOwn(s, now) !== null;
