/**
 * lastVisit.js — 👋 "Since you last looked": on open, one line of what changed since
 * the numbers you last SAW — "Since Thu 09:05 · SPY +1.2% · VIX −8% · F&G +6 · 10Y −5bp ·
 * 🆕 Claims, Sahm Rule".
 *
 * A tiny record (`fd:seen:v1`, a few hundred bytes of plain numbers) is updated every time
 * live numbers land on a visible page. It is NOT keyed to the deploy like the saved copies
 * (lib/snapshot.js): flat numbers cannot break new code, and a deploy must not wipe it.
 *
 * Honesty rules:
 *  - Only LIVE values are recorded: the page passes a feed as null unless its live answer
 *    landed AFTER the baseline was read (a saved copy, or numbers still on screen from
 *    before the tab was hidden, are not a new look), and each field is stamped with the
 *    time ITS feed landed (`landedAt`), never with the time of an unrelated re-render.
 *  - Every field carries its own time. SPY's time is the visit's clock; a field last seen
 *    more than COHERENT_MS away from it is left out, never compared across the wrong gap.
 *  - Shown only when that visit is SINCE_MIN_GAP_MS or more ago (a reload is not a visit).
 *  - 🆕 lists "print" metrics (lib/marks.js) whose value differs from that visit.
 */
import { change, fmtMove, roundsToZero, LIVE_FG_SOURCES } from './whatMoved';
import { SHEET_METRICS } from './marks';
import { savedLabel } from './snapshot';

export const SEEN_KEY = 'fd:seen:v1';
export const SINCE_MIN_GAP_MS = 60 * 60 * 1000;
export const COHERENT_MS = 60 * 60 * 1000;
export const MAX_PRINTS_SHOWN = 3;

export const SEEN_FIELDS = [
    { key: 'spy', label: 'SPY', unit: 'pct' },
    { key: 'vix', label: 'VIX', unit: 'pct' },
    { key: 'fg', label: 'F&G', unit: 'pts' },
    { key: 'tnx', label: '10Y', unit: 'bp' },
];

const fin = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/**
 * The numbers on screen now. Pass LIVE feeds only (a saved copy is not a new look);
 * `prints` = collectLiveValues(fred, extra, sheets), which already drops stale values.
 */
export function pickSeen({ spy, vol, fg, extra } = {}, prints = {}) {
    const v = {
        spy: spy && !spy.error ? fin(spy.current) : null,
        vix: fin(vol?.vixDay?.value),
        fg: LIVE_FG_SOURCES.includes(fg?._meta?.source) ? fin(fg.score) : null,
        tnx: fin(extra?.rates?.tnx?.current),
    };
    const p = {};
    for (const [k, x] of Object.entries(prints || {})) {
        if (SHEET_METRICS[k]?.kind === 'print' && fin(x) != null) p[k] = x;
    }
    return { v, p };
}

/** The page feed each field comes from (FEEDS keys in app/page.js). */
export const FIELD_FEED = { spy: 'spy', vix: 'vol', fg: 'fg', tnx: 'extra' };
const PRINT_FEED = { aaiiDiff: 'sheets', rentIndex: 'extra', mortgagePayment: 'extra', mortgageRate: 'extra', atnhpi: 'extra' };
export const feedOf = (group, k) => (group === 'v' ? FIELD_FEED[k] : PRINT_FEED[k] || 'fred');

/**
 * Fold the numbers on screen into the record; a field not live now keeps its old value and
 * time. `landedAt` (feed → ms its live answer landed) stamps each field; `now` is the fallback.
 */
export function mergeSeen(prev, picked, now = Date.now(), landedAt = {}) {
    const at = (g, k) => { const t = landedAt?.[feedOf(g, k)]; return Number.isFinite(t) ? t : now; };
    const out = { v: { ...(prev?.v || {}) }, p: { ...(prev?.p || {}) } };
    for (const [k, x] of Object.entries(picked?.v || {})) if (x != null) out.v[k] = { x, at: at('v', k) };
    for (const [k, x] of Object.entries(picked?.p || {})) if (x != null) out.p[k] = { x, at: at('p', k) };
    return out;
}

const okField = (f) => f && fin(f.x) != null && fin(f.at) != null;
function clean(group) {
    const out = {};
    if (group && typeof group === 'object') for (const [k, f] of Object.entries(group)) if (okField(f)) out[k] = { x: f.x, at: f.at };
    return out;
}

function storage() {
    try { return typeof window !== 'undefined' ? window.localStorage : null; } catch { return null; }
}

/** @returns {{v:object, p:object}|null} — junk fields are dropped, never trusted. */
export function readSeen(store = storage()) {
    try {
        const raw = store?.getItem(SEEN_KEY);
        if (!raw) return null;
        const r = JSON.parse(raw);
        return { v: clean(r?.v), p: clean(r?.p) };
    } catch { return null; }
}

export function writeSeen(rec, store = storage()) {
    try { store.setItem(SEEN_KEY, JSON.stringify(rec)); return true; } catch { return false; }
}

/**
 * @returns {{since:string, anchorAt:number, items:Array<{key,label,text,dir}>, prints:string[], more:number}|null}
 */
export function sinceLastVisit(base, picked, now = Date.now()) {
    const anchor = base?.v?.spy?.at;
    if (fin(anchor) == null || now - anchor < SINCE_MIN_GAP_MS) return null;
    const near = (f) => okField(f) && Math.abs(f.at - anchor) <= COHERENT_MS;
    const items = [];
    for (const f of SEEN_FIELDS) {
        const b = base.v[f.key], x = picked?.v?.[f.key];
        if (!near(b) || fin(x) == null) continue;
        const d = change(b.x, x, f.unit);
        if (d == null || !Number.isFinite(d)) continue;
        const flat = roundsToZero(d, f.unit);
        items.push({ key: f.key, label: f.label, dir: flat ? 0 : Math.sign(d), text: flat ? 'flat' : fmtMove(d, f.unit) });
    }
    const changed = [];
    for (const [k, x] of Object.entries(picked?.p || {})) {
        const b = base.p?.[k];
        if (!near(b) || fin(x) == null) continue;
        if (Math.abs(x - b.x) > 1e-9 * Math.max(1, Math.abs(b.x))) changed.push(SHEET_METRICS[k]?.label || k);
    }
    if (!items.length && !changed.length) return null;
    return {
        since: savedLabel(anchor, now), anchorAt: anchor, items,
        prints: changed.slice(0, MAX_PRINTS_SHOWN), more: Math.max(0, changed.length - MAX_PRINTS_SHOWN),
    };
}
