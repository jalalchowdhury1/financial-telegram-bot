/**
 * ↩ Back pill — every jump (What moved chip, Market Pulse chip, ☰ menu) used to be one-way.
 * Before a jump, the page remembers where he was; the pill (components/BackPill.js) takes
 * him back with one tap.
 *
 * The spot is kept as "section + offset", not a raw scrollY: cards still landing above
 * (Jev pills, a slow feed) push the page down while he peeks, and the section moves with
 * them. No history entries and no #hash: the iOS edge-swipe and the back button behave
 * exactly as before. One spot only — a second jump overwrites it.
 */

let spot = null;
let nextId = 1;
const subs = new Set();
const emit = () => subs.forEach((fn) => { try { fn(spot); } catch { /* a listener never breaks a jump */ } });

/**
 * The section he is reading: the last one starting above the top third of the screen.
 * @param {number} y scrollY  @param {{label:string, top:number}[]} sections document tops
 * @returns {{label:string, offset:number}} offset = y − that section's top ('Top' = from 0)
 */
export function pickAnchor(y, sections, vh) {
    const line = y + (Number.isFinite(vh) ? vh / 3 : 0);
    let best = null;
    for (const s of Array.isArray(sections) ? sections : []) {
        if (!s || !s.label || !Number.isFinite(s.top) || s.top > line) continue;
        if (!best || s.top > best.top) best = s;
    }
    return best ? { label: best.label, offset: y - best.top } : { label: 'Top', offset: y };
}

/** Where "back" is now: the section's current top + the offset (the raw y if it is gone). */
export function resolveBack(s, topOf) {
    if (!s) return null;
    if (s.label === 'Top') return Math.max(0, s.offset);
    let t = null;
    try { t = topOf(s.label); } catch { t = null; }
    return Math.max(0, Number.isFinite(t) ? t + s.offset : s.y);
}

/** More than one screen of manual scrolling from where the jump landed. */
export function movedTooFar(y, landY, vh) {
    return Number.isFinite(landY) && Math.abs(y - landY) > vh;
}

export function rememberJump({ y, vh, sections }) {
    if (!Number.isFinite(y)) return null;
    spot = { ...pickAnchor(y, sections, vh), y, id: nextId++ };
    emit();
    return spot;
}

export function clearJump() {
    if (!spot) return;
    spot = null;
    emit();
}

export const currentJump = () => spot;

/** Subscribe to the spot; returns the unsubscribe function. */
export function onJump(fn) {
    subs.add(fn);
    return () => subs.delete(fn);
}
