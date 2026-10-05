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
 * On a desk two or three cards share a grid row (the same top, ±2px). Then the card under
 * the screen's centre wins; if the centre falls in a gutter, or widths are unknown, the
 * pill names the row ("S&P 500 EPS · Economy") and the first of them anchors the way back.
 * @param {number} y scrollY  @param {{label:string, top:number, left?:number, right?:number}[]} sections
 *   document rects  @param {number} vh  @param {number} [cx] the screen's horizontal centre
 * @returns {{label:string, offset:number, name?:string}} offset = y − that section's top ('Top' = from 0)
 */
export function pickAnchor(y, sections, vh, cx) {
    const line = y + (Number.isFinite(vh) ? vh / 3 : 0);
    const ok = (Array.isArray(sections) ? sections : []).filter((s) => s && s.label && Number.isFinite(s.top) && s.top <= line);
    if (!ok.length) return { label: 'Top', offset: y };
    const rowTop = Math.max(...ok.map((s) => s.top));
    const row = ok.filter((s) => s.top >= rowTop - 2);
    if (row.length === 1) return { label: row[0].label, offset: y - row[0].top };
    const under = Number.isFinite(cx) ? row.find((s) => Number.isFinite(s.left) && Number.isFinite(s.right) && s.left <= cx && cx <= s.right) : null;
    if (under) return { label: under.label, offset: y - under.top };
    return { label: row[0].label, offset: y - row[0].top, name: row.slice(0, 2).map((s) => s.label).join(' · ') };
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

export function rememberJump({ y, vh, sections, cx }) {
    if (!Number.isFinite(y)) return null;
    spot = { ...pickAnchor(y, sections, vh, cx), y, id: nextId++ };
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
