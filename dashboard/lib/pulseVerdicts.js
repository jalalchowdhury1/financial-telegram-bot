/**
 * 📡 Market Pulse verdicts — one short chip per card that sits far below the fold, so one
 * glance at the top answers "is anything wrong down there?". Tap a chip = jump to its card.
 *
 * Every chip is built from the SAME answer and the SAME threshold its card uses:
 *   Vol       ← /api/vol regime.curve.state     (VolMetricsTable's 🟢 Calm / 🟡 Watch / 🔴 Stress pill)
 *   Horsemen  ← /api/fred, FourHorsemen's "N of 4 riding" rules (a parity test renders the card)
 *   Curve     ← /api/fred yieldCurve.current    (FourHorsemen's 10Y−2Y row; < 0 = inverted)
 *   Bull      ← /api/fred checklist              (BullChecklist's "7/8" badge and its colours)
 *   Dips      ← /api/rubber-band verdict.colour (RubberBandRadar's OK / WATCH / STOP badge)
 * A missing or failed source leaves its chip out — never a guessed 0. A saved copy (this
 * device) or a stale source (the route says so) keeps the chip but marks it `old`.
 */
import { latestYoY } from './horsemenRunup';

const RB = {
    green: { text: 'Dips pay ✓', tone: 'good', word: 'OK' },
    amber: { text: 'Dips: watch', tone: 'watch', word: 'WATCH' },
    red: { text: 'Dips: stop', tone: 'bad', word: 'STOP' },
};
const VOL = {
    calm: { text: 'Vol calm', tone: 'good' },
    watch: { text: 'Vol watch', tone: 'watch' },
    stress: { text: 'Vol stress', tone: 'bad' },
};

const isObj = (o) => !!o && typeof o === 'object' && !Array.isArray(o);
const usable = (o) => isObj(o) && !o.error;
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const savedMark = (label) => (label ? { kind: 'saved', note: `saved copy ${label}` } : null);

/**
 * The Recession watch card's tells, with its thresholds (FourHorsemen.js `warn`):
 * claims > +10 % vs 1y · Sahm ≥ 0.5 · 10Y−2Y < 0 · bankruptcies > +10 % vs 1y.
 * "vs 1y" = the card's own latestYoY (latest print vs the print a year before it, ±7 days);
 * bankruptcies fall back to the route's changePct only when the history can't answer.
 * @returns {{riding:number, known:number}|null} null when nothing is known.
 */
export function horsemenRiding(fred) {
    if (!usable(fred)) return null;
    const h = isObj(fred.horsemen) ? fred.horsemen : {};
    const claims = latestYoY(h.claims?.history, 'pct');
    const sahm = num(fred.indicators?.sahmRule?.value);
    const spread = num(fred.yieldCurve?.current);
    const bk = num(latestYoY(h.bankruptcies?.history, 'pct')) ?? num(h.bankruptcies?.changePct);
    const tells = [
        claims == null ? null : claims > 10,
        sahm == null ? null : sahm >= 0.5,
        spread == null ? null : spread < 0,
        bk == null ? null : bk > 10,
    ];
    const known = tells.filter((t) => t !== null).length;
    if (!known) return null;
    return { riding: tells.filter((t) => t === true).length, known };
}

function dipsChip(rb) {
    if (!usable(rb)) return null;
    const v = RB[rb.verdict?.colour];
    if (!v) return null; // grey = NO DATA on the card
    const meta = rb._meta || {};
    const days = num(meta.ageDays);
    return {
        key: 'dips', jump: 'Rubber band', text: v.text, tone: v.tone,
        why: `Rubber band ${v.word}${rb.asOf ? ` (${rb.asOf})` : ''}: ${rb.verdict.text || ''}`.trim(),
        old: meta.stale ? { kind: 'stale', note: days != null ? `stale · ${days} days old` : 'stale' } : null,
    };
}

function volChip(vol, savedLabel) {
    if (!usable(vol)) return null;
    const curve = vol.regime?.curve;
    const v = VOL[curve?.state];
    if (!v || !Array.isArray(curve.points) || !curve.points.length) return null;
    const ratio = num(curve.ratio);
    return {
        key: 'vol', jump: 'Volatility', text: v.text, tone: v.tone,
        why: `VIX curve ${curve.state}${ratio != null ? `: VIX ÷ VIX3M ${ratio.toFixed(2)}` : ''}${curve.asOf ? ` (${curve.asOf})` : ''}`,
        old: curve.stale ? { kind: 'stale', note: `saved copy from ${curve.asOf || 'an earlier day'}` } : savedMark(savedLabel),
    };
}

function horsemenChip(fred, savedLabel) {
    const r = horsemenRiding(fred);
    if (!r) return null;
    return {
        key: 'horsemen', jump: 'Recession watch', text: `Horsemen ${r.riding}/4`,
        tone: r.riding >= 3 ? 'bad' : r.riding >= 1 ? 'caution' : 'good', // the card's badge-yellow
        why: `Recession watch: ${r.riding} of 4 riding${r.known < 4 ? ` (${4 - r.known} with no data)` : ''}`,
        old: savedMark(savedLabel),
    };
}

function curveChip(fred, savedLabel) {
    if (!usable(fred)) return null;
    const yc = fred.yieldCurve || {};
    const c = num(yc.current);
    if (c == null) return null;
    return {
        // a real minus (U+2212), as What moved prints "VIX −6.6%"
        key: 'curve', jump: 'Yield curve', text: `Curve ${c >= 0 ? '+' : '−'}${Math.abs(c).toFixed(2)}%`,
        tone: c >= 0 ? 'good' : 'bad',
        why: `10Y−2Y yield curve ${c >= 0 ? 'positive' : 'inverted'}${yc.asOf ? ` (${yc.asOf})` : ''}`,
        old: yc.stale ? { kind: 'stale', note: `stale · as of ${yc.asOf || '?'}` } : savedMark(savedLabel),
    };
}

function bullChip(fred, savedLabel) {
    if (!usable(fred) || !isObj(fred.checklist)) return null;
    const items = Object.values(fred.checklist).filter(isObj);
    if (!items.length) return null;
    const bullish = items.filter((i) => i.bullish).length;
    const pct = (bullish / items.length) * 100;
    return {
        key: 'bull', jump: 'Bull checklist', text: `Bull ${bullish}/${items.length}`,
        tone: pct >= 75 ? 'good' : pct >= 50 ? 'caution' : 'bad', // the card's badge-yellow
        why: `Bull market checklist: ${bullish} of ${items.length} bullish`,
        old: savedMark(savedLabel),
    };
}

/**
 * @param {{fred?:any, vol?:any, rubberBand?:any, saved?:{fred?:string, vol?:string}}} src
 *   `saved` = the page's "🕐 19:43" label for a feed currently shown from a saved copy.
 * @returns {Array<{key, jump, text, tone:'good'|'caution'|'watch'|'bad', why, old:null|{kind:'saved'|'stale', note}}>}
 */
export function pulseVerdicts({ fred, vol, rubberBand, saved = {} } = {}) {
    const s = isObj(saved) ? saved : {};
    // Dips goes LAST: its card fetches it after mount, so it lands after fred + vol (which
    // paint from saved copies at once). Appended, it never slides a chip under a thumb.
    const make = [
        () => volChip(vol, s.vol),
        () => horsemenChip(fred, s.fred),
        () => curveChip(fred, s.fred),
        () => bullChip(fred, s.fred),
        () => dipsChip(rubberBand),
    ];
    const chips = [];
    for (const m of make) {
        try { const c = m(); if (c) chips.push(c); } catch { /* one bad source never hides the others */ }
    }
    return chips;
}
