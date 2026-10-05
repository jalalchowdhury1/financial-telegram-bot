/**
 * chartAxis.js — small pure helpers shared by MiniChart and SpyChart.
 *
 * - yearTicks: x-axis year labels that stay readable (the Profit Margin card used to
 *   print all 80 years 1947→2026 on one 480-unit axis: an unreadable smear).
 * - indexFromPointer: which data point sits under a mouse/finger (tap-to-read).
 * - tfAvailable: a timeframe tab is offered only if the history really covers it
 *   (a "5Y" tab over 14 months of bars is a mislabel, not a choice).
 * - readChoice / saveChoice: remember a chart's timeframe per device, never throwing
 *   (private mode / blocked storage just means "not remembered").
 * - unitsForPx / spreadLabels / axisPos: axis labels are HTML spans laid over the SVG
 *   (an SVG with preserveAspectRatio="none" squashed 7-unit text to ~4.5px on a phone),
 *   so spacing is decided in real CSS pixels once the chart has been measured.
 */

const STEPS = [1, 2, 5, 10, 20, 25, 50, 100];

/**
 * @param {string[]} dates  ascending ISO dates, one per plotted point
 * @param {(i:number)=>number} toX  index → x in viewBox units
 * @returns {{x:number,label:string,year:number}[]}
 */
export function yearTicks(dates, toX, { maxLabels = 8, minGap = 26 } = {}) {
    const all = [];
    let last = '';
    for (let i = 0; i < dates.length; i++) {
        const yr = String(dates[i]).slice(0, 4);
        if (yr !== last) { all.push({ x: toX(i), label: yr, year: Number(yr) }); last = yr; }
    }
    if (all.length <= 1) return all;
    const span = all[all.length - 1].year - all[0].year + 1;
    const step = STEPS.find((s) => Math.ceil(span / s) <= maxLabels) || STEPS[STEPS.length - 1];
    const kept = step === 1 ? all : all.filter((t) => t.year % step === 0);
    // A label crowding its right-hand neighbour is the partial first year
    // ("2021" at the left edge, then "2022" a few units later) — drop it.
    return kept.filter((t, i) => !(kept[i + 1] && kept[i + 1].x - t.x < minGap));
}

/** Index of the data point under a pointer event, or null if it can't be known. */
export function indexFromPointer(clientX, rect, n, { w, padL, padR }) {
    if (!rect || !(rect.width > 0) || !(n >= 2) || !Number.isFinite(clientX)) return null;
    const xSvg = ((clientX - rect.left) / rect.width) * w;
    const frac = (xSvg - padL) / (w - padL - padR);
    return Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
}

/** A timeframe of `points` points is honest if the history covers ≥90% of it. */
export function tfAvailable(points, historyLength) {
    return points == null || historyLength >= 0.9 * points;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
/** '2026-09-25' → 'Sep 25, 2026' (no Date parsing, so no timezone drift). */
export function fmtDay(iso) {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ''));
    if (!m) return String(iso || '');
    return `${MONTHS[Number(m[2]) - 1]} ${Number(m[3])}, ${m[1]}`;
}

const CHOICE_RE = /^(\d{1,2}[MY]|YTD|ALL)$/;
export function readChoice(key) {
    try {
        const v = window.localStorage.getItem(key);
        return v && CHOICE_RE.test(v) ? v : null;
    } catch { return null; }
}
export function saveChoice(key, value) {
    try { window.localStorage.setItem(key, value); } catch { /* not remembered */ }
}

/** viewBox units spanned by `px` CSS pixels on a chart `w` units wide drawn `pxWidth` px wide; null until measured. */
export function unitsForPx(px, w, pxWidth) {
    return Number.isFinite(pxWidth) && pxWidth > 0 ? (px * w) / pxWidth : null;
}

/**
 * Keep labels that do not crowd each other: `items` in priority order, each with `pos`
 * (CSS px along the axis). An item closer than `minGap` to one already kept is dropped.
 */
export function spreadLabels(items, minGap) {
    const kept = [];
    for (const it of items || []) {
        if (!Number.isFinite(it?.pos)) continue;
        if (kept.every((k) => Math.abs(k.pos - it.pos) >= minGap)) kept.push(it);
    }
    return kept;
}

/** CSS left/top that put a label at viewBox point (x, y) whatever size the SVG is drawn. */
export function axisPos(x, y, w, h) {
    const pct = (v) => `${Number((v * 100).toFixed(4))}%`;
    return { left: pct(x / w), top: pct(y / h) };
}

/** Axis labels are 10px JetBrains Mono: about 6.1px per character (600/1000 em, plus slack). */
export const LABEL_CHAR_PX = 6.1;

/**
 * Left gutter in viewBox units: room for the longest y label plus 4px either side, and
 * never narrower than the chart's old gutter `minUnits` (also the answer until measured).
 */
export function gutterFor(labels, w, pxWidth, minUnits) {
    const longest = Math.max(0, ...(labels || []).map((l) => String(l).length));
    if (!longest) return minUnits;
    const u = unitsForPx(longest * LABEL_CHAR_PX + 8, w, pxWidth);
    return u == null ? minUnits : Math.max(minUnits, Math.ceil(u));
}

/** Clamp the centre `x` of a label `halfPx` wide so it stays inside [0, w]; unchanged until measured. */
export function clampUnits(x, halfPx, w, pxWidth) {
    const half = unitsForPx(halfPx, w, pxWidth);
    if (half == null) return x;
    return Math.min(Math.max(x, half), w - half);
}
