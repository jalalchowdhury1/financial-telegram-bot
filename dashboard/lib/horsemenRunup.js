/**
 * Four Horsemen — "run-up" maths.
 *
 * The old overlay chart plotted four series on four invisible scales, so the only
 * readable thing was the shape. But the LEVEL is not the tell: in March 2020 jobless
 * claims sat at the 10th percentile of their own history the month the recession
 * started. What actually preceded recessions is the 12-MONTH CHANGE — claims and
 * unemployment had been rising into 6 of the last 8.
 *
 * So these helpers measure, for each horseman, how far it has moved in a year, and
 * compare that with how far it had typically moved by the start of past recessions.
 * Everything is derived from the payload's own history + NBER recession list — no
 * hard-coded thresholds to drift out of date.
 *
 * The yield spread is deliberately NOT run through this: by the time 4 of 6 recessions
 * began it had already re-steepened, so its 12-month change points the wrong way. Its
 * tell is the inversion a year or more earlier — see lastInversion.
 */

const YEAR_MS = 365.25 * 86400000;
const ms = (d) => new Date(`${String(d).slice(0, 10)}T00:00:00Z`).getTime();
/** Exactly one calendar year earlier — a 365.25-day offset lands hours off and can
 *  fall just outside a monthly series' first observation. */
const yearBefore = (t) => { const d = new Date(t); d.setUTCFullYear(d.getUTCFullYear() - 1); return d.getTime(); };

/** Last observation on or before `when` (never a future one). Null before the series starts. */
export function valueAt(history, when) {
    if (!history?.length) return null;
    let out = null;
    for (const p of history) {
        if (p?.value == null) continue;
        if (ms(p.date) <= when) out = p.value; else break;
    }
    return out;
}

/**
 * Change over the 12 months ending at `when`.
 * mode 'pp'  -> difference in the units themselves (percentage points)
 * mode 'pct' -> percentage change
 * Null unless the series covers both ends.
 */
export function changeOver(history, when, mode) {
    const now = valueAt(history, when);
    const then = valueAt(history, yearBefore(when));
    if (now == null || then == null) return null;
    if (mode === 'pct') return then === 0 ? null : 100 * (now / then - 1);
    return now - then;
}

/** The 12-month change as each past recession began, for the recessions the series covers. */
export function preRecessionRunups(history, recessions, mode, minYear = 1970) {
    if (!history?.length || !recessions?.length) return [];
    return recessions
        .filter((r) => Number(String(r.start).slice(0, 4)) >= minYear)
        .map((r) => ({ start: r.start, change: changeOver(history, ms(r.start), mode) }))
        .filter((r) => r.change != null);
}

export function runupMedian(runups) {
    const xs = (runups || []).map((r) => r.change).filter((x) => x != null).sort((a, b) => a - b);
    if (!xs.length) return null;
    const m = Math.floor(xs.length / 2);
    return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2;
}

/**
 * 'improving'      — moving the healthy way
 * 'watch'          — moving the wrong way, but short of the typical pre-recession move
 * 'recession-like' — has moved as far as it usually had by the start of a recession
 * worseIsUp: true for claims/unemployment/bankruptcies, false for a series where a FALL is bad.
 */
export function horsemanStatus(change, median, worseIsUp = true) {
    if (change == null || median == null) return 'unknown';
    const sign = worseIsUp ? 1 : -1;
    if (change * sign <= 0) return 'improving';
    return change * sign >= median * sign ? 'recession-like' : 'watch';
}

/** How far a print may sit from the exact year-ago date and still count as "a year ago".
 *  Weekly claims land 1-2 days off (52 weeks = 364 days), daily series a few days off over
 *  holidays; a missing month or quarter is 30+ days off and must give no answer. */
const YEAR_AGO_TOLERANCE_MS = 7 * 86400000;

/**
 * The latest print against the print one calendar year before THAT print's date.
 * One helper for the card header and the run-up rail, so they cannot disagree.
 * Anchoring at the print (not today) keeps a months-old quarterly series honest, and
 * picking the print nearest the year-ago date (±7 days) never stretches a gap in the
 * series into a 13-month change labelled "1y" (UNRATE has no Oct-2025 print).
 * mode 'pp' -> difference in the units; 'pct' -> percentage change. Null when unsure.
 */
export function latestYoY(history, mode) {
    const pts = (history || []).filter((p) => p?.date && p.value != null && Number.isFinite(Number(p.value)));
    if (pts.length < 2) return null;
    const last = pts[pts.length - 1];
    const target = yearBefore(ms(last.date));
    let then = null, gap = Infinity;
    for (const p of pts) {
        const g = Math.abs(ms(p.date) - target);
        if (g < gap) { gap = g; then = p; }
    }
    if (!then || then === last || gap > YEAR_AGO_TOLERANCE_MS) return null;
    const a = Number(last.value), b = Number(then.value);
    if (mode === 'pct') return b === 0 ? null : 100 * (a / b - 1);
    return a - b;
}

/**
 * Why latestYoY gave no answer, when the reason is a hole in the series rather than a short
 * one: the missing year-ago print as 'Oct 2025' (monthly or slower) or 'Sep 26, 2025'
 * (weekly/daily). Null when latestYoY has an answer, or the series starts after that date.
 */
export function yearAgoGap(history) {
    const pts = (history || []).filter((p) => p?.date && p.value != null && Number.isFinite(Number(p.value)));
    if (pts.length < 2) return null;
    const last = pts[pts.length - 1];
    const target = yearBefore(ms(last.date));
    if (ms(pts[0].date) > target || latestYoY(pts, 'pp') != null) return null;
    const weekly = ms(last.date) - ms(pts[pts.length - 2].date) < 25 * 86400000;
    return new Date(target).toLocaleDateString('en-US', weekly
        ? { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }
        : { month: 'short', year: 'numeric', timeZone: 'UTC' });
}

/** A positive stretch shorter than this between two negative spells is a blip, not the
 *  end of the inversion (T10Y2Y re-steepened 2024-08-27, then dipped for single days on
 *  2024-09-03 and 2024-09-05 — that is still the 2022 inversion ending, not a new one). */
const INVERSION_GAP_MS = 90 * 86400000;

/** The most recent stretch of a negative (inverted) spread, and how long since it ended. */
export function lastInversion(history, now = Date.now()) {
    if (!history?.length) return null;
    const pts = history.filter((p) => p?.value != null);
    let end = null, start = null, crossedPositive = false;
    for (let i = pts.length - 1; i >= 0; i -= 1) {
        if (pts[i].value < 0) {
            if (end == null) end = pts[i].date;
            // An earlier negative spell joins only across a short positive gap.
            else if (crossedPositive && ms(start) - ms(pts[i].date) >= INVERSION_GAP_MS) break;
            start = pts[i].date;
            crossedPositive = false;
        } else if (end != null) crossedPositive = true;
    }
    if (end == null) return null;
    const currentlyInverted = pts[pts.length - 1].value < 0;
    return {
        start, end,
        startYear: Number(String(start).slice(0, 4)),
        endYear: Number(String(end).slice(0, 4)),
        monthsSince: currentlyInverted ? 0 : Math.round((now - ms(end)) / (YEAR_MS / 12)),
        currentlyInverted,
    };
}
