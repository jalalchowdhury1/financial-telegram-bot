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

/** The most recent stretch of a negative (inverted) spread, and how long since it ended. */
export function lastInversion(history, now = Date.now()) {
    if (!history?.length) return null;
    const pts = history.filter((p) => p?.value != null);
    let end = null, start = null;
    for (let i = pts.length - 1; i >= 0; i -= 1) {
        if (pts[i].value < 0) { if (end == null) end = pts[i].date; start = pts[i].date; }
        else if (end != null) break;
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
