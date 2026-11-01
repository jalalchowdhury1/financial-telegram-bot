/**
 * The SPY chart's valuation lenses (components/ValuationLens.js). Pure.
 *
 * Forward P/E every trading day from the weekly FactSet rows (lib/fwdPe.js): a row's P/E sits on the last
 * SPY close before its report date, which pins SPY's "forward earnings" that day (k = SPY ÷ P/E, in SPY
 * dollars). Between two reports k moves in a straight line; after the newest it holds, so the line keeps
 * moving with SPY until the next report. On a report's own close day the line equals FactSet's number.
 * a5/a10 = the 5- and 10-yr averages that report printed, held until the next (null before FactSet printed one).
 */

const DAY = 864e5;
const ms = (iso) => Date.parse(`${iso}T00:00:00Z`);

/** `[{date, price, eps, pe, a5, a10}]` on every SPY day from the first anchored report; [] when none anchors. */
export function fwdDaily(rows, spy) {
    const anchors = [];
    let i = 0;
    for (const r of rows || []) {
        while (i + 1 < spy.length && spy[i + 1].date < r.date) i++;
        const s = spy[i];
        if (!s || s.date >= r.date || ms(r.date) - ms(s.date) > 5 * DAY || !(r.pe > 0)) continue;
        if (anchors.length && anchors[anchors.length - 1].i === i) anchors.pop(); // two reports, one close: the newer wins
        anchors.push({ i, t: ms(s.date), k: s.price / r.pe, a5: r.a5 ?? null, a10: r.a10 ?? null });
    }
    if (!anchors.length) return [];
    const out = [];
    for (let j = anchors[0].i, a = 0; j < spy.length; j++) {
        const t = ms(spy[j].date);
        while (a + 1 < anchors.length && anchors[a + 1].t <= t) a++;
        const A = anchors[a], B = anchors[a + 1];
        const k = B ? A.k + ((B.k - A.k) * (t - A.t)) / (B.t - A.t) : A.k;
        out.push({ date: spy[j].date, price: spy[j].price, eps: k, pe: spy[j].price / k, a5: A.a5, a10: A.a10 });
    }
    return out;
}

/**
 * Is the last reading the lowest (or highest) in at least `minDays`? `{low, since}`: since = the date of the
 * last reading past it, null when it is the extreme of the whole series. null when neither.
 */
export function extremeSince(points, key = 'value', minDays = 90) {
    const n = points?.length || 0;
    if (n < 2) return null;
    const v = points[n - 1][key], end = ms(points[n - 1].date);
    for (const low of [true, false]) {
        let j = n - 2;
        while (j >= 0 && (low ? points[j][key] >= v : points[j][key] <= v)) j--;
        const since = j < 0 ? null : points[j].date;
        if (since == null ? end - ms(points[0].date) >= minDays * DAY : end - ms(since) >= minDays * DAY) return { low, since };
    }
    return null;
}

/** Time-weighted mean of `[{date, value}]` from `fromIso` to the last point (each value holds until the next). */
export function timeAvg(points, fromIso) {
    let sum = 0, span = 0;
    for (let j = 0; j < points.length - 1; j++) {
        const a = Math.max(ms(points[j].date), ms(fromIso)), b = ms(points[j + 1].date);
        if (b > a) { sum += points[j].value * (b - a); span += b - a; }
    }
    return span ? sum / span : null;
}

/** SPY's move over a window split into earnings and P/E: (1 + price) = (1 + eps)(1 + pe). */
export function decompose(first, last) {
    return { eps: last.eps / first.eps - 1, pe: last.pe / first.pe - 1, price: last.price / first.price - 1 };
}

/** The "Then what?" study's points: how many turns the forward P/E sits under FactSet's 5-yr average then. */
export function underAvg(rows) {
    return (rows || []).filter((r) => r.a5 != null).map((r) => ({ date: r.date, value: Math.round((r.a5 - r.pe) * 10) / 10, pe: r.pe, a5: r.a5 }));
}
