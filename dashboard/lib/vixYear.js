import { realizedVol } from './vol';

// How VIX is usually read: under 15 calm, 15–20 normal, 20–30 nervous, 30+ fear.
export const VIX_ZONES = [
    { from: 0, to: 15, name: 'calm', tone: 'good' },
    { from: 15, to: 20, name: 'normal', tone: 'flat' },
    { from: 20, to: 30, name: 'nervous', tone: 'caution' },
    { from: 30, to: Infinity, name: 'fear', tone: 'bad' },
];
export const vixZone = (v) => VIX_ZONES.find((z) => v < z.to);

// Which way SPY went over a month: within ±1% reads flat (gray), so the dotted line doesn't flicker around zero.
export const moveDir = (m) => (!Number.isFinite(m) ? 'flat' : m > 0.01 ? 'up' : m < -0.01 ? 'down' : 'flat');

/**
 * The Volatility card's picture: the 252 VIX closes behind SPY's rank and %ile (+ today's when
 * live), each with its day when /api/vol sends `vixYearDates` (an older saved copy has none), and
 * how much SPY really moved on those days (1-month realized vol from the SPY chart's closes,
 * the same formula as the table's RV21), and which way SPY went over those same 21 sessions (`move`,
 * inked along the dotted line by `ink`: hard color stops halfway between days). The scale always
 * spans 15–20, so the zones read the same in a calm year and a stormy one. null = too little to draw.
 */
export function vixYearGeo({ year, dates, t, spyHistory }) {
    const vals = year || [];
    const dated = Array.isArray(dates) && dates.length === vals.length;
    let rows = vals.map((v, i) => ({ v, d: dated ? dates[i] : null })).filter((r) => Number.isFinite(r.v)).slice(-252);
    if (t.live && Number.isFinite(t.iv)) {
        if (rows.at(-1)?.d && rows.at(-1).d === t.asOf) rows = rows.slice(0, -1); // the live quote replaces its own day
        rows = [...rows, { v: t.iv, d: rows.at(-1)?.d ? t.asOf || null : null }];
    }
    const n = rows.length;
    if (n < 20) return null;

    // SPY's newest bar on or before each VIX day: CBOE's file also has NYSE holidays (Thanksgiving, MLK…),
    // when SPY did not trade, so that day keeps the last move. A bar over 4 days older is not that day's.
    const hist = (spyHistory || []).filter((p) => /^\d{4}-\d{2}-\d{2}$/.test(p?.date || '') && p.price > 0);
    const px = hist.map((p) => p.price);
    let j = -1;
    const bar = rows.map((r) => {
        if (!r.d) return -1;
        while (j + 1 < hist.length && hist[j + 1].date <= r.d) j++;
        return j >= 21 && Date.parse(r.d) - Date.parse(hist[j].date) <= 4 * 864e5 ? j : -1;
    });
    let real = bar.map((k) => (k < 0 ? null : realizedVol(px.slice(k - 21, k + 1))));
    if (real.filter(Number.isFinite).length < 20) real = null; // SPY's days don't line up with VIX's: no line, not a stub
    const move = real && bar.map((k) => (k < 0 ? null : px[k] / px[k - 21] - 1));

    const vix = rows.map((r) => r.v);
    const all = [...vix, ...(real || []).filter(Number.isFinite), 15, 20];
    const lo = Math.min(...all), hi = Math.max(...all);
    const y = (v) => 4 + (1 - (v - lo) / (hi - lo)) * 92; // % from the top, with a little air at both ends
    const x = (i) => (i / (n - 1)) * 100;
    const path = (vs) => vs.map((v, i) => (Number.isFinite(v) ? `${i && Number.isFinite(vs[i - 1]) ? 'L' : 'M'}${x(i).toFixed(2)} ${y(v).toFixed(2)}` : '')).join('');
    const clampY = (v) => Math.min(100, Math.max(0, y(Math.min(hi, Math.max(lo, v)))));
    const zones = VIX_ZONES.filter((z) => z.to > lo && z.from < hi).map((z) => ({ ...z, top: clampY(z.to), bottom: clampY(z.from) }));
    const peak = vix.indexOf(Math.max(...vix));
    const ink = move && move.reduce((runs, m, i) => {
        const dir = moveDir(m), to = Math.min(100, x(i + 0.5));
        if (runs.at(-1)?.dir === dir) runs.at(-1).to = to;
        else runs.push({ dir, from: runs.at(-1)?.to ?? 0, to });
        return runs;
    }, []);
    return { rows, n, vix, real, move, lo, hi, y, x, line: path(vix), realLine: real && path(real), ink, zones, peak };
}
