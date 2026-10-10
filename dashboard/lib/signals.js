/**
 * NotSoBoring + FrontRunner computed from daily closes (added 2026-10-10), so the two
 * strategy pills no longer depend on Google Sheets being fed. The sheets stay as backups
 * (lib/sheetsCascade.js).
 *
 * Both are exact ports of the owner's sheets, proven against them:
 *  - FrontRunner = "Robinhood - Ultimate Frontrunners" Sheet1!K2: a decision tree over 16
 *    RSIs. The sheet's RSI inputs (column G) are typed in by an n8n job that stopped on
 *    2026-08-24; Wilder RSI from Nasdaq closes reproduced all 16 of that day's inputs to
 *    the cent. Inputs are rounded to 2 dp before comparing, like the sheet's.
 *  - NotSoBoring = "TMF - TQQQ - Composer Daily Triggers" 'Trading Days'!I2: TMF's 10-day
 *    max/min drawdown vs 8.5%, an ±8% open-to-open "big move" check, and "nothing changed →
 *    keep yesterday's state".
 *
 * Price data: Nasdaq historical (keyless, reachable from Vercel) → CNBC daily bars (~2 y, raw,
 * equal to Nasdaq's) → Yahoo (raw closes; often blocked from Vercel), plus
 * ONE CNBC quote call for all 17 tickers that adds today's session as a bar (same rule as
 * the SPY RSI: the spot's session counts). If any ticker lacks today's bar, nobody gets one,
 * so the 16 RSIs always describe the same day.
 *
 * Fault switches (`?_fail=`): signals (whole tier off), signals_nasdaq, signals_cnbc,
 * signals_yahoo, signals_spot.
 */
import { calculateRSI } from './finance';
import { nasdaqHistory, cnbcHistory, yahooChart, cnbcQuotes } from './sources';
import { gate } from './faults';
import { latestCompletedSessionDate } from './marketClock';

// Sheet1 rows 2..17 (H2..H17) in order: [ticker, RSI window].
export const FR_INPUTS = [
    ['QQQ', 9], ['VIXY', 50], ['SPY', 9], ['IOO', 9], ['XLP', 9], ['VTV', 9], ['XLF', 9], ['VOX', 9],
    ['CURE', 9], ['RETL', 9], ['LABU', 9], ['SOXL', 9], ['FNGU', 9], ['TQQQ', 9], ['TECL', 9], ['UPRO', 9],
];
const G15 = '1.5x VIX Group (VXX, UVIX)';
const BLEND = 'VIX Blend (VXX=0.45, VIXM=0.2, UVIX=0.35)';
const VIXY = '1x VIX (VIXY)';

/** Sheet1!K2, verbatim (the trailing digits are the sheet's branch ids; the UI strips them). */
export function frontRunnerDecision(h) {
    const [H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15, H16, H17] = h;
    if (H2 > 79) return H3 > 40 ? `${G15}10` : H4 > 82.5 ? `${G15}1` : BLEND;
    if (H4 > 79) return H3 > 40 ? `${G15}11` : H2 > 82.5 ? `${G15}2` : BLEND;
    if (H5 > 80) return H3 > 40 ? `${G15}12` : H5 > 82.5 ? `${G15}3` : VIXY;
    if (H6 > 77) return H6 > 82.5 ? `${G15}4` : VIXY;
    if (H7 > 79) return H7 > 82.5 ? `${G15}5` : VIXY;
    if (H8 > 81) return H8 > 85 ? `${G15}6` : VIXY;
    if (H9 > 84) return H9 > 87 ? `${G15}7` : VIXY;
    if (H10 > 82) return H10 > 85 ? `${G15}8` : VIXY;
    if (H11 > 82) return H11 > 85 ? `${G15}9` : VIXY;
    if (H12 > 87) return 'LABD';
    if (H13 < 25) return 'SOXL';
    if (H14 < 25) return 'FNGU';
    if (H15 < 28) return 'Select bottom 2 RSI assets from (SOXL, TECL, TQQQ, FNGU)';
    if (H16 < 25) return 'TECL';
    if (H17 < 25) return 'UPRO';
    return 'BIL (T-Bill ETF)1';
}

const r2 = (x) => Math.round(x * 100) / 100;

/** {ticker: ascending [{date, price}]} -> {value, inputs, asOf}; asOf = the oldest last bar. */
export function computeFrontRunner(histories) {
    const inputs = {};
    let asOf = null;
    const h = FR_INPUTS.map(([t, n]) => {
        const bars = histories[t];
        if (!bars || bars.length < n * 5) throw new Error(`FrontRunner: too little history for ${t} (${bars?.length || 0} bars)`);
        const v = r2(calculateRSI(bars, n));
        inputs[`${t}${n === 9 ? '' : `_${n}`}`] = v;
        const last = bars[bars.length - 1].date;
        if (!asOf || last < asOf) asOf = last;
        return v;
    });
    return { value: frontRunnerDecision(h), inputs, asOf };
}

/** One 'Trading Days' row's state from the row itself (i) and the row before it in time (i+1). */
function nsbState(cur, prev, prevState) {
    if (cur.D === prev.D && cur.E === prev.E && cur.H === 'Normal') return prevState;
    if (cur.G < 0.085) return 'ON';
    if (cur.G > 0.085 && cur.E <= prev.E && cur.D <= prev.D) return 'OFF';
    return cur.H !== 'Negative Big Move' ? 'ON > 8.5%' : 'NEG MOVE';
}

/**
 * TMF ascending [{date, open, price}] -> {value, asOf, detail}. Walks oldest → newest so the
 * carry-over ("D, E unchanged and a normal day → yesterday's state") has a past; run from two
 * opposite starting states, it must land on the same answer or it throws (not converged).
 */
export function computeNotSoBoring(bars) {
    if (!bars || bars.length < 60) throw new Error(`NotSoBoring: too little TMF history (${bars?.length || 0} bars)`);
    const rows = [];
    for (let i = 9; i < bars.length; i++) {
        const win = bars.slice(i - 9, i + 1).map((b) => b.price);
        const D = Math.max(...win);
        const E = Math.min(...win);
        const o = bars[i].open;
        const po = bars[i - 1].open;
        const mv = Number.isFinite(o) && Number.isFinite(po) && po ? (o - po) / po : 0;
        rows.push({ date: bars[i].date, D, E, G: (D - E) / D, H: mv > 0.08 ? 'Positive Big Move' : mv < -0.08 ? 'Negative Big Move' : 'Normal' });
    }
    const run = (start) => {
        let st = start;
        for (let i = 1; i < rows.length; i++) st = nsbState(rows[i], rows[i - 1], st);
        return st;
    };
    const a = run('ON');
    const b = run('OFF');
    if (a !== b) throw new Error('NotSoBoring: state did not converge over the history');
    const last = rows[rows.length - 1];
    return {
        value: a,
        asOf: last.date,
        detail: { max10: last.D, min10: last.E, drawdownPct: r2(last.G * 100), bigMove: last.H, watchOutAt: r2(last.D * (1 - 0.083)) },
    };
}

/**
 * Today's session as a bar from a live quote {price, open, asOf}. A quote dated after the
 * last bar is appended (open = the session's real open when the quote has one); a quote
 * for the last bar's date replaces that bar's close only while that session is still
 * open (a partial row). A completed session's official bar is never overwritten.
 */
export function withSpot(bars, q, expected) {
    if (!bars?.length || !q || !Number.isFinite(q.price) || q.price <= 0 || !q.asOf) return bars;
    const last = bars[bars.length - 1];
    if (q.asOf > last.date) return [...bars, { date: q.asOf, price: q.price, open: Number.isFinite(q.open) && q.open > 0 ? q.open : q.price }];
    if (q.asOf === last.date && last.date > expected) return [...bars.slice(0, -1), { ...last, price: q.price }];
    return bars;
}

/** Daily bars for one ticker: Nasdaq → CNBC → Yahoo (all raw closes). Returns {bars, source}. */
async function barsFor(ticker, { faults, nasdaq, cnbcBars, yahoo, messages }) {
    const tiers = [
        ['Nasdaq', 'signals_nasdaq', () => nasdaq(ticker, { years: 3, revalidate: 3600, timeout: 6000 })],
        ['CNBC', 'signals_cnbc', () => cnbcBars(ticker, { revalidate: 3600, timeout: 6000, tries: 1 })],
        ['Yahoo', 'signals_yahoo', async () => (await yahoo(ticker, { range: '5y', interval: '1d', revalidate: 3600, adjusted: false, tries: 1, timeout: 6000 })).history],
    ];
    for (const [source, faultName, fetchBars] of tiers) {
        try {
            const bars = await gate(faultName, faults, fetchBars);
            if (bars.length >= 60) return { bars, source };
            messages.push(`${ticker}: ${source} gave ${bars.length} bars`);
        } catch (e) { messages.push(`${ticker}: ${source} failed (${String(e?.message).slice(0, 80)})`); }
    }
    throw new Error('every price source failed');
}

const lastDate = (bars) => bars[bars.length - 1].date;

/**
 * A group's bars with today's quote added — all or none: if the quotes don't bring every
 * ticker to the same newest date, the plain closes are used.
 */
function groupBars(tickers, got, spot, expected) {
    const plain = Object.fromEntries(tickers.map((t) => [t, got[t].bars]));
    if (!spot) return { bars: plain, spot: false };
    const lifted = Object.fromEntries(tickers.map((t) => [t, withSpot(got[t].bars, spot[t], expected)]));
    const dates = new Set(tickers.map((t) => lastDate(lifted[t])));
    const changed = tickers.some((t) => lifted[t] !== plain[t]);
    return dates.size === 1 && changed ? { bars: lifted, spot: true } : { bars: plain, spot: false };
}

/**
 * Both signals -> { NotSoBoring, FrontRunner, messages }; each is
 * {value, source, live, stale, asOf, intraday, detail} or null. "live" = built from at
 * least the latest completed session; "intraday" = includes today's unfinished session.
 * Never throws.
 */
export async function resolveSignals({ faults = new Set(), now = Date.now(), nasdaq = nasdaqHistory, cnbcBars = cnbcHistory, yahoo = yahooChart, cnbc = cnbcQuotes } = {}) {
    const messages = [];
    const out = { NotSoBoring: null, FrontRunner: null, messages };
    if (faults.has('signals')) { messages.push('computed signals: switched off (_fail=signals)'); return out; }
    const expected = latestCompletedSessionDate(now);
    const tickers = [...new Set([...FR_INPUTS.map(([t]) => t), 'TMF'])];
    const got = {};
    const spotP = gate('signals_spot', faults, () => cnbc(tickers, { revalidate: 120, timeout: 5000, tries: 1 }))
        .catch((e) => { messages.push(`CNBC live quotes failed (${String(e?.message).slice(0, 80)}): closes only`); return null; });
    await Promise.all(tickers.map(async (t) => {
        try { got[t] = await barsFor(t, { faults, nasdaq, cnbcBars, yahoo, messages }); }
        catch (e) { messages.push(`${t}: no daily bars (${String(e?.message).slice(0, 80)})`); }
    }));
    const spot = await spotP;

    const run = (field, group, compute) => {
        try {
            const missing = group.filter((t) => !got[t]);
            if (missing.length) throw new Error(`no daily bars for ${missing.join(', ')}`);
            const g = groupBars(group, got, spot, expected);
            const r = compute(g.bars);
            const live = r.asOf >= expected;
            const intraday = r.asOf > expected;
            const srcs = [...new Set(group.map((t) => got[t].source))].join('+');
            const when = intraday ? `intraday ${r.asOf}, live CNBC prices` : `${r.asOf} close`;
            out[field] = {
                value: r.value,
                source: `Computed from ${srcs}${g.spot ? ' + CNBC' : ''} prices (${when})${live ? '' : `; latest session is ${expected}`}`,
                live,
                stale: !live,
                asOf: r.asOf,
                intraday,
                detail: r.detail || r.inputs,
            };
            messages.push(`${field}: ${r.value} computed (${when}${live ? '' : ', STALE'})`);
        } catch (e) { messages.push(`${field} compute failed: ${String(e?.message).slice(0, 120)}`); }
    };
    run('FrontRunner', FR_INPUTS.map(([t]) => t), computeFrontRunner);
    run('NotSoBoring', ['TMF'], (b) => computeNotSoBoring(b.TMF));
    return out;
}
