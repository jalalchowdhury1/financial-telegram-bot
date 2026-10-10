/**
 * lib/signals.js: NotSoBoring + FrontRunner computed from daily prices (the sheets' exact
 * logic), the live "today" bar, and the source cascade.
 */
import { frontRunnerDecision, computeFrontRunner, computeNotSoBoring, withSpot, resolveSignals, FR_INPUTS } from '../signals';

// Real TMF bars (Nasdaq, [date, open, close]) 2026-06-15 → 2026-10-09.
const TMF = [
    ['2026-06-15', 35.45, 35.17], ['2026-06-16', 35.47, 35.73], ['2026-06-17', 35.92, 35.86], ['2026-06-18', 36.79, 36.36], ['2026-06-22', 35.81, 35.58],
    ['2026-06-23', 35.23, 35.36], ['2026-06-24', 36.45, 36.74], ['2026-06-25', 36.96, 36.7], ['2026-06-26', 36.27, 36.65], ['2026-06-29', 36.68, 36.75],
    ['2026-06-30', 36.41, 35.55], ['2026-07-01', 34.73, 34.8], ['2026-07-02', 34.56, 34.78], ['2026-07-06', 34.61, 34.62], ['2026-07-07', 34.2, 33.52],
    ['2026-07-08', 33.26, 33.31], ['2026-07-09', 33.2, 33.44], ['2026-07-10', 33.435, 33.43], ['2026-07-13', 33.04, 32.81], ['2026-07-14', 33.02, 32.92],
    ['2026-07-15', 32.92, 33.05], ['2026-07-16', 32.6, 33.04], ['2026-07-17', 33.5, 33.35], ['2026-07-20', 33.09, 32.66], ['2026-07-21', 32.42, 32.33],
    ['2026-07-22', 32.29, 32.06], ['2026-07-23', 31.55, 31.78], ['2026-07-24', 31.88, 31.85], ['2026-07-27', 32.38, 32.39], ['2026-07-28', 32.52, 32.91],
    ['2026-07-29', 32.77, 31.31], ['2026-07-30', 31.24, 31.24], ['2026-07-31', 30.83, 30.59], ['2026-08-03', 30.96, 30.88], ['2026-08-04', 31.37, 31.56],
    ['2026-08-05', 31.72, 31.78], ['2026-08-06', 31.52, 31.22], ['2026-08-07', 31.52, 31.43], ['2026-08-10', 31.06, 30.64], ['2026-08-11', 30.9, 30.81],
    ['2026-08-12', 31.08, 30.72], ['2026-08-13', 31.25, 31.24], ['2026-08-14', 30.81, 30.59], ['2026-08-17', 30.27, 29.83], ['2026-08-18', 29.71, 30.12],
    ['2026-08-19', 31.37, 31.59], ['2026-08-20', 30.86, 30.87], ['2026-08-21', 30.61, 30.5], ['2026-08-24', 30.99, 31.08], ['2026-08-25', 31.65, 32.05],
    ['2026-08-26', 31.83, 31.84], ['2026-08-27', 31.75, 31.64], ['2026-08-28', 31.53, 31.33], ['2026-08-31', 30.87, 30.93], ['2026-09-01', 30.58, 30.55],
    ['2026-09-02', 30.66, 30.61], ['2026-09-03', 31.08, 30.77], ['2026-09-04', 30.98, 30.88], ['2026-09-08', 31.12, 30.85], ['2026-09-09', 30.85, 30.34],
    ['2026-09-10', 29.49, 29.3], ['2026-09-11', 29.83, 29.31], ['2026-09-14', 29.29, 29.42], ['2026-09-15', 29.05, 29.16], ['2026-09-16', 29.4, 29.34],
    ['2026-09-17', 30.02, 30.29], ['2026-09-18', 29.89, 29.64], ['2026-09-21', 30.17, 30.23], ['2026-09-22', 30, 29.86], ['2026-09-23', 29.53, 28.44],
    ['2026-09-24', 28.3, 27.35], ['2026-09-25', 27.23, 27.21], ['2026-09-28', 26.48, 26.49], ['2026-09-29', 26.37, 26.07], ['2026-09-30', 26.02, 25.68],
    ['2026-10-01', 25.09, 25.82], ['2026-10-02', 26.06, 25.58], ['2026-10-05', 25.26, 25.19], ['2026-10-06', 25.26, 25.36], ['2026-10-07', 24.55, 25.23],
    ['2026-10-08', 25.26, 25.9], ['2026-10-09', 25.76, 25.99],
].map(([date, open, price]) => ({ date, open, price }));

// The sheet's 'Trading Days' I column for the rows with a full 10-day window (xlsx export 10 Oct 2026).
const SHEET_I = {
    '2026-09-30': 'OFF', '2026-10-01': 'OFF', '2026-10-02': 'OFF', '2026-10-05': 'OFF',
    '2026-10-06': 'OFF', '2026-10-07': 'ON', '2026-10-08': 'ON', '2026-10-09': 'ON',
};

describe('FrontRunner decision tree (Sheet1!K2)', () => {
    const base = () => Array(16).fill(50);
    const at = (pairs) => { const h = base(); for (const [i, v] of pairs) h[i] = v; return h; };
    test('the sheet\'s frozen 2026-08-24 inputs → BIL', () => {
        expect(frontRunnerDecision([41.88, 40.86, 47.83, 47.29, 63.9, 62.23, 61.31, 53.83, 69.43, 49.32, 58.89, 34.4, 47.95, 40.15, 36.56, 45.91])).toBe('BIL (T-Bill ETF)1');
    });
    test('branches', () => {
        expect(frontRunnerDecision(at([[0, 80], [1, 41]]))).toBe('1.5x VIX Group (VXX, UVIX)10');
        expect(frontRunnerDecision(at([[0, 80], [1, 30], [2, 83]]))).toBe('1.5x VIX Group (VXX, UVIX)1');
        expect(frontRunnerDecision(at([[0, 80], [1, 30]]))).toBe('VIX Blend (VXX=0.45, VIXM=0.2, UVIX=0.35)');
        // The sheet's "H2 > 82.5" inside the H4 > 79 arm can never fire (H2 > 79 was caught first).
        expect(frontRunnerDecision(at([[2, 80], [1, 30]]))).toBe('VIX Blend (VXX=0.45, VIXM=0.2, UVIX=0.35)');
        expect(frontRunnerDecision(at([[2, 80], [1, 41]]))).toBe('1.5x VIX Group (VXX, UVIX)11');
        expect(frontRunnerDecision(at([[3, 81], [1, 30]]))).toBe('1x VIX (VIXY)');
        expect(frontRunnerDecision(at([[4, 83]]))).toBe('1.5x VIX Group (VXX, UVIX)4');
        expect(frontRunnerDecision(at([[10, 88]]))).toBe('LABD');
        expect(frontRunnerDecision(at([[11, 24]]))).toBe('SOXL');
        expect(frontRunnerDecision(at([[13, 27]]))).toBe('Select bottom 2 RSI assets from (SOXL, TECL, TQQQ, FNGU)');
        expect(frontRunnerDecision(at([[15, 24]]))).toBe('UPRO');
        expect(frontRunnerDecision(base())).toBe('BIL (T-Bill ETF)1');
    });
    test('a threshold is strict: exactly 79 does not trigger', () => {
        expect(frontRunnerDecision(at([[0, 79]]))).toBe('BIL (T-Bill ETF)1');
    });
});

// Synthetic ascending daily series ending on `end`.
function series(n, end, f) {
    const out = []; const t = Date.parse(`${end}T12:00:00Z`);
    for (let i = n - 1; i >= 0; i--) out.push({ date: new Date(t - i * 864e5).toISOString().slice(0, 10), price: f(n - 1 - i), open: f(n - 1 - i) });
    return out;
}
const wavy = (end, n = 600) => series(n, end, (i) => 100 + 10 * Math.sin(i / 7) + i * 0.01);

describe('computeFrontRunner', () => {
    test('16 inputs rounded to 2 dp, asOf = the oldest last bar', () => {
        const h = Object.fromEntries(FR_INPUTS.map(([t]) => [t, wavy(t === 'XLP' ? '2026-10-08' : '2026-10-09')]));
        const r = computeFrontRunner(h);
        expect(Object.keys(r.inputs)).toHaveLength(16);
        expect(r.inputs).toHaveProperty('VIXY_50');
        for (const v of Object.values(r.inputs)) expect(Math.round(v * 100) / 100).toBe(v);
        expect(r.asOf).toBe('2026-10-08');
        expect(typeof r.value).toBe('string');
    });
    test('throws when a ticker has too little history', () => {
        const h = Object.fromEntries(FR_INPUTS.map(([t]) => [t, wavy('2026-10-09')]));
        h.VIXY = h.VIXY.slice(-100); // RSI(50) needs 250 bars
        expect(() => computeFrontRunner(h)).toThrow(/VIXY/);
    });
});

describe('computeNotSoBoring (\'Trading Days\'!I2)', () => {
    test('matches the sheet day by day on real TMF prices, incl. the OFF → ON flip on 7 Oct', () => {
        for (const [d, want] of Object.entries(SHEET_I)) {
            expect([d, computeNotSoBoring(TMF.filter((b) => b.date <= d)).value]).toEqual([d, want]);
        }
    });
    test('detail: 10-day max/min, drawdown, watch-out level (sheet J = D × (1 − 8.3%))', () => {
        const r = computeNotSoBoring(TMF);
        expect(r.asOf).toBe('2026-10-09');
        expect(r.detail).toMatchObject({ max10: 26.49, min10: 25.19, drawdownPct: 4.91, bigMove: 'Normal', watchOutAt: 24.29 });
    });
    test('a fresh slide to new lows past 8.5% → OFF', () => {
        const slide = series(80, '2026-10-09', (i) => (i < 70 ? 100 : 100 - (i - 69) * 1.5));
        expect(computeNotSoBoring(slide).value).toBe('OFF');
    });
    // The 10-day low (85) rolls out of the window while the drawdown from 100 is still 10%.
    const lowRollsOff = series(79, '2026-10-09', (i) => (i < 68 ? 95 : i === 68 ? 85 : i === 69 ? 100 : 90));
    test('drawdown > 8.5% but the 10-day low is rising → ON > 8.5%', () => {
        expect(computeNotSoBoring(lowRollsOff).detail).toMatchObject({ max10: 100, min10: 90, drawdownPct: 10 });
        expect(computeNotSoBoring(lowRollsOff).value).toBe('ON > 8.5%');
    });
    test('same, on a day that opens > 8% below the previous open → NEG MOVE', () => {
        const r = computeNotSoBoring([...lowRollsOff.slice(0, -1), { ...lowRollsOff.at(-1), open: 82 }]);
        expect(r.detail.bigMove).toBe('Negative Big Move');
        expect(r.value).toBe('NEG MOVE');
    });
    test('needs 60 bars', () => {
        expect(() => computeNotSoBoring(TMF.slice(-30))).toThrow(/too little/);
    });
});

describe('withSpot (today\'s session as a bar)', () => {
    const bars = [{ date: '2026-10-08', price: 10, open: 9.9 }, { date: '2026-10-09', price: 11, open: 10.5 }];
    test('a newer quote is appended, with the session\'s real open', () => {
        expect(withSpot(bars, { price: 12, open: 11.2, asOf: '2026-10-12' }, '2026-10-09').at(-1)).toEqual({ date: '2026-10-12', price: 12, open: 11.2 });
        expect(withSpot(bars, { price: 12, asOf: '2026-10-12' }, '2026-10-09').at(-1).open).toBe(12);
    });
    test('a partial row for an unfinished session gets the live price; a completed bar is never overwritten', () => {
        expect(withSpot(bars, { price: 12, asOf: '2026-10-09' }, '2026-10-08').at(-1)).toEqual({ date: '2026-10-09', price: 12, open: 10.5 });
        expect(withSpot(bars, { price: 12, asOf: '2026-10-09' }, '2026-10-09')).toBe(bars);
        expect(withSpot(bars, null, '2026-10-09')).toBe(bars);
    });
});

describe('resolveSignals', () => {
    const TICKERS = [...FR_INPUTS.map(([t]) => t), 'TMF'];
    // Mon 12 Oct 2026 14:00 ET: in session, latest completed session = Fri 9 Oct.
    const NOW = Date.parse('2026-10-12T18:00:00Z');
    const nasdaq = jest.fn(async (t) => (t === 'TMF' ? TMF : wavy('2026-10-09')));
    const quotes = (skip) => Object.fromEntries(TICKERS.filter((t) => t !== skip).map((t) => [t, { price: 100, open: 99, asOf: '2026-10-12' }]));

    test('Nasdaq closes + one CNBC call for today → live, intraday', async () => {
        const cnbc = jest.fn(async () => quotes());
        const r = await resolveSignals({ now: NOW, nasdaq, cnbc, yahoo: jest.fn() });
        expect(cnbc).toHaveBeenCalledTimes(1);
        expect(cnbc.mock.calls[0][0]).toHaveLength(17);
        expect(r.FrontRunner).toMatchObject({ live: true, stale: false, intraday: true, asOf: '2026-10-12' });
        expect(r.FrontRunner.source).toBe('Computed from Nasdaq + CNBC prices (intraday 2026-10-12, live CNBC prices)');
        expect(r.NotSoBoring).toMatchObject({ live: true, intraday: true, asOf: '2026-10-12' });
    });

    test('one ticker without a quote → nobody gets today\'s bar (16 RSIs stay on one day)', async () => {
        const r = await resolveSignals({ now: NOW, nasdaq, cnbc: async () => quotes('SOXL'), yahoo: jest.fn() });
        expect(r.FrontRunner).toMatchObject({ live: true, intraday: false, asOf: '2026-10-09' });
        expect(r.FrontRunner.source).toBe('Computed from Nasdaq prices (2026-10-09 close)');
        expect(r.NotSoBoring.intraday).toBe(true); // TMF's own quote is there
    });

    test('Nasdaq down for a ticker → Yahoo raw closes; CNBC down → closes only', async () => {
        const yahoo = jest.fn(async () => ({ history: wavy('2026-10-09') }));
        const r = await resolveSignals({ now: NOW, nasdaq: async (t) => { if (t === 'SPY') throw new Error('403'); return nasdaq(t); }, yahoo, cnbc: async () => { throw new Error('down'); } });
        expect(yahoo).toHaveBeenCalledWith('SPY', expect.objectContaining({ adjusted: false }));
        expect(r.FrontRunner.source).toBe('Computed from Nasdaq+Yahoo prices (2026-10-09 close)');
        expect(r.messages.join(' | ')).toMatch(/CNBC live quotes failed/);
    });

    test('closes older than the latest completed session → stale', async () => {
        const r = await resolveSignals({ now: Date.parse('2026-10-13T22:00:00Z'), nasdaq, cnbc: async () => ({}), yahoo: jest.fn() });
        expect(r.FrontRunner).toMatchObject({ live: false, stale: true });
        expect(r.FrontRunner.source).toMatch(/latest session is 2026-10-13/);
    });

    test('a ticker with no data at all → that signal is null, the other still computes', async () => {
        const r = await resolveSignals({ now: NOW, nasdaq: async (t) => { if (t === 'VIXY') throw new Error('x'); return nasdaq(t); }, yahoo: async () => { throw new Error('y'); }, cnbc: async () => ({}) });
        expect(r.FrontRunner).toBeNull();
        expect(r.NotSoBoring.value).toBe('ON');
        expect(r.messages.join(' | ')).toMatch(/FrontRunner compute failed: no daily bars for VIXY/);
    });

    test('fault switches: signals turns the tier off; signals_nasdaq forces Yahoo', async () => {
        expect(await resolveSignals({ faults: new Set(['signals']), now: NOW, nasdaq })).toMatchObject({ NotSoBoring: null, FrontRunner: null });
        const yahoo = jest.fn(async (t) => ({ history: t === 'TMF' ? TMF : wavy('2026-10-09') }));
        const r = await resolveSignals({ faults: new Set(['signals_nasdaq', 'signals_spot']), now: NOW, nasdaq, yahoo, cnbc: jest.fn() });
        expect(r.FrontRunner.source).toBe('Computed from Yahoo prices (2026-10-09 close)');
    });
});
