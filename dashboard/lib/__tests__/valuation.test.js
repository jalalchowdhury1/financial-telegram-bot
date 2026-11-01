import { fwdDaily, extremeSince, timeAvg, decompose, underAvg, attention } from '../valuation';

const day = (i) => new Date(Date.UTC(2026, 0, 1 + i)).toISOString().slice(0, 10);
// a close every day: 100, 101, 102 …
const spy = Array.from({ length: 30 }, (_, i) => ({ date: day(i), price: 100 + i }));

describe('fwdDaily', () => {
    const rows = [{ date: day(5), pe: 20 }, { date: day(15), pe: 10 }];
    const out = fwdDaily(rows, spy);
    it("starts on the close before the first report and equals FactSet's number on each anchor day", () => {
        expect(out[0].date).toBe(day(4));
        expect(out[0].pe).toBeCloseTo(20);
        expect(out.find((p) => p.date === day(14)).pe).toBeCloseTo(10);
    });
    it('moves forward earnings in a straight line between reports, then holds it', () => {
        const k4 = 104 / 20, k14 = 114 / 10;
        expect(out.find((p) => p.date === day(9)).eps).toBeCloseTo((k4 + k14) / 2);
        const last = out[out.length - 1];
        expect(last.eps).toBeCloseTo(k14);
        expect(last.pe).toBeCloseTo(129 / k14); // keeps moving with SPY after the newest report
    });
    it("carries each report's own 5- and 10-yr averages until the next one", () => {
        const o = fwdDaily([{ date: day(5), pe: 20, a5: null, a10: 15 }, { date: day(15), pe: 10, a5: 17, a10: 16 }], spy);
        expect([o[0], o.find((p) => p.date === day(13)), o[o.length - 1]].map((p) => [p.a5, p.a10])).toEqual([[null, 15], [null, 15], [17, 16]]);
    });
    it('skips a report with no close in the 5 days before it, and an empty input', () => {
        expect(fwdDaily([{ date: '2025-01-01', pe: 20 }], spy)).toEqual([]);
        expect(fwdDaily([], spy)).toEqual([]);
    });
});

describe('extremeSince', () => {
    const pts = (vals) => vals.map((value, i) => ({ date: day(i * 30), value }));
    it('lowest since the last lower reading', () => {
        expect(extremeSince(pts([15, 20, 22, 21, 19, 18, 16]))).toEqual({ low: true, since: day(0) });
    });
    it('a record low or high says since null', () => {
        expect(extremeSince(pts([20, 22, 21, 19, 14]))).toEqual({ low: true, since: null });
        expect(extremeSince(pts([20, 18, 21, 19, 25]))).toEqual({ low: false, since: null });
    });
    it('a flat series is no extreme, a flat tail after a fall still is (audit C10)', () => {
        expect(extremeSince(Array.from({ length: 400 }, (_, i) => ({ date: day(i), value: 18 })))).toBeNull();
        expect(extremeSince(pts([20, 22, 19, 19, 19, 19]))).toEqual({ low: true, since: null });
    });
    it('null when the run is under 90 days', () => {
        expect(extremeSince(pts([20, 25, 22, 23]))).toBeNull();
    });
});

it('timeAvg weights each reading by how long it held', () => {
    const p = [{ date: day(0), value: 10 }, { date: day(30), value: 20 }, { date: day(40), value: 99 }];
    expect(timeAvg(p, day(0))).toBeCloseTo((10 * 30 + 20 * 10) / 40);
    expect(timeAvg(p, day(20))).toBeCloseTo((10 * 10 + 20 * 10) / 20);
});

it('decompose: (1 + price) = (1 + earnings)(1 + P/E)', () => {
    const d = decompose({ price: 100, eps: 5, pe: 20 }, { price: 99, eps: 6, pe: 16.5 });
    expect(d.eps).toBeCloseTo(0.2);
    expect(d.pe).toBeCloseTo(-0.175);
    expect((1 + d.eps) * (1 + d.pe)).toBeCloseTo(1 + d.price);
});

it('underAvg: turns under the 5-yr average, rows without one left out', () => {
    expect(underAvg([{ date: day(0), pe: 18.7, a5: 19.8 }, { date: day(7), pe: 13.1, a5: null }]))
        .toEqual([{ date: day(0), value: 1.1, pe: 18.7, a5: 19.8 }]);
});

describe('attention: the view that needs a look', () => {
    // 600 days: SPY and forward earnings climb (new highs are routine), forward P/E waves between 17 and 19
    const run = (last = {}) => {
        const rows = Array.from({ length: 600 }, (_, i) => ({ date: day(i), price: 100 + i, eps: 5 + i / 100, pe: 18 + Math.sin(i / 20) }));
        Object.assign(rows[599], last);
        return rows;
    };
    it('nothing unusual: null, so the card opens on Price', () => {
        expect(attention(run(), run())).toBeNull();
    });
    it('forward P/E at a 1-year+ low or high', () => {
        expect(attention(run({ pe: 15 }), run())).toEqual({ lens: 'Fwd P/E', low: true, since: null });
        expect(attention(run({ pe: 21 }), run())).toEqual({ lens: 'Fwd P/E', low: false, since: null });
    });
    it('earnings and SPY only on a low', () => {
        expect(attention(run({ eps: 1 }), run())).toEqual({ lens: 'Earnings', low: true, since: null });
        expect(attention(run(), run({ price: 50 }))).toEqual({ lens: 'Price', low: true, since: null });
    });
    it('the rarest wins', () => {
        const fwd = run({ pe: 16.5, eps: 1 });
        fwd[100].pe = 16; // both are 1-year+ lows: the P/E was lower 499 days back, earnings never were
        expect(attention(fwd, run())).toEqual({ lens: 'Earnings', low: true, since: null });
        fwd[100].pe = 18;
        fwd[50].eps = 0.5; // now the P/E is the record and earnings were lower 549 days back
        expect(attention(fwd, run())).toEqual({ lens: 'Fwd P/E', low: true, since: null });
    });
});
