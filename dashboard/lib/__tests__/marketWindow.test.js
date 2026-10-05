import { marketWindow, spotLabel } from '../marketWindow';

// Tails copied from live /api/market-extra on 2026-10-04 (history points are {date, price}).
const h = (...pairs) => pairs.map(([date, price]) => ({ date, price }));
const row = (history, extra = {}) => ({
    current: history.length ? history[history.length - 1].price : undefined,
    history,
    ...extra,
});

describe('marketWindow: the four real cadences', () => {
    test('TNX: one session, in basis points, tagged with its weekday', () => {
        const r = row(h(['2026-09-29', 5.26], ['2026-09-30', 5.29], ['2026-10-01', 5.24]));
        expect(marketWindow(r, { rate: true, ticker: 'TNX' })).toEqual({ text: '−5bp', tag: 'Thu', dir: -1 });
    });

    test('T2Y: −10bp on Thursday', () => {
        const r = row(h(['2026-09-29', 4.89], ['2026-09-30', 4.88], ['2026-10-01', 4.78]));
        expect(marketWindow(r, { rate: true, ticker: 'T2Y' })).toEqual({ text: '−10bp', tag: 'Thu', dir: -1 });
    });

    test('MORT30: a weekly print is +25bp over 1w, not +3.56%', () => {
        const r = row(h(['2026-09-17', 6.95], ['2026-09-24', 7.03], ['2026-10-01', 7.28]));
        expect(marketWindow(r, { rate: true, ticker: 'MORT30' })).toEqual({ text: '+25bp', tag: '1w', dir: 1 });
    });

    test('MTGPMT: weekly, price in %', () => {
        const r = row(h(['2026-09-17', 2174.9], ['2026-09-24', 2192.54], ['2026-10-01', 2248.05]));
        expect(marketWindow(r, { ticker: 'MTGPMT' })).toEqual({ text: '+2.53%', tag: '1w', dir: 1 });
    });

    test('ZRI: monthly', () => {
        const r = row(h(['2026-06-01', 1971.02745], ['2026-07-01', 1975.5168300000003], ['2026-08-01', 1980.0767700000001]));
        expect(marketWindow(r, { ticker: 'ZRI' })).toEqual({ text: '+0.23%', tag: '1mo', dir: 1 });
    });

    test('ATNHPI: quarterly', () => {
        const r = row(h(['2025-10-01', 439.88], ['2026-01-01', 444.32], ['2026-04-01', 452.64]));
        expect(marketWindow(r, { ticker: 'ATNHPI' })).toEqual({ text: '+1.87%', tag: '3mo', dir: 1 });
    });
});

describe('marketWindow: weekend and holiday tails', () => {
    test('CL: a real Sunday futures bar (value moved) is kept and tagged Sun', () => {
        const r = row(h(['2026-10-01', 92.87], ['2026-10-02', 91.11], ['2026-10-04', 91.03]));
        expect(marketWindow(r, { ticker: 'CL' })).toEqual({ text: '−0.09%', tag: 'Sun', dir: -1 });
    });

    test('CL: a Sunday bar that only repeats Friday is skipped, so the change is Friday\'s real move', () => {
        const r = row(h(['2026-10-01', 92.87], ['2026-10-02', 91.11], ['2026-10-04', 91.11]),
            { dailyChange: { value: 0, pct: 0 } });
        expect(marketWindow(r, { ticker: 'CL' })).toEqual({ text: '−1.90%', tag: 'Fri', dir: -1 });
    });

    test('GOLD: Friday\'s session', () => {
        const r = row(h(['2026-09-30', 4155.820000000001], ['2026-10-01', 4181.76], ['2026-10-02', 4139.49]));
        expect(marketWindow(r, { ticker: 'GOLD' })).toEqual({ text: '−1.01%', tag: 'Fri', dir: -1 });
    });

    test('GOLD: a Saturday repeat of Friday is skipped', () => {
        const r = row(h(['2026-10-01', 4181.76], ['2026-10-02', 4139.49], ['2026-10-03', 4139.49]));
        expect(marketWindow(r, { ticker: 'GOLD' })).toEqual({ text: '−1.01%', tag: 'Fri', dir: -1 });
    });

    test('BTC: trades every day, so a Saturday bar is a real session', () => {
        const r = row(h(['2026-10-01', 84848.73], ['2026-10-02', 84504.88], ['2026-10-03', 84742.22]));
        expect(marketWindow(r, { ticker: 'BTC' })).toEqual({ text: '+0.28%', tag: 'Sat', dir: 1 });
    });

    test('BTC: an unchanged Saturday is NOT skipped (exempt) and prints a real zero', () => {
        const r = row(h(['2026-10-02', 84504.88], ['2026-10-03', 84504.88]));
        expect(marketWindow(r, { ticker: 'BTC' })).toEqual({ text: '0.00%', tag: 'Sat', dir: 0 });
    });

    test('an unchanged weekday session is a real zero, not skipped', () => {
        const r = row(h(['2026-10-01', 4181.76], ['2026-10-02', 4181.76]));
        expect(marketWindow(r, { ticker: 'GOLD' })).toEqual({ text: '0.00%', tag: 'Fri', dir: 0 });
    });

    test('a Thanksgiving repeat bar is skipped (NYSE holiday)', () => {
        const r = row(h(['2026-11-24', 80], ['2026-11-25', 82], ['2026-11-26', 82]));
        expect(marketWindow(r, { ticker: 'CL' })).toEqual({ text: '+2.50%', tag: 'Wed', dir: 1 });
    });

    test('weekly / monthly series are never trimmed, even on a holiday or weekend date', () => {
        // MORT30 unchanged on Thanksgiving (a Thursday): a real 1-week zero, not last week's move.
        const w = row(h(['2026-11-12', 6.9], ['2026-11-19', 7.0], ['2026-11-26', 7.0]));
        expect(marketWindow(w, { rate: true, ticker: 'MORT30' })).toEqual({ text: '0bp', tag: '1w', dir: 0 });
        // ZRI's Aug 1 is a Saturday; a flat month stays a flat month.
        const m = row(h(['2026-06-01', 1971.02], ['2026-07-01', 1975.51], ['2026-08-01', 1975.51]));
        expect(marketWindow(m, { ticker: 'ZRI' })).toEqual({ text: '0.00%', tag: '1mo', dir: 0 });
    });

    test('a long weekend (Fri to Tue over Labor Day, 4 days) is still one session', () => {
        const r = row(h(['2026-09-04', 5.00], ['2026-09-08', 5.03]));
        expect(marketWindow(r, { rate: true, ticker: 'TNX' })).toEqual({ text: '+3bp', tag: 'Tue', dir: 1 });
    });

    test('a 5-day gap matches no cadence: the number stays, the tag goes', () => {
        const r = row(h(['2026-09-03', 5.00], ['2026-09-08', 5.03]));
        expect(marketWindow(r, { rate: true, ticker: 'TNX' })).toEqual({ text: '+3bp', tag: null, dir: 1 });
    });
});

describe('marketWindow: missing or odd data shows nothing fake', () => {
    test.each([
        ['no history', { current: 122.88, dailyChange: { value: 0, pct: 0 }, history: [] }],
        ['history missing', { current: 1.27 }],
        ['one point', row(h(['2026-10-02', 1.42]))],
        ['null payload', null],
        ['empty object', {}],
    ])('%s → null', (_, d) => {
        expect(marketWindow(d, { ticker: 'USD/BDT' })).toBeNull();
    });

    test('N/A and null prices are dropped before the last two are picked', () => {
        const r = { current: 5.24, history: [
            { date: '2026-09-30', price: 5.29 }, { date: '2026-10-01', price: 5.24 }, { date: '2026-10-02', price: 'N/A' }, { date: '2026-10-02', price: null },
        ] };
        expect(marketWindow(r, { rate: true, ticker: 'TNX' })).toEqual({ text: '−5bp', tag: 'Thu', dir: -1 });
    });

    test('a zero previous price gives no % (never Infinity)', () => {
        const r = row(h(['2026-10-01', 0], ['2026-10-02', 5]));
        expect(marketWindow(r, { ticker: 'GOLD' })).toBeNull();
    });

    test('an irregular gap keeps the number and drops the tag', () => {
        const r = row(h(['2026-09-01', 100], ['2026-09-16', 101]));
        expect(marketWindow(r, { ticker: 'ZRI' })).toEqual({ text: '+1.00%', tag: null, dir: 1 });
    });

    test('when the shown value is not on the history, the source\'s own change is used, untagged', () => {
        const hist = h(['2026-10-01', 100], ['2026-10-02', 101]);
        expect(marketWindow({ current: 103, dailyChange: { value: 2, pct: 1.98 }, history: hist }, { ticker: 'GOLD' }))
            .toEqual({ text: '+1.98%', tag: null, dir: 1 });
        expect(marketWindow({ current: 5.30, dailyChange: { value: 0.06, pct: 1.1 }, history: h(['2026-10-01', 5.2], ['2026-10-02', 5.24]) }, { rate: true, ticker: 'TNX' }))
            .toEqual({ text: '+6bp', tag: null, dir: 1 });
        // ...and with no usable dailyChange, nothing.
        expect(marketWindow({ current: 103, history: hist }, { ticker: 'GOLD' })).toBeNull();
    });

    test('a rounding-level gap between current and the last bar still counts as on the history', () => {
        const r = { current: 1.4248, history: h(['2026-10-01', 1.42204], ['2026-10-02', 1.424800001]) };
        expect(marketWindow(r, { ticker: 'USD/CAD' })).toEqual({ text: '+0.19%', tag: 'Fri', dir: 1 });
    });

    test('a move that rounds to zero carries no sign', () => {
        const r = row(h(['2026-09-30', 5.29], ['2026-10-01', 5.2901]));
        expect(marketWindow(r, { rate: true, ticker: 'TNX' })).toEqual({ text: '0bp', tag: 'Thu', dir: 0 });
    });
});

describe('spotLabel: what a row with no history says instead', () => {
    test.each(['USD/BDT', 'INR/BDT', 'CAD/INR', 'CAD/BDT', 'USD/CAD', 'USD/INR', 'DXY'])('%s → daily rate', (t) => {
        expect(spotLabel(t)).toBe('daily rate');
    });
    test.each(['GOLD', 'BTC'])('%s spot fallback → live', (t) => {
        expect(spotLabel(t)).toBe('live');
    });
    test('anything else → nothing', () => {
        expect(spotLabel('TNX')).toBeNull();
        expect(spotLabel(undefined)).toBeNull();
    });
});
