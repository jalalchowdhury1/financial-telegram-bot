import { fgHistoryCells } from '../fgHistory';

// Shapes /api/fear-greed really serves (app/api/fear-greed/route.js).
const cnn = { score: 31.2, rating: 'fear', previousClose: 28.4, previousWeek: 36.9428571428571, previousMonth: 46.057142857142864, previousYear: 61.6 };

test('CNN answer: every cell is a whole number with its change vs today', () => {
    expect(fgHistoryCells(cnn)).toEqual([
        { label: 'Prev Close', val: 28, diff: 3 },
        { label: '1 Week', val: 37, diff: -6 },
        { label: '1 Month', val: 46, diff: -15 },
        { label: '1 Year', val: 62, diff: -31 },
    ]);
});

test("VIX-proxy tier: previousYear 'N/A' is a blank cell, never NaN", () => {
    const cells = fgHistoryCells({ ...cnn, previousYear: 'N/A' });
    expect(cells[3]).toEqual({ label: '1 Year', val: null, diff: null });
    expect(cells[0]).toEqual({ label: 'Prev Close', val: 28, diff: 3 });
});

test('FRED tier: a null cell is blank, never a fake 0', () => {
    const cells = fgHistoryCells({ ...cnn, previousMonth: null, previousWeek: undefined, previousClose: '' });
    expect(cells.slice(0, 3).map((c) => c.val)).toEqual([null, null, null]);
    expect(cells.slice(0, 3).map((c) => c.diff)).toEqual([null, null, null]);
});

test('a non-numeric score keeps the cells but drops every arrow; missing payload = all blank', () => {
    expect(fgHistoryCells({ ...cnn, score: 'N/A' }).map((c) => [c.val, c.diff])).toEqual([[28, null], [37, null], [46, null], [62, null]]);
    expect(fgHistoryCells(null).map((c) => c.val)).toEqual([null, null, null, null]);
    expect(fgHistoryCells({}).map((c) => c.val)).toEqual([null, null, null, null]);
});

test('numeric strings still count; junk strings do not', () => {
    expect(fgHistoryCells({ ...cnn, previousClose: '28.4', previousWeek: 'abc' }).slice(0, 2).map((c) => c.val)).toEqual([28, null]);
});
