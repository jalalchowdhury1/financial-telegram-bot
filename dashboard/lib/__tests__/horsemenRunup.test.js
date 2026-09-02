import {
    valueAt, changeOver, preRecessionRunups, runupMedian, horsemanStatus, lastInversion,
} from '../horsemenRunup';

const D = (s) => new Date(`${s}T00:00:00Z`).getTime();
// Monthly series: 4.0 for 2019, stepping up through 2020.
const monthly = [
    { date: '2019-01-01', value: 4.0 }, { date: '2019-07-01', value: 4.0 },
    { date: '2020-01-01', value: 4.5 }, { date: '2020-07-01', value: 5.5 },
];

describe('valueAt', () => {
    test('takes the last observation on or before the date, never a future one', () => {
        expect(valueAt(monthly, D('2019-12-31'))).toBe(4.0);
        expect(valueAt(monthly, D('2020-01-01'))).toBe(4.5);
    });
    test('returns null before the series starts', () => {
        expect(valueAt(monthly, D('2018-01-01'))).toBeNull();
    });
});

describe('changeOver', () => {
    test('pp mode subtracts the value a year earlier', () => {
        expect(changeOver(monthly, D('2020-01-01'), 'pp')).toBeCloseTo(0.5, 6);
    });
    test('pct mode is a percentage change', () => {
        const s = [{ date: '2019-01-01', value: 200 }, { date: '2020-01-01', value: 250 }];
        expect(changeOver(s, D('2020-01-01'), 'pct')).toBeCloseTo(25, 6);
    });
    test('is null when the series does not reach back a year', () => {
        expect(changeOver(monthly, D('2019-03-01'), 'pp')).toBeNull();
    });
});

describe('preRecessionRunups', () => {
    const recessions = [
        { start: '1960-04-01', end: '1961-02-01' },   // before the series -> skipped
        { start: '2020-01-01', end: '2020-04-01' },
    ];
    test('measures the 12-month change at each recession start, skipping ones without data', () => {
        const r = preRecessionRunups(monthly, recessions, 'pp');
        expect(r).toHaveLength(1);
        expect(r[0].start).toBe('2020-01-01');
        expect(r[0].change).toBeCloseTo(0.5, 6);
    });
    test('median of the run-ups', () => {
        expect(runupMedian([{ change: 1 }, { change: 3 }, { change: 2 }])).toBe(2);
        expect(runupMedian([])).toBeNull();
    });
});

describe('horsemanStatus', () => {
    // For claims/unemployment/bankruptcies a RISE is bad, so worseIsUp = true.
    test('improving when the change runs the good way', () => {
        expect(horsemanStatus(-21.6, 11.2, true)).toBe('improving');
    });
    test('watch when worsening but short of the typical pre-recession move', () => {
        expect(horsemanStatus(16.9, 23.1, true)).toBe('watch');
    });
    test('recession-like once it reaches the typical run-up', () => {
        expect(horsemanStatus(25, 23.1, true)).toBe('recession-like');
    });
    test('handles series where a FALL is the bad direction', () => {
        expect(horsemanStatus(-2.5, -2.0, false)).toBe('recession-like');
        expect(horsemanStatus(0.4, -2.0, false)).toBe('improving');
    });
    test('unknown without a change or a median', () => {
        expect(horsemanStatus(null, 11.2, true)).toBe('unknown');
        expect(horsemanStatus(5, null, true)).toBe('unknown');
    });
});

describe('lastInversion', () => {
    const spread = [
        { date: '2021-01-01', value: 1.0 },
        { date: '2022-07-01', value: -0.2 },
        { date: '2023-07-01', value: -0.8 },
        { date: '2024-09-01', value: 0.1 },
        { date: '2026-09-01', value: 0.4 },
    ];
    test('reports when the curve was last inverted and how long ago it ended', () => {
        const r = lastInversion(spread, D('2026-09-01'));
        expect(r.startYear).toBe(2022);
        expect(r.endYear).toBe(2023);
        expect(r.monthsSince).toBeGreaterThan(20);
        expect(r.currentlyInverted).toBe(false);
    });
    test('flags a live inversion', () => {
        const r = lastInversion([{ date: '2026-01-01', value: -0.3 }], D('2026-09-01'));
        expect(r.currentlyInverted).toBe(true);
    });
    test('null when it never inverted', () => {
        expect(lastInversion([{ date: '2026-01-01', value: 0.3 }], D('2026-09-01'))).toBeNull();
    });
});
