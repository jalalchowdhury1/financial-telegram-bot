import {
    valueAt, changeOver, preRecessionRunups, runupMedian, horsemanStatus, lastInversion, latestYoY, yearAgoGap,
} from '../horsemenRunup';
import live from './fixtures/fred-horsemen-2026-10-04.json';

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

// Real /api/fred slices saved 2026-10-04 (see the fixture's _note).
describe('latestYoY — latest print vs the print one calendar year before it', () => {
    const H = live.horsemen;
    test('unemployment: Sep-26 vs Sep-25 = -0.2pp, even though Oct-25 never printed', () => {
        // The old header counted 13 entries back and, with Oct-25 missing, landed on Aug-25 (-0.1pp).
        expect(latestYoY(H.unemployment.history, 'pp')).toBeCloseTo(-0.2, 6);
    });
    test('bankruptcies: a months-old quarterly print compares with its own quarter a year earlier (+16.9%)', () => {
        // Anchoring at TODAY compared Jun-26 with Sep-25 (+12%) while the header said +16.9%.
        expect(latestYoY(H.bankruptcies.history, 'pct')).toBeCloseTo(16.916, 2);
    });
    test('weekly claims: the same week a year earlier (52 weeks back), not 53', () => {
        // 2026-09-26 vs 2025-09-27 (225K), one day off the calendar date; 2025-09-20 is six days off.
        expect(latestYoY(H.claims.history, 'pct')).toBeCloseTo(100 * (197000 / 225000 - 1), 6);
    });
    test('no print near the year-ago date -> null, never a 13-month change labelled "1y"', () => {
        const oct = [...H.unemployment.history, { date: '2026-10-01', value: 4.3 }];
        expect(latestYoY(oct, 'pp')).toBeNull();
    });
    test('skips null prints and handles missing or short histories', () => {
        expect(latestYoY(null, 'pct')).toBeNull();
        expect(latestYoY([], 'pct')).toBeNull();
        expect(latestYoY([{ date: '2026-01-01', value: 1 }], 'pct')).toBeNull();
        expect(latestYoY([{ date: '2025-01-01', value: 0 }, { date: '2026-01-01', value: 1 }], 'pct')).toBeNull();
        const withNull = [{ date: '2025-01-01', value: 4 }, { date: '2026-01-01', value: 4.5 }, { date: '2026-02-01', value: null }];
        expect(latestYoY(withNull, 'pp')).toBeCloseTo(0.5, 6);
    });
});

describe('lastInversion — real 2022-2024 curve with the September 2024 one-day dips', () => {
    test('a one-day dip after a short positive gap belongs to the same inversion', () => {
        const r = lastInversion(live.yieldCurve.history, D('2026-10-04'));
        expect(r.startYear).toBe(2022);
        expect(r.endYear).toBe(2024);
        expect(r.start).toBe('2022-07-06');
        expect(r.end).toBe('2024-09-05');
        expect(r.monthsSince).toBe(25);
        expect(r.currentlyInverted).toBe(false);
    });
    test('a positive gap of ~90 days or more starts a separate inversion', () => {
        const s = [
            { date: '2019-08-27', value: -0.04 }, { date: '2019-08-29', value: -0.01 },
            { date: '2019-09-03', value: 0.02 }, { date: '2022-04-01', value: 0.1 },
            { date: '2022-07-06', value: -0.02 }, { date: '2023-07-03', value: -1.05 },
            { date: '2024-08-26', value: -0.01 }, { date: '2024-08-27', value: 0.02 },
        ];
        const r = lastInversion(s, D('2026-10-04'));
        expect(r.start).toBe('2022-07-06');
        expect(r.end).toBe('2024-08-26');
    });
    test('a curve inverted right now is live, even straight after a short positive gap', () => {
        const s = [
            { date: '2026-05-01', value: -0.1 }, { date: '2026-06-01', value: 0.05 }, { date: '2026-07-01', value: -0.2 },
        ];
        const r = lastInversion(s, D('2026-07-02'));
        expect(r.currentlyInverted).toBe(true);
        expect(r.start).toBe('2026-05-01');
        expect(r.monthsSince).toBe(0);
    });
});

// Review: UNRATE never printed Oct 2025. When Oct 2026 lands (early Nov), latestYoY is null
// for a month; the rail must say WHY instead of "not enough history" (it has 50+ years).
describe('yearAgoGap — names the missing year-ago print', () => {
    const H = live.horsemen;
    test('UNRATE with an Oct-2026 print: the Oct-2025 print is missing', () => {
        const oct = [...H.unemployment.history, { date: '2026-10-01', value: 4.3 }];
        expect(yearAgoGap(oct)).toBe('Oct 2025');
    });
    test('a year-ago print exists: no gap (today\'s live series)', () => {
        expect(yearAgoGap(H.unemployment.history)).toBeNull();
        expect(yearAgoGap(H.claims.history)).toBeNull();
        expect(yearAgoGap(H.bankruptcies.history)).toBeNull();
    });
    test('weekly series: the gap is named by its day, not a whole month', () => {
        const wk = (d, v) => ({ date: d, value: v });
        const h = [wk('2025-09-06', 1), wk('2025-09-13', 1), wk('2025-10-04', 1), wk('2026-09-19', 2), wk('2026-09-26', 2)];
        expect(yearAgoGap(h)).toBe('Sep 26, 2025');
    });
    test('a series that starts after the year-ago date is short, not gapped', () => {
        expect(yearAgoGap([{ date: '2026-03-01', value: 1 }, { date: '2026-09-01', value: 2 }])).toBeNull();
    });
    test('missing or tiny histories -> null', () => {
        expect(yearAgoGap(null)).toBeNull();
        expect(yearAgoGap([])).toBeNull();
        expect(yearAgoGap([{ date: '2026-01-01', value: 1 }])).toBeNull();
    });
});
