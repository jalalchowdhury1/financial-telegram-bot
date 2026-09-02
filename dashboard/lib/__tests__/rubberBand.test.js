import { validateSnapshot, snapshotAge, DIAL_ORDER, dialLabel } from '../rubberBand';

const good = {
    asOf: '2026-09-01', generatedAt: '2026-09-02T04:34:10',
    spec: { version: '1.0' },
    verdict: { colour: 'green', text: 'The rubber band is working.' },
    dials: {
        slow: { colour: 'green', n: 30, excess_pct: 0.63 },
        fast: { colour: 'green', n: 20, excess_pct: 0.64 },
        age: { colour: 'green', years: 2.8 },
        rip: { colour: 'green', excess_pct: -0.07 },
        machines: { colour: 'green', legs: [] },
    },
    history: [{ d: '2026-09-01', slow: 0.63, fast: 0.64, rip: -0.07 }],
};

describe('validateSnapshot', () => {
    test('accepts a well-formed snapshot', () => {
        expect(validateSnapshot(good)).toBe(true);
    });
    test('rejects garbage, missing dials, or an unknown colour', () => {
        expect(validateSnapshot(null)).toBe(false);
        expect(validateSnapshot({})).toBe(false);
        expect(validateSnapshot({ ...good, dials: { ...good.dials, rip: undefined } })).toBe(false);
        expect(validateSnapshot({ ...good, dials: { ...good.dials, slow: { colour: 'blue' } } })).toBe(false);
        expect(validateSnapshot({ ...good, verdict: null })).toBe(false);
    });
});

describe('snapshotAge', () => {
    // The engine runs after each close; a snapshot dated Friday is fresh all weekend and
    // on Monday, stale once it's more than 4 calendar days old (a missed nightly run).
    test('fresh over a weekend, stale after a missed run', () => {
        expect(snapshotAge('2026-08-28', new Date('2026-08-31T13:00:00Z'))).toEqual({ ageDays: 3, stale: false });
        expect(snapshotAge('2026-08-28', new Date('2026-09-02T13:00:00Z'))).toEqual({ ageDays: 5, stale: true });
        expect(snapshotAge('2026-09-01', new Date('2026-09-02T13:00:00Z'))).toEqual({ ageDays: 1, stale: false });
    });
    test('missing or invalid asOf is stale', () => {
        expect(snapshotAge(null, new Date()).stale).toBe(true);
        expect(snapshotAge('nope', new Date()).stale).toBe(true);
    });
});

test('dial order and labels are fixed', () => {
    expect(DIAL_ORDER).toEqual(['slow', 'fast', 'age', 'rip', 'machines']);
    expect(dialLabel('slow')).toMatch(/dip/i);
    expect(dialLabel('machines')).toMatch(/machine/i);
});
