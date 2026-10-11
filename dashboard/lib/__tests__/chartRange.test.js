import { sliceRange, rangeLabel, getRange, setRange, resetRange, DEFAULT_RANGE, rangesFor, pickFor } from '../chartRange';

const day = (i) => new Date(Date.parse('2026-10-09T00:00:00Z') - i * 86400000).toISOString().slice(0, 10);
const pts = (n) => Array.from({ length: n }, (_, k) => ({ date: day(n - 1 - k), value: k }));

beforeEach(() => { resetRange(); window.localStorage.clear(); });

test('slices to the window ending at the newest point', () => {
    const p = pts(212);
    expect(sliceRange(p, '1M')).toHaveLength(30);
    expect(sliceRange(p, '3M')).toHaveLength(90);
    expect(sliceRange(p, '6M')).toHaveLength(180);
    expect(sliceRange(p, 'MAX')).toHaveLength(212);
    expect(sliceRange(p, 'nope')).toHaveLength(90); // unknown → the default 3M
    expect(sliceRange(p, '1M').at(-1).date).toBe('2026-10-09');
});

test('the eyebrow never claims more history than there is', () => {
    const p = pts(212);
    expect(rangeLabel(sliceRange(p, '1M'), '1M')).toBe('30 days');
    expect(rangeLabel(sliceRange(p, '3M'), '3M')).toBe('90 days');
    expect(rangeLabel(sliceRange(p, '6M'), '6M')).toBe('180 days');
    expect(rangeLabel(sliceRange(p, 'MAX'), 'MAX')).toBe('since Mar 12');
    const short = pts(100); // a metric added later: 6M can only show 100 days
    expect(rangeLabel(sliceRange(short, '6M'), '6M')).toBe('since Jul 2');
    expect(rangeLabel(sliceRange(pts(400), 'MAX'), 'MAX')).toBe('since Sep 5, 2025'); // year once it is ambiguous
});

test('the pick defaults to 3M and is remembered across popovers and visits', () => {
    expect(getRange()).toBe(DEFAULT_RANGE);
    setRange('6M');
    expect(getRange()).toBe('6M');
    resetRange(); // a fresh page load
    expect(getRange()).toBe('6M');
    setRange('bogus');
    expect(getRange()).toBe('6M');
});

test('1Y and 5Y: a full window reads "1 year" / "5 years", even when old history is monthly', () => {
    expect(rangeLabel(sliceRange(pts(800), '1Y'), '1Y')).toBe('1 year');
    // monthly points: the 5Y window's first point can sit up to a month after the window opens
    const monthly = Array.from({ length: 120 }, (_, k) => ({ date: day((119 - k) * 30), value: k }));
    expect(rangeLabel(sliceRange(monthly, '5Y'), '5Y')).toBe('5 years');
    expect(rangeLabel(sliceRange(pts(900), '5Y'), '5Y')).toBe('since Apr 23, 2024'); // only ~2.5 years exist
    // monthly baked points + today's sheet point (savings): 1Y's first point lands 9 days in
    const savings = [...Array.from({ length: 24 }, (_, k) => ({ date: day(25 + (23 - k) * 30), value: k })), { date: day(0), value: 9 }];
    expect(rangeLabel(sliceRange(savings, '1Y'), '1Y')).toBe('1 year');
});

test('chips: only the windows shorter than the history, then MAX; a missing pick falls back to MAX', () => {
    const ids = (span) => rangesFor(span).map((r) => r.id);
    expect(ids(13000)).toEqual(['1M', '3M', '6M', '1Y', '5Y', 'MAX']);
    expect(ids(1100)).toEqual(['1M', '3M', '6M', '1Y', 'MAX']);
    expect(ids(212)).toEqual(['1M', '3M', '6M', 'MAX']);
    expect(ids(20)).toEqual(['MAX']);
    expect(pickFor('3M', rangesFor(212))).toBe('3M');
    expect(pickFor('5Y', rangesFor(212))).toBe('MAX');
});

test('the old ALL pick migrates to MAX', () => {
    window.localStorage.setItem('ftb:chartRange', 'ALL');
    expect(getRange()).toBe('MAX');
});
