import { sliceRange, rangeLabel, getRange, setRange, resetRange, DEFAULT_RANGE } from '../chartRange';

const day = (i) => new Date(Date.parse('2026-10-09T00:00:00Z') - i * 86400000).toISOString().slice(0, 10);
const pts = (n) => Array.from({ length: n }, (_, k) => ({ date: day(n - 1 - k), value: k }));

beforeEach(() => { resetRange(); window.localStorage.clear(); });

test('slices to the window ending at the newest point', () => {
    const p = pts(212);
    expect(sliceRange(p, '1M')).toHaveLength(30);
    expect(sliceRange(p, '3M')).toHaveLength(90);
    expect(sliceRange(p, '6M')).toHaveLength(180);
    expect(sliceRange(p, 'ALL')).toHaveLength(212);
    expect(sliceRange(p, 'nope')).toHaveLength(90); // unknown → the default 3M
    expect(sliceRange(p, '1M').at(-1).date).toBe('2026-10-09');
});

test('the eyebrow never claims more history than there is', () => {
    const p = pts(212);
    expect(rangeLabel(sliceRange(p, '1M'), '1M')).toBe('30 days');
    expect(rangeLabel(sliceRange(p, '3M'), '3M')).toBe('90 days');
    expect(rangeLabel(sliceRange(p, '6M'), '6M')).toBe('180 days');
    expect(rangeLabel(sliceRange(p, 'ALL'), 'ALL')).toBe('since Mar 12');
    const short = pts(100); // a metric added later: 6M can only show 100 days
    expect(rangeLabel(sliceRange(short, '6M'), '6M')).toBe('since Jul 2');
    expect(rangeLabel(sliceRange(pts(400), 'ALL'), 'ALL')).toBe('since Sep 5, 2025'); // year once it is ambiguous
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
