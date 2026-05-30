import { isStale, withFreshness, formatAsOf, freshnessNote } from '../freshness';

const NOW = new Date('2026-05-30T12:00:00Z');

describe('isStale', () => {
    test('fresh: data within the deadline is not stale', () => {
        expect(isStale('2026-05-28', 5, NOW)).toBe(false);
    });

    test('exactly at the deadline is not stale', () => {
        // 2026-05-25 is exactly 5 days before NOW
        expect(isStale('2026-05-25T12:00:00Z', 5, NOW)).toBe(false);
    });

    test('past the deadline is stale', () => {
        expect(isStale('2026-05-20', 5, NOW)).toBe(true);
    });

    test('quarterly data months old is fine under a generous deadline', () => {
        expect(isStale('2026-03-31', 130, NOW)).toBe(false);
    });

    test('missing or invalid date is treated as stale', () => {
        expect(isStale(null, 5, NOW)).toBe(true);
        expect(isStale(undefined, 50, NOW)).toBe(true);
        expect(isStale('not-a-date', 5, NOW)).toBe(true);
    });
});

describe('withFreshness', () => {
    test('keeps a fresh value', () => {
        expect(withFreshness(1.23, '2026-05-29', 5, NOW)).toEqual({
            value: 1.23, asOf: '2026-05-29', stale: false, unavailable: false,
        });
    });

    test('nulls a stale value and flags stale', () => {
        const r = withFreshness(1.23, '2026-05-01', 5, NOW);
        expect(r.value).toBeNull();
        expect(r.stale).toBe(true);
        expect(r.unavailable).toBe(false);
    });

    test('missing value is unavailable, not stale', () => {
        const r = withFreshness(undefined, null, 5, NOW);
        expect(r.value).toBeNull();
        expect(r.unavailable).toBe(true);
        expect(r.stale).toBe(false);
    });
});

describe('freshnessNote', () => {
    test('fresh value shows an "As of" suffix and no amber', () => {
        const note = freshnessNote({ value: 1.2, asOf: '2026-05-29', stale: false });
        expect(note.amber).toBe(false);
        expect(note.suffix).toContain('As of');
    });

    test('stale value is amber with a "couldn\'t refresh" note', () => {
        const note = freshnessNote({ value: null, asOf: '2026-05-01', stale: true });
        expect(note.amber).toBe(true);
        expect(note.suffix).toContain("couldn't refresh");
    });

    test('unavailable value is amber', () => {
        const note = freshnessNote({ value: null, asOf: null, stale: false });
        expect(note.amber).toBe(true);
    });
});

describe('formatAsOf', () => {
    test('formats a bare date without an off-by-one shift', () => {
        expect(formatAsOf('2026-05-28')).toBe('May 28, 2026');
    });

    test('returns null for missing/invalid input', () => {
        expect(formatAsOf(null)).toBeNull();
        expect(formatAsOf('nope')).toBeNull();
    });
});
