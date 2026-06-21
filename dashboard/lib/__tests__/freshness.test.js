import { isStale, withFreshness, formatAsOf, freshnessNote } from '../freshness';
import { FRED_FRESHNESS } from '../constants';

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
            value: 1.23, asOf: '2026-05-29', stale: false, unavailable: false, staleDays: 0,
        });
    });

    test('KEEPS a stale value (no longer nulled) and flags stale + staleDays', () => {
        const r = withFreshness(1.23, '2026-05-01', 5, NOW); // 29 days old, deadline 5
        expect(r.value).toBe(1.23);          // value is kept, not nulled
        expect(r.stale).toBe(true);
        expect(r.unavailable).toBe(false);
        expect(r.staleDays).toBe(24);        // floor(29 - 5)
    });

    test('fresh value has staleDays 0', () => {
        expect(withFreshness(1.23, '2026-05-29', 5, NOW).staleDays).toBe(0);
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

    test('stale value is orange-toned with a clock note and keeps showing', () => {
        const note = freshnessNote({ value: 1.23, asOf: '2026-05-01', stale: true });
        expect(note.tone).toBe('stale');
        expect(note.amber).toBe(true);          // back-compat getter
        expect(note.suffix).toContain('🕐');
        expect(note.suffix).toContain('stale');
    });

    test('truly unavailable value is unavailable-toned', () => {
        const note = freshnessNote({ value: null, asOf: null, stale: false, unavailable: true });
        expect(note.tone).toBe('unavailable');
        expect(note.suffix).toContain('Unavailable');
    });

    test('unavailable value is amber', () => {
        const note = freshnessNote({ value: null, asOf: null, stale: false });
        expect(note.amber).toBe(true);
    });
});

// Regression guard: real FRED reporting lag must NOT be flagged stale, but a
// genuinely dead feed must be. (On 2026-05-30, monthly data is dated 2026-04-01
// or earlier; JOLTS ~2026-03-01; quarterly ~2026-01-01.)
describe('FRED_FRESHNESS vs real reporting lag', () => {
    test('monthly series dated Apr 1 (59 days) is NOT stale', () => {
        for (const id of ['UMCSENT', 'M2SL', 'RSXFS', 'HOUST', 'INDPRO', 'DGORDER', 'PSAVERT', 'UNRATE']) {
            expect(isStale('2026-04-01', FRED_FRESHNESS[id], NOW)).toBe(false);
        }
    });

    // UMCSENT's free FRED series is delayed ONE MONTH at the source's request: the
    // month-M reading lands on FRED ~the 26th of month M+2. So the latest point can
    // legitimately age to ~86 days right before the next print (e.g. the Apr 1 point
    // is newest until the May reading publishes ~Jun 26). Its deadline must cover
    // that worst case, or it false-alarms N/A for ~a week every month.
    test('UMCSENT dated Apr 1 is NOT stale the day before the next print (~85 days)', () => {
        const dayBeforeNextPrint = new Date('2026-06-25T12:00:00Z'); // May reading prints ~Jun 26
        expect(isStale('2026-04-01', FRED_FRESHNESS.UMCSENT, dayBeforeNextPrint)).toBe(false);
    });

    test('JOLTS dated Mar 1 (90 days) is NOT stale', () => {
        expect(isStale('2026-03-01', FRED_FRESHNESS.JTSJOL, NOW)).toBe(false);
    });

    test('quarterly dated Jan 1 (149 days) is NOT stale', () => {
        expect(isStale('2026-01-01', FRED_FRESHNESS.GDP, NOW)).toBe(false);
        expect(isStale('2026-01-01', FRED_FRESHNESS.A053RC1Q027SBEA, NOW)).toBe(false);
    });

    test('daily series 2 days old is NOT stale', () => {
        expect(isStale('2026-05-28', FRED_FRESHNESS.DFII10, NOW)).toBe(false);
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
