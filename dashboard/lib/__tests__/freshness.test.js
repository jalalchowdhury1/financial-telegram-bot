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

    test('value exactly at the deadline is fresh, one day past is stale (staleDays 1)', () => {
        const atEdge = withFreshness(1, '2026-05-25T12:00:00Z', 5, NOW); // exactly 5 days
        expect(atEdge.stale).toBe(false);
        expect(atEdge.staleDays).toBe(0);
        const justOver = withFreshness(1, '2026-05-24T12:00:00Z', 5, NOW); // 6 days
        expect(justOver.stale).toBe(true);
        expect(justOver.value).toBe(1);
        expect(justOver.staleDays).toBe(1);
    });

    test('a legitimate 0 reading is kept, not treated as unavailable', () => {
        const r = withFreshness(0, '2026-05-29', 5, NOW);
        expect(r.value).toBe(0);
        expect(r.unavailable).toBe(false);
    });

    test('NaN value is unavailable', () => {
        const r = withFreshness(NaN, '2026-05-29', 5, NOW);
        expect(r.value).toBeNull();
        expect(r.unavailable).toBe(true);
    });

    test('stale with missing asOf stays unavailable (no value) and staleDays 0', () => {
        const r = withFreshness(null, undefined, 5, NOW);
        expect(r.unavailable).toBe(true);
        expect(r.stale).toBe(false);
        expect(r.staleDays).toBe(0);
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

    test('M2/Durable/Savings are NOT stale at their worst-case age (~85d, day before next print)', () => {
        const dayBeforeNextPrint = new Date('2026-06-25T12:00:00Z'); // May prints ~Jun 26-27
        for (const id of ['M2SL', 'DGORDER', 'PSAVERT']) {
            expect(isStale('2026-04-01', FRED_FRESHNESS[id], dayBeforeNextPrint)).toBe(false);
        }
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

// ─────────────────────────────────────────────────────────────────────────────
// Deadlines must cover the publisher's REAL cadence, or graceful staleness turns
// into a standing false alarm — the same trap that made UMCSENT/M2SL/DGORDER/
// PSAVERT need 95 instead of 80. Now that these cards are in the health check's
// N/A sweep, a too-tight deadline is not just an orange 🕐, it is a daily warn.
// ─────────────────────────────────────────────────────────────────────────────
describe('FRED_FRESHNESS deadlines vs the real publication cadence', () => {
    test('profitMargin (corporate profits) survives a full BEA quarterly cycle', () => {
        // FRED dates a quarter at its START; BEA publishes corporate profits with the
        // GDP 2nd estimate ~2 months after the quarter ENDS. So Q2 2026 (dated
        // 2026-04-01) prints ~2026-08-27 and stays the newest point until Q3 prints
        // ~2026-11-25 — by which time it is 238 days old and still perfectly current.
        const asOf = '2026-04-01';
        const justBeforeNextPrint = new Date('2026-11-25T12:00:00Z');
        expect(isStale(asOf, FRED_FRESHNESS.A053RC1Q027SBEA, justBeforeNextPrint)).toBe(false);
    });

    test('but a genuinely dead corporate-profits feed still trips it', () => {
        // A full extra quarter with no print (~330 days) must NOT pass.
        expect(isStale('2026-04-01', FRED_FRESHNESS.A053RC1Q027SBEA,
            new Date('2027-02-25T12:00:00Z'))).toBe(true);
    });
});
