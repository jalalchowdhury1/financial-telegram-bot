/**
 * Tests for jevInputs.js — pure helpers for repairing pill inputs.
 */
import { FRESH, claims4wkFromHistory, sahmFromHistory, resolvePillInput } from '../jevInputs';

jest.mock('../store', () => ({
    loadLastGood: jest.fn(),
    saveLastGood: jest.fn(),
}));
import { loadLastGood, saveLastGood } from '../store';

beforeEach(() => {
    jest.clearAllMocks();
});

// ---------------------------------------------------------------------------
// FRESH
// ---------------------------------------------------------------------------

describe('FRESH', () => {
    test('defines freshness days for all four inputs', () => {
        expect(FRESH.T10Y3M).toBe(7);
        expect(FRESH.NFCI).toBe(14);
        expect(FRESH.ICSA).toBe(14);
        expect(FRESH.UNRATE).toBe(75); // observation-dated monthly: ~65 days old just before the next print
    });
});

// ---------------------------------------------------------------------------
// claims4wkFromHistory
// ---------------------------------------------------------------------------

describe('claims4wkFromHistory', () => {
    function makeClaimsHistory(values) {
        return values.map((v, i) => ({ date: `2026-01-${String(i + 1).padStart(2, '0')}`, value: v }));
    }

    test('averages last 4 weeks ÷ 1000', () => {
        // 4 values: [210000, 220000, 230000, 240000] → sum=900000, avg=225000, ÷1000 = 225
        const history = makeClaimsHistory([210000, 220000, 230000, 240000]);
        expect(claims4wkFromHistory(history)).toBe(225);
    });

    test('returns null when less than 4 points', () => {
        expect(claims4wkFromHistory(makeClaimsHistory([210000, 220000, 230000]))).toBeNull();
        expect(claims4wkFromHistory([])).toBeNull();
        expect(claims4wkFromHistory(null)).toBeNull();
        expect(claims4wkFromHistory(undefined)).toBeNull();
    });

    test('not rounded — preserves decimals', () => {
        const history = makeClaimsHistory([210123, 220456, 230789, 240111]);
        const avg = (210123 + 220456 + 230789 + 240111) / 4000;
        expect(claims4wkFromHistory(history)).toBe(avg);
    });

    test('uses the last 4 when history has more than 4 points', () => {
        const history = makeClaimsHistory([100000, 200000, 210000, 220000, 230000, 240000]);
        // last 4: 210000 + 220000 + 230000 + 240000 = 900000, avg/1000 = 225
        expect(claims4wkFromHistory(history)).toBe(225);
    });
});

// ---------------------------------------------------------------------------
// sahmFromHistory
// ---------------------------------------------------------------------------

describe('sahmFromHistory', () => {
    // Real FRED UNRATE (current vintage) May 2023 → Aug 2024. The official rule
    // (FRED SAHMCURRENT) fired at 0.50 in Jul 2024 and 0.57 in Aug 2024; the old
    // "3-mo average minus the lowest single month" formula read 0.37 / 0.47 (safe).
    const UNRATE_2023_05 = [3.6, 3.6, 3.5, 3.7, 3.7, 3.9, 3.7, 3.8, 3.7, 3.9, 3.9, 3.9, 3.9, 4.1, 4.2, 4.2];
    const monthly = (values, y = 2023, m = 5) => values.map((v, i) => {
        const k = (m - 1) + i;
        return { date: `${y + Math.floor(k / 12)}-${String((k % 12) + 1).padStart(2, '0')}-01`, value: v };
    });

    test('known numbers: Jul 2024 = 0.50 and Aug 2024 = 0.57 (FRED SAHMCURRENT)', () => {
        expect(sahmFromHistory(monthly(UNRATE_2023_05.slice(0, 15)))).toBeCloseTo(0.5, 2);
        expect(sahmFromHistory(monthly(UNRATE_2023_05))).toBe(0.57);
        expect(sahmFromHistory(monthly(UNRATE_2023_05.slice(0, 15)))).toBeGreaterThanOrEqual(0.5 - 1e-9);
    });

    test('compares 3-mo averages, not single months: one low month does not set the floor', () => {
        // A one-month dip to 3.0 lifts the old single-month formula; the 3-mo averages barely move.
        const values = [4.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0];
        expect(sahmFromHistory(monthly(values))).toBe(0.33);       // 4.0 − (4+3+4)/3
    });

    test('goes by calendar month: a month FRED skipped is a gap, not the month before', () => {
        // Oct 2025 style: month 14 missing. By month the latest average is (4.0 + 5.0) / 2; by row it would be (4 + 4 + 5) / 3.
        const values = monthly([...Array(16).fill(4.0)]).map((p, i) => (i === 15 ? { ...p, value: 5.0 } : p)).filter((_, i) => i !== 14);
        expect(sahmFromHistory(values)).toBe(0.5);
        const twoGone = monthly(Array(18).fill(4.0)).filter((_, i) => i !== 15 && i !== 16);
        expect(sahmFromHistory(twoGone)).toBeNull(); // the latest average has 1 of its 3 months
    });

    test('known number: Sahm = 0.0 when unemployment is stable', () => {
        expect(sahmFromHistory(monthly(Array(15).fill(4.0)))).toBeCloseTo(0, 5);
    });

    test('rounds to FRED\'s 2 dp: float noise never shows as -0.00', () => {
        // Raw maths gives -8.88e-16 here, and (-8.88e-16).toFixed(2) is '-0.00' on the card.
        const v = sahmFromHistory(monthly([3.9, 4.4, 4.1, 4.5, 4.4, 4.4, 3.9, 4.1, 4.5, 4.1, 4.5, 4.4, 3.9, 3.9, 4.4]));
        expect(Object.is(v, 0)).toBe(true);
        expect(v.toFixed(2)).toBe('0.00');
    });

    test('uses only the newest 15 months of a long history', () => {
        expect(sahmFromHistory(monthly([2.0, 2.0, 2.0, ...UNRATE_2023_05.slice(0, 15)], 2023, 2))).toBeCloseTo(0.5, 2);
    });

    test('returns null when less than 15 points', () => {
        expect(sahmFromHistory([])).toBeNull();
        expect(sahmFromHistory(monthly(Array(14).fill(4.0)))).toBeNull();
        expect(sahmFromHistory(null)).toBeNull();
    });

    test('returns null for non-array input', () => {
        expect(sahmFromHistory(null)).toBeNull();
        expect(sahmFromHistory(undefined)).toBeNull();
        expect(sahmFromHistory('not-an-array')).toBeNull();
    });
});

// ---------------------------------------------------------------------------
// resolvePillInput
// ---------------------------------------------------------------------------

describe('resolvePillInput', () => {
    // Helper: make a source that resolves to a given history
    const makeSource = (name, freshnessDays, history) => ({
        name,
        freshnessDays,
        fetch: jest.fn().mockResolvedValue(history),
    });

    // Helper: ascending history
    const hist = (values, dates) =>
        values.map((v, i) => ({ date: dates?.[i] || `2026-01-${String(i + 1).padStart(2, '0')}`, value: v }));

    test('first source wins when it returns a fresh history', async () => {
        const sources = [
            makeSource('alpha', 7, hist([10, 20, 30], ['2026-01-01', '2026-01-02', '2026-01-03'])),
            makeSource('beta', 7, hist([40, 50, 60])),
        ];
        // now = 2026-01-05, so the newest point '2026-01-03' is 2 days old — fresh (≤ 7)
        const now = new Date('2026-01-05');
        const result = await resolvePillInput({ sources, now });
        expect(result.value).toBe(30);
        expect(result.source).toBe('alpha');
        expect(result.tried).toContain('alpha:ok');
        // Beta should not have been called
        expect(sources[1].fetch).not.toHaveBeenCalled();
    });

    test('second source is tried when first throws', async () => {
        const sources = [
            { name: 'alpha', freshnessDays: 7, fetch: jest.fn().mockRejectedValue(new Error('down')) },
            makeSource('beta', 7, hist([10, 20, 30])),
        ];
        const now = new Date('2026-01-05');
        const result = await resolvePillInput({ sources, now });
        expect(result.value).toBe(30);
        expect(result.source).toBe('beta');
        expect(result.tried.some((t) => t.startsWith('alpha:err'))).toBe(true);
        expect(result.tried).toContain('beta:ok');
    });

    test('stale source is skipped (newest date past freshnessDays)', async () => {
        const sources = [
            makeSource('alpha', 7, hist([10, 20, 30], ['2025-12-20', '2025-12-21', '2025-12-22'])),
            makeSource('beta', 7, hist([40, 50, 60], ['2026-01-16', '2026-01-17', '2026-01-18'])),
        ];
        // now = 2026-01-20 — alpha newest is 2025-12-22 (29 days ago, stale for 7d deadline)
        // beta newest is 2026-01-18 (2 days ago, fresh)
        const now = new Date('2026-01-20');
        const result = await resolvePillInput({ sources, now });
        expect(result.value).toBe(60);
        expect(result.source).toBe('beta');
        expect(result.tried).toContain('alpha:stale(2025-12-22)');
        expect(result.tried).toContain('beta:ok');
    });

    test('hm_<name> fault skips a source', async () => {
        const sources = [
            makeSource('alpha', 7, hist([10, 20, 30])),
            makeSource('beta', 7, hist([40, 50, 60])),
        ];
        const faults = new Set(['hm_alpha']);
        const now = new Date('2026-01-05');
        const result = await resolvePillInput({ sources, faults, now });
        expect(result.value).toBe(60);
        expect(result.source).toBe('beta');
        expect(result.tried).toContain('alpha:off');
        expect(sources[0].fetch).not.toHaveBeenCalled();
    });

    test('last-good is read when all sources fail', async () => {
        loadLastGood.mockReturnValue({ data: { value: 99, asOf: '2026-01-03', source: 'alpha' }, savedAt: '2026-01-03T12:00:00Z' });
        const sources = [
            { name: 'alpha', freshnessDays: 7, fetch: jest.fn().mockRejectedValue(new Error('down')) },
            { name: 'beta', freshnessDays: 7, fetch: jest.fn().mockRejectedValue(new Error('down too')) },
        ];
        const now = new Date('2026-01-05');
        const result = await resolvePillInput({ sources, now, lastGoodKey: 'jev-test' });
        expect(result.value).toBe(99);
        expect(result.source).toBe('lastgood');
        expect(loadLastGood).toHaveBeenCalledWith('jev-test', 7 * 864e5);
    });

    test('lastgood fault blocks the last-good read', async () => {
        loadLastGood.mockReturnValue({ data: { value: 99, asOf: '2026-01-03', source: 'alpha' }, savedAt: '2026-01-03T12:00:00Z' });
        const sources = [
            { name: 'alpha', freshnessDays: 7, fetch: jest.fn().mockRejectedValue(new Error('down')) },
        ];
        const faults = new Set(['lastgood']);
        const now = new Date('2026-01-05');
        const result = await resolvePillInput({ sources, faults, now, lastGoodKey: 'jev-test' });
        expect(result.value).toBeNull();
        expect(result.source).toBeNull();
        expect(loadLastGood).not.toHaveBeenCalled();
    });

    test('does not write last-good when faults are present', async () => {
        const sources = [
            makeSource('alpha', 7, hist([10, 20, 30])),
        ];
        const faults = new Set(['some-fault']);
        const now = new Date('2026-01-05');
        await resolvePillInput({ sources, faults, now, lastGoodKey: 'jev-test' });
        expect(saveLastGood).not.toHaveBeenCalled();
    });

    test('writes last-good on live success when no faults present', async () => {
        const sources = [
            makeSource('alpha', 7, hist([10, 20, 30])),
        ];
        const now = new Date('2026-01-05');
        await resolvePillInput({ sources, now, lastGoodKey: 'jev-test' });
        expect(saveLastGood).toHaveBeenCalledWith('jev-test', { value: 30, asOf: '2026-01-03', source: 'alpha' });
    });

    test('returns null for value when every source fails and no last-good', async () => {
        loadLastGood.mockReturnValue(null);
        const sources = [
            { name: 'alpha', freshnessDays: 7, fetch: jest.fn().mockRejectedValue(new Error('down')) },
        ];
        const result = await resolvePillInput({ sources, lastGoodKey: 'jev-test' });
        expect(result.value).toBeNull();
        expect(result.source).toBeNull();
        expect(result.tried.some((t) => t.startsWith('alpha:err'))).toBe(true);
        expect(result.tried).toContain('lastgood:none');
    });
});
describe('resolvePillInput derive hook', () => {
    const hist = [
        { date: '2026-09-01', value: 200000 }, { date: '2026-09-08', value: 210000 },
        { date: '2026-09-15', value: 220000 }, { date: '2026-09-18', value: 230000 },
    ];
    test('derive(history) replaces the newest point as the kept value', async () => {
        const r = await resolvePillInput({
            sources: [{ name: 'a', freshnessDays: 14, derive: claims4wkFromHistory, fetch: async () => hist }],
            now: new Date('2026-09-19T00:00:00Z'),
        });
        expect(r.value).toBeCloseTo(215, 6);
        expect(r.asOf).toBe('2026-09-18');
        expect(r.source).toBe('a');
    });
    test('a source whose derive returns null is skipped, next source wins', async () => {
        const r = await resolvePillInput({
            sources: [
                { name: 'a', freshnessDays: 14, derive: () => null, fetch: async () => hist },
                { name: 'b', freshnessDays: 14, fetch: async () => hist },
            ],
            now: new Date('2026-09-19T00:00:00Z'),
        });
        expect(r.source).toBe('b');
        expect(r.tried).toContain('a:underived');
        expect(r.value).toBe(230000);
    });
});
