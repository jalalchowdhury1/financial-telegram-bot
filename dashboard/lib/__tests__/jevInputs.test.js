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
        expect(FRESH.UNRATE).toBe(45);
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
    function makeUnrateHistory(values) {
        return values.map((v, i) => ({ date: `202${Math.floor(i / 12)}-${String((i % 12) + 1).padStart(2, '0')}-01`, value: v }));
    }

    test('computes sahm from 12+ months of UNRATE data', () => {
        // Last 3 months: 4.0, 4.1, 4.2 → mean = 4.1
        // Min of all 12: 3.5 (first value)
        // Sahm = 4.1 - 3.5 = 0.6
        const values = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 3.9, 3.8, 3.9, 4.0, 4.1, 4.2];
        expect(sahmFromHistory(makeUnrateHistory(values))).toBeCloseTo(0.6, 5);
    });

    test('known number: Sahm = 0.0 when unemployment is stable', () => {
        const values = Array(14).fill(4.0);
        expect(sahmFromHistory(values.map((v, i) => ({ date: `2026-${String(i + 1).padStart(2, '0')}-01`, value: v })))).toBeCloseTo(0, 5);
    });

    test('known number: Sahm = 0.2 when last 3 months avg is 0.2 above the 12-month low', () => {
        // Min of 12 = 4.0, last 3 mean = (4.1+4.2+4.3)/3 = 4.2 → sahm 0.2
        const values = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.1, 4.2, 4.3];
        expect(sahmFromHistory(values.map((v, i) => ({ date: `2026-${String(i + 1).padStart(2, '0')}-01`, value: v })))).toBeCloseTo(0.2, 5);
    });

    test('returns null when less than 12 points', () => {
        expect(sahmFromHistory([])).toBeNull();
        const short = Array(11).fill(0).map((v, i) => ({ date: `2026-${String(i + 1).padStart(2, '0')}-01`, value: 4.0 }));
        expect(sahmFromHistory(short)).toBeNull();
        expect(sahmFromHistory(null)).toBeNull();
    });

    test('returns null for non-array input', () => {
        expect(sahmFromHistory(null)).toBeNull();
        expect(sahmFromHistory(undefined)).toBeNull();
        expect(sahmFromHistory('not-an-array')).toBeNull();
    });

    test('computed value can be near-zero when unemployment barely changes', () => {
        // Last 3 months: 4.0, 4.01, 4.02 → mean ≈ 4.01
        // Min of last 12: 3.98 (value[2])
        // Sahm ≈ 4.01 - 3.98 = 0.03
        const values = [4.1, 4.0, 3.98, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.01, 4.02];
        const history = values.map((v, i) => ({ date: `2026-${String(i + 1).padStart(2, '0')}-01`, value: v }));
        const result = sahmFromHistory(history);
        expect(result).toBeCloseTo(0.03, 5);
        expect(result).toBeGreaterThan(0);
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
