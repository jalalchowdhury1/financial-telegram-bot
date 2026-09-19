/**
 * v3.1 tiers for jevInputs — Treasury spread parser, single-value `read`
 * sources, deep cascades, injectable store.
 */
import { parseTreasurySpreadCsv, resolvePillInput } from '../jevInputs';

jest.mock('../store', () => ({
    loadLastGood: jest.fn(),
    saveLastGood: jest.fn(),
}));
import { loadLastGood, saveLastGood } from '../store';

beforeEach(() => jest.clearAllMocks());

const TREASURY_CSV = [
    'Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"',
    '09/18/2026,3.97,3.98,4.10,4.14,4.24,4.24,4.44,4.76,4.83,4.86,4.93,5.01,5.38,5.34',
    '09/17/2026,3.98,3.98,4.11,4.15,4.25,4.25,4.45,4.77,4.84,4.87,4.94,5.03,5.40,5.36',
    '09/16/2026,3.98,3.98,4.11,N/A,4.25,4.25,4.45,4.77,4.84,4.87,4.94,5.03,5.40,5.36',
].join('\n');

describe('parseTreasurySpreadCsv', () => {
    test('10 Yr − 3 Mo by header name, ascending, 2dp (matches FRED T10Y3M 0.87 on 2026-09-18)', () => {
        const rows = parseTreasurySpreadCsv(TREASURY_CSV, '3 mo', '10 yr');
        expect(rows).toEqual([
            { date: '2026-09-17', value: 0.88 },
            { date: '2026-09-18', value: 0.87 },
        ]);
    });

    test('is case-insensitive on the column names and skips rows with a missing tenor', () => {
        const rows = parseTreasurySpreadCsv(TREASURY_CSV, '3 Mo', '10 Yr');
        expect(rows.map((r) => r.date)).not.toContain('2026-09-16');
    });

    test('throws when a named column is absent (never a positional guess)', () => {
        expect(() => parseTreasurySpreadCsv(TREASURY_CSV, '3 mo', '40 yr')).toThrow(/column not found/);
    });

    test('empty / junk input → []', () => {
        expect(parseTreasurySpreadCsv('', '3 mo', '10 yr')).toEqual([]);
        expect(parseTreasurySpreadCsv(null, '3 mo', '10 yr')).toEqual([]);
    });
});

describe('resolvePillInput — read (single value) sources', () => {
    test('a fresh snapshot value wins', async () => {
        const r = await resolvePillInput({
            sources: [{ name: 'sheet', freshnessDays: 14, read: async () => ({ value: -0.56, asOf: '2026-09-11' }) }],
            now: new Date('2026-09-19'),
        });
        expect(r).toMatchObject({ value: -0.56, asOf: '2026-09-11', source: 'sheet' });
        expect(r.tried).toEqual(['sheet:ok']);
    });

    test('a stale snapshot is skipped with its date in tried', async () => {
        const r = await resolvePillInput({
            sources: [{ name: 'sheet', freshnessDays: 14, read: async () => ({ value: -0.56, asOf: '2026-08-01' }) }],
            now: new Date('2026-09-19'),
        });
        expect(r.value).toBeNull();
        expect(r.tried[0]).toBe('sheet:stale(2026-08-01)');
    });

    test('null / non-finite / undated snapshots count as empty', async () => {
        const mk = (v) => ({ name: 'sheet', freshnessDays: 14, read: async () => v });
        for (const v of [null, { value: NaN, asOf: '2026-09-11' }, { value: 1, asOf: null }]) {
            const r = await resolvePillInput({ sources: [mk(v)], now: new Date('2026-09-19') });
            expect(r.tried[0]).toBe('sheet:empty');
        }
    });
});

describe('resolvePillInput — deep cascade and diagnostics', () => {
    const hist = (dates) => dates.map((date, i) => ({ date, value: i + 1 }));

    test('stale → throwing → empty → last-good, with the error text recorded', async () => {
        loadLastGood.mockReturnValue({ data: { value: 42, asOf: '2026-09-10', source: 'fred' }, savedAt: '2026-09-10T00:00:00Z' });
        const r = await resolvePillInput({
            sources: [
                { name: 'a', freshnessDays: 7, fetch: async () => hist(['2026-08-01', '2026-08-02']) },
                { name: 'b', freshnessDays: 7, fetch: async () => { throw new Error('Fetch timed out for x after 5000ms'); } },
                { name: 'c', freshnessDays: 7, fetch: async () => [] },
            ],
            now: new Date('2026-09-19'),
            lastGoodKey: 'jev-x',
        });
        expect(r.value).toBe(42);
        expect(r.source).toBe('lastgood');
        expect(r.asOf).toBe('2026-09-10');
        expect(r.tried).toEqual([
            'a:stale(2026-08-02)',
            'b:err(Fetch timed out for x after 5000ms)',
            'c:empty',
            'lastgood:ok(2026-09-10T00:00:00Z)',
        ]);
    });

    test('a fault named without the hm_ prefix does not skip a source', async () => {
        const r = await resolvePillInput({
            sources: [{ name: 'fred', freshnessDays: 7, fetch: async () => hist(['2026-09-17', '2026-09-18']) }],
            faults: new Set(['fred']),
            now: new Date('2026-09-19'),
        });
        expect(r.source).toBe('fred');
        expect(saveLastGood).not.toHaveBeenCalled(); // any fault present → no write
    });

    test('injected store is used for both read and write, and the write omits tried', async () => {
        const store = { load: jest.fn(async () => null), save: jest.fn(async () => true) };
        const ok = await resolvePillInput({
            sources: [{ name: 'a', freshnessDays: 7, fetch: async () => hist(['2026-09-17', '2026-09-18']) }],
            now: new Date('2026-09-19'), lastGoodKey: 'jev-x', store,
        });
        expect(ok.source).toBe('a');
        expect(store.save).toHaveBeenCalledWith('jev-x', { value: 2, asOf: '2026-09-18', source: 'a' });
        expect(saveLastGood).not.toHaveBeenCalled();

        store.load.mockResolvedValue({ data: { value: 7, asOf: '2026-09-15' }, savedAt: '2026-09-15T00:00:00Z' });
        const miss = await resolvePillInput({
            sources: [{ name: 'a', freshnessDays: 7, fetch: async () => { throw new Error('down'); } }],
            now: new Date('2026-09-19'), lastGoodKey: 'jev-x', maxStaleMs: 14 * 864e5, store,
        });
        expect(store.load).toHaveBeenCalledWith('jev-x', 14 * 864e5);
        expect(miss).toMatchObject({ value: 7, asOf: '2026-09-15', source: 'lastgood' });
        expect(loadLastGood).not.toHaveBeenCalled();
    });

    test('a store that throws never breaks the cascade', async () => {
        const store = { load: jest.fn(async () => { throw new Error('kv down'); }), save: jest.fn(async () => { throw new Error('kv down'); }) };
        const ok = await resolvePillInput({
            sources: [{ name: 'a', freshnessDays: 7, fetch: async () => hist(['2026-09-17', '2026-09-18']) }],
            now: new Date('2026-09-19'), lastGoodKey: 'jev-x', store,
        });
        expect(ok.source).toBe('a');
        const miss = await resolvePillInput({
            sources: [{ name: 'a', freshnessDays: 7, fetch: async () => [] }],
            now: new Date('2026-09-19'), lastGoodKey: 'jev-x', store,
        });
        expect(miss.value).toBeNull();
        expect(miss.tried).toContain('lastgood:none');
    });
});
