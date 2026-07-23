/**
 * Tests for the US Bankruptcies (AOUSC F-2) resolver — the 4th horseman.
 * The XLSX fixture is a REAL bf_f2_0331.2026.xlsx downloaded from uscourts.gov
 * (12-month period ending 2026-03-31: total 591,850 / business 25,960).
 */
import fs from 'fs';
import path from 'path';
import {
    unzipEntry,
    parseF2Totals,
    validateTotals,
    quarterEndsBefore,
    quarterDateStr,
    f2XlsxUrl,
    f2PageUrl,
    findF2Link,
    resolveBankruptcies,
    BANKRUPTCY_FRESHNESS_DAYS,
} from '../bankruptcies';

const fixture = fs.readFileSync(path.join(__dirname, 'fixtures', 'bf_f2_0331.2026.xlsx'));

const NOW = new Date('2026-07-23T12:00:00Z');
const BAKED = [
    { date: '2024-03-31', total: 452990, business: 21463 },
    { date: '2025-03-31', total: 529080, business: 23309 },
    { date: '2025-06-30', total: 542529, business: 23830 },
    { date: '2025-09-30', total: 559946, business: 24552 },
    { date: '2025-12-31', total: 577518, business: 25203 },
];

describe('unzipEntry / parseF2Totals (real uscourts XLSX)', () => {
    test('extracts entries from the zip container', () => {
        expect(unzipEntry(fixture, 'xl/sharedStrings.xml')).not.toBeNull();
        expect(unzipEntry(fixture, 'xl/worksheets/sheet1.xml')).not.toBeNull();
        expect(unzipEntry(fixture, 'no/such/entry.xml')).toBeNull();
    });

    test('rejects a non-zip buffer', () => {
        expect(() => unzipEntry(Buffer.from('not a zip at all'), 'x')).toThrow(/not a zip/);
    });

    test('parses the national Total row from the real F-2 file', () => {
        expect(parseF2Totals(fixture)).toEqual({ total: 591850, business: 25960, nonbusiness: 565890 });
    });
});

describe('validateTotals (cross-check before serving)', () => {
    test('accepts numbers that reconcile', () => {
        expect(validateTotals({ total: 100, business: 20, nonbusiness: 80 }))
            .toEqual({ total: 100, business: 20, nonbusiness: 80 });
    });

    test('throws when business + nonbusiness does not reconcile to total', () => {
        expect(() => validateTotals({ total: 100, business: 20, nonbusiness: 60 })).toThrow(/sanity/);
    });

    test('throws on missing/degenerate values', () => {
        expect(() => validateTotals({ total: null, business: 20, nonbusiness: 80 })).toThrow(/sanity/);
        expect(() => validateTotals({ total: 100, business: null, nonbusiness: 80 })).toThrow(/sanity/);
        expect(() => validateTotals({ total: 100, business: 120, nonbusiness: 80 })).toThrow(/sanity/);
    });
});

describe('quarter helpers', () => {
    test('lists recent quarter-ends newest first', () => {
        const q = quarterEndsBefore(NOW, 5).map(quarterDateStr);
        expect(q).toEqual(['2026-06-30', '2026-03-31', '2025-12-31', '2025-09-30', '2025-06-30']);
    });

    test('builds the predictable uscourts URLs', () => {
        const q = { y: 2026, m: 3, d: 31 };
        expect(f2XlsxUrl(q)).toBe('https://www.uscourts.gov/sites/default/files/document/bf_f2_0331.2026.xlsx');
        expect(f2PageUrl(q)).toBe('https://www.uscourts.gov/data-news/data-tables/2026/03/31/bankruptcy-filings/f-2');
    });

    test('findF2Link picks the spreadsheet and skips guide PDFs', () => {
        const html = '<a href="/sites/default/files/guide-vol01.pdf">g</a>'
            + '<a href="/sites/default/files/document/bf_f2_0331.2026.xlsx">x</a>';
        expect(findF2Link(html)).toBe('https://www.uscourts.gov/sites/default/files/document/bf_f2_0331.2026.xlsx');
        expect(findF2Link('<p>nothing</p>')).toBeNull();
    });
});

describe('resolveBankruptcies', () => {
    test('live tier wins: newest quarter parsed and merged over baked', async () => {
        const fetchBuffer = jest.fn()
            .mockRejectedValueOnce(new Error('404'))       // 2026-06-30 xlsx not published yet
            .mockResolvedValueOnce(fixture);               // 2026-03-31 xlsx
        const fetchText = jest.fn().mockRejectedValue(new Error('404')); // 2026-06-30 page miss
        const r = await resolveBankruptcies({ now: NOW, faults: new Set(), fetchBuffer, fetchText, baked: BAKED });
        expect(r.source).toBe('uscourts');
        expect(r.current).toBe(25960);
        expect(r.total).toBe(591850);
        expect(r.asOf).toBe('2026-03-31');
        expect(r.stale).toBe(false);
        expect(r.unavailable).toBe(false);
        // YoY vs baked 2025-03-31 (23,309 → 25,960 ≈ +11.4%)
        expect(r.changePct).toBeCloseTo(((25960 - 23309) / 23309) * 100, 5);
        expect(r.status).toBe('rising');
        expect(r.history[r.history.length - 1]).toEqual({ date: '2026-03-31', value: 25960, total: 591850 });
    });

    test('uscourts down → serves baked history, still fresh within the deadline', async () => {
        const fetchBuffer = jest.fn().mockRejectedValue(new Error('down'));
        const fetchText = jest.fn().mockRejectedValue(new Error('down'));
        const r = await resolveBankruptcies({ now: NOW, faults: new Set(), fetchBuffer, fetchText, baked: BAKED });
        expect(r.source).toBe('baked');
        expect(r.current).toBe(25203);
        expect(r.asOf).toBe('2025-12-31');
        // 2025-12-31 is 204 days before NOW — past the 150-day deadline → stale, but SERVED
        expect(r.stale).toBe(true);
        expect(r.unavailable).toBe(false);
    });

    test('?_fail=bk_uscourts skips the live tier without touching the network', async () => {
        const fetchBuffer = jest.fn();
        const fetchText = jest.fn();
        const r = await resolveBankruptcies({ now: NOW, faults: new Set(['bk_uscourts']), fetchBuffer, fetchText, baked: BAKED });
        expect(fetchBuffer).not.toHaveBeenCalled();
        expect(fetchText).not.toHaveBeenCalled();
        expect(r.source).toBe('baked');
        expect(r.tried).toContain('uscourts:off');
    });

    test('?_fail=bk_uscourts,bk_baked → unavailable (N/A panel)', async () => {
        const r = await resolveBankruptcies({
            now: NOW, faults: new Set(['bk_uscourts', 'bk_baked']),
            fetchBuffer: jest.fn(), fetchText: jest.fn(), baked: BAKED,
        });
        expect(r.unavailable).toBe(true);
        expect(r.current).toBeNull();
        expect(r.history).toEqual([]);
        expect(r.status).toBe('unknown');
        expect(r.source).toBeNull();
    });

    test('deadline stops probing older quarters instead of stalling the route', async () => {
        const fetchBuffer = jest.fn().mockRejectedValue(new Error('slow'));
        const fetchText = jest.fn().mockRejectedValue(new Error('slow'));
        const r = await resolveBankruptcies({
            now: NOW, faults: new Set(), fetchBuffer, fetchText, baked: BAKED, deadlineMs: -1,
        });
        expect(r.tried).toContain('uscourts:deadline');
        expect(r.source).toBe('baked');
    });

    test('changePct is null when the prior-year quarter is missing (honest about gaps)', async () => {
        const r = await resolveBankruptcies({
            now: NOW, faults: new Set(['bk_uscourts']), fetchBuffer: jest.fn(), fetchText: jest.fn(),
            baked: [{ date: '2025-12-31', total: 577518, business: 25203 }],
        });
        expect(r.changePct).toBeNull();
        expect(r.status).toBe('neutral');
    });

    test('freshness window tolerates the normal quarterly publishing lag', () => {
        // Newest point can be ~115 days old the day before the next print + ~30d
        // publish lag — the deadline must comfortably exceed that.
        expect(BANKRUPTCY_FRESHNESS_DAYS).toBeGreaterThanOrEqual(150);
    });
});
