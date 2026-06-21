import { parseLkgCsv, reconstructFred } from '../sheetLkg';

const NOW = new Date('2026-06-21T18:00:00Z');

// A representative helper-tab CSV (key,value rows). Note: gviz wraps every cell in
// double quotes; include a couple of unquoted rows too to prove both parse.
const CSV = [
    '"key","value"',
    '"updated_at","2026-06-21T10:00:00Z"',
    '"peRatio","32.2"',
    '"yieldCurve.current","0.27"',
    '"yieldCurve.asOf","2026-06-18"',
    '"profitMargin.current","14.83"',
    '"profitMargin.asOf","2026-01-01"',
    '"indicators.sentiment.value","49.8"',
    '"indicators.sentiment.asOf","2026-04-01"',
    '"indicators.sentiment.status","weak"',
    '"checklist.m2.value","4.72"',
    '"checklist.m2.asOf","2026-04-01"',
    '"checklist.m2.status","good"',
    '"checklist.m2.bullish","true"',
    '"checklist.m2.label","M2 Money Supply"',
    '"checklist.savings.value","2.6"',
    '"checklist.savings.asOf","2026-04-01"',
    '"checklist.savings.status","weak"',
    '"checklist.savings.bullish","false"',
    '"checklist.savings.label","Savings Rate"',
].join('\n');

describe('parseLkgCsv', () => {
    test('parses quoted key,value rows into a map and skips the header', () => {
        const m = parseLkgCsv(CSV);
        expect(m['updated_at']).toBe('2026-06-21T10:00:00Z');
        expect(m['peRatio']).toBe('32.2');
        expect(m['checklist.m2.label']).toBe('M2 Money Supply');
        expect(m['key']).toBeUndefined(); // header skipped
    });

    test('tolerates blank lines and unquoted cells', () => {
        const m = parseLkgCsv('key,value\n\nfoo,1.5\n"bar","hi"\n');
        expect(m['foo']).toBe('1.5');
        expect(m['bar']).toBe('hi');
    });

    test('empty / junk input yields an empty map, never throws', () => {
        expect(parseLkgCsv('')).toEqual({});
        expect(parseLkgCsv(null)).toEqual({});
        expect(parseLkgCsv(undefined)).toEqual({});
    });
});

describe('reconstructFred', () => {
    test('builds a fred-shaped payload with correct types and stale flags', () => {
        const p = reconstructFred(parseLkgCsv(CSV), NOW);
        // top cards
        expect(p.yieldCurve).toMatchObject({ current: 0.27, asOf: '2026-06-18', stale: true });
        expect(Array.isArray(p.yieldCurve.history)).toBe(true);
        expect(p.yieldCurve.history.length).toBe(0);
        expect(p.profitMargin.current).toBe(14.83);
        expect(p.peRatio).toBe(32.2);
        expect(p.recessions).toEqual([]);
        // indicator: number value, string status, stale flagged
        expect(p.indicators.sentiment).toMatchObject({
            value: 49.8, asOf: '2026-04-01', status: 'weak', stale: true, unavailable: false,
        });
        expect(p.indicators.sentiment.staleDays).toBeGreaterThan(3); // health check should warn
        // checklist: bullish coerced to real booleans + label kept
        expect(p.checklist.m2).toMatchObject({ value: 4.72, bullish: true, status: 'good', label: 'M2 Money Supply', stale: true });
        expect(p.checklist.savings.bullish).toBe(false);
    });

    test('_meta marks the data stale + hasErrors so the health check still alerts', () => {
        const p = reconstructFred(parseLkgCsv(CSV), NOW);
        expect(p._meta.stale).toBe(true);
        expect(p._meta.hasErrors).toBe(true);
        expect(p._meta.source).toContain('Google Sheet');
        expect(p._meta.source).toContain('2026-06-21T10:00:00Z'); // updated_at surfaced
        expect(p._meta.loadedCount).toBe(0);
    });

    test('omits metrics absent from the sheet (cut-out N/A) rather than emitting null tiles', () => {
        const p = reconstructFred(parseLkgCsv(CSV), NOW);
        expect(p.indicators.claims).toBeUndefined();   // not in CSV
        expect(p.indicators.sentiment).toBeDefined();  // in CSV
    });

    test('drops a metric whose value is non-numeric junk (keeps the valid ones)', () => {
        const m = {
            'checklist.m2.value': '4.72',
            'indicators.sentiment.value': 'N/A', 'indicators.sentiment.asOf': '2026-04-01',
        };
        const p = reconstructFred(m, NOW);
        expect(p.indicators.sentiment).toBeUndefined();
        expect(p.checklist.m2.value).toBe(4.72);
    });

    test('returns null when there are no usable metrics', () => {
        expect(reconstructFred({}, NOW)).toBeNull();
        expect(reconstructFred({ updated_at: '2026-06-21T10:00:00Z' }, NOW)).toBeNull();
    });
});
