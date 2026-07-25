import { parseLkgCsv, reconstructFred, parsePackedHistory } from '../sheetLkg';

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

// ─── Four Horsemen in the last-resort tier (added 2026-07-25) ───────────────
// Regression guard: before this, reconstructFred emitted no `horsemen` key and
// no history, so the deepest fallback restored every card EXCEPT the
// recession-watch one, which rendered "N/A — Unavailable".
describe('reconstructFred — Four Horsemen', () => {
    const csv = [
        'key,value',
        'updated_at,2026-07-25T02:00:00Z',
        'yieldCurve.current,0.36',
        'yieldCurve.asOf,2026-07-24',
        'yieldCurve.history,2026-07-17:0.30|2026-07-24:0.36',
        'horsemen.claims.value,187000',
        'horsemen.claims.asOf,2026-07-18',
        'horsemen.claims.history,2026-07-11:190000|2026-07-18:187000',
        'horsemen.unemployment.value,4.2',
        'horsemen.unemployment.asOf,2026-06-01',
        'horsemen.unemployment.history,2026-05-01:4.1|2026-06-01:4.2',
        'horsemen.bankruptcies.value,25960',
        'horsemen.bankruptcies.asOf,2026-03-31',
        'horsemen.bankruptcies.total,591850',
        'horsemen.bankruptcies.changePct,11.37',
        'horsemen.bankruptcies.status,rising',
        'horsemen.bankruptcies.history,2025-03-31:23310|2026-03-31:25960',
    ].join('\n');

    const out = () => reconstructFred(parseLkgCsv(csv), new Date('2026-07-25T12:00:00Z'));

    test('restores all four lines with enough history to draw the chart', () => {
        const p = out();
        expect(p.horsemen.claims.current).toBe(187000);
        expect(p.horsemen.unemployment.current).toBe(4.2);
        expect(p.horsemen.bankruptcies.current).toBe(25960);
        expect(p.yieldCurve.current).toBe(0.36);
        // >= 2 points each is what the card's hasAnySeries check requires.
        for (const h of [p.horsemen.claims, p.horsemen.unemployment, p.horsemen.bankruptcies, p.yieldCurve]) {
            expect(h.history.length).toBeGreaterThanOrEqual(2);
        }
    });

    test('history is ascending, matching the live payload contract', () => {
        const h = out().horsemen.claims.history;
        expect(h).toEqual([
            { date: '2026-07-11', value: 190000 },
            { date: '2026-07-18', value: 187000 },
        ]);
    });

    test('carries the bankruptcies extras the stat chip reads', () => {
        const bk = out().horsemen.bankruptcies;
        expect(bk).toMatchObject({ total: 591850, changePct: 11.37, status: 'rising' });
    });

    test('every restored line is flagged stale so the health check still alerts', () => {
        const p = out();
        expect(p.horsemen.claims.stale).toBe(true);
        expect(p.horsemen.claims.staleDays).toBeGreaterThan(3);
        expect(p._meta.hasErrors).toBe(true);
    });

    test('a tab written by the OLD scraper (no horsemen keys) still parses', () => {
        const legacy = parseLkgCsv('key,value\nyieldCurve.current,0.36\npeRatio,28.1');
        const p = reconstructFred(legacy, new Date('2026-07-25T12:00:00Z'));
        expect(p).not.toBeNull();
        expect(p.horsemen).toEqual({});
        expect(p.yieldCurve.current).toBe(0.36);
    });

    test('a horseman present as history only still yields a current value', () => {
        const m = parseLkgCsv('key,value\nhorsemen.claims.history,2026-07-11:190000|2026-07-18:187000');
        const p = reconstructFred(m, new Date('2026-07-25T12:00:00Z'));
        expect(p.horsemen.claims.current).toBe(187000);
        expect(p.horsemen.claims.asOf).toBe('2026-07-18');
    });
});

describe('parsePackedHistory', () => {
    test('skips malformed segments rather than guessing', () => {
        expect(parsePackedHistory('2026-07-18:187000|garbage|2026-07-11:notanumber|:5'))
            .toEqual([{ date: '2026-07-18', value: 187000 }]);
    });
    test('tolerates empty/undefined input', () => {
        expect(parsePackedHistory('')).toEqual([]);
        expect(parsePackedHistory(undefined)).toEqual([]);
    });
});
