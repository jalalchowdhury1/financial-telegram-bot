import {
    parseTreasuryCsv, parseBlsSeries, parseFredGraphCsv,
    resolveHorseman, buildHorseman, needsRepair, isUpgrade, splitCsvLine, mergeHorsemenOverBase,
} from '../horsemen';

const NOW = new Date('2026-07-25T12:00:00Z');
const iso = (daysAgo) => new Date(NOW.getTime() - daysAgo * 86400000).toISOString().slice(0, 10);

// A source descriptor whose fetch resolves to `result` (or throws if it's an Error).
const src = (name, freshnessDays, result) => ({
    name, freshnessDays,
    fetch: async () => { if (result instanceof Error) throw result; return result; },
});
const healthy = (name, value, daysAgo = 0) =>
    src(name, 7, [{ date: iso(daysAgo + 7), value: value - 1 }, { date: iso(daysAgo), value }]);

// ─────────────────────────────────────────────────────────────────────────────
describe('parseTreasuryCsv', () => {
    // Real header shape from home.treasury.gov (note "1.5 Month", added later —
    // this is exactly why the parser is column-anchored rather than positional).
    const HEADER = 'Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"';
    const CSV = [
        HEADER,
        '07/24/2026,3.80,3.88,3.95,3.96,4.04,4.08,4.14,4.33,4.36,4.43,4.55,4.69,5.18,5.16',
        '07/23/2026,3.81,3.89,3.96,3.97,4.05,4.09,4.15,4.30,4.34,4.41,4.53,4.66,5.16,5.14',
    ].join('\n');

    test('computes 10Y-2Y and returns ascending points', () => {
        const out = parseTreasuryCsv(CSV);
        expect(out).toEqual([
            { date: '2026-07-23', value: 0.36 },
            { date: '2026-07-24', value: 0.36 },
        ]);
    });

    test('matches the live FRED T10Y2Y value exactly (4.69 - 4.33 = 0.36)', () => {
        // Guards against float drift like 0.3599999999999999 reaching the UI.
        expect(parseTreasuryCsv(CSV).at(-1).value).toBe(0.36);
    });

    test('reads columns by NAME, so an inserted tenor cannot shift the result', () => {
        const shifted = CSV.replace('"1 Mo",', '"1 Wk","1 Mo",')
            .replace(/^(07\/\d\d\/2026),/gm, '$1,3.70,');
        expect(parseTreasuryCsv(shifted).at(-1).value).toBe(0.36);
    });

    test('never mistakes "20 Yr" for "2 Yr"', () => {
        // If it did, the spread would be 4.69 - 5.18 = -0.49 (a false inversion —
        // the single most consequential way this parser could be wrong).
        expect(parseTreasuryCsv(CSV).at(-1).value).not.toBeCloseTo(-0.49);
    });

    test('throws rather than guessing when the tenor columns are gone', () => {
        expect(() => parseTreasuryCsv('Date,"1 Mo"\n07/24/2026,3.80')).toThrow(/2 Yr/);
    });

    test('skips rows with blank yields and tolerates junk input', () => {
        expect(parseTreasuryCsv(`${HEADER}\n07/24/2026,,,,,,,,,,,,,,`)).toEqual([]);
        expect(parseTreasuryCsv('')).toEqual([]);
        expect(parseTreasuryCsv(null)).toEqual([]);
    });
});

// ─────────────────────────────────────────────────────────────────────────────
describe('parseBlsSeries', () => {
    const payload = (data) => ({ status: 'REQUEST_SUCCEEDED', Results: { series: [{ data }] } });

    test('parses monthly points ascending, dated on the 1st like FRED', () => {
        const out = parseBlsSeries(payload([
            { year: '2026', period: 'M06', value: '4.2' },
            { year: '2026', period: 'M05', value: '4.1' },
        ]));
        expect(out).toEqual([
            { date: '2026-05-01', value: 4.1 },
            { date: '2026-06-01', value: 4.2 },
        ]);
    });

    test('drops M13 — the ANNUAL AVERAGE, not a 13th month', () => {
        const out = parseBlsSeries(payload([
            { year: '2025', period: 'M13', value: '4.0' },
            { year: '2025', period: 'M12', value: '4.3' },
        ]));
        expect(out).toEqual([{ date: '2025-12-01', value: 4.3 }]);
    });

    test('throws on an error payload instead of returning an empty series', () => {
        expect(() => parseBlsSeries({ status: 'REQUEST_NOT_PROCESSED', Results: {} })).toThrow(/BLS/);
    });
});

// ─────────────────────────────────────────────────────────────────────────────
describe('parseFredGraphCsv', () => {
    test('parses the keyless graph CSV', () => {
        const out = parseFredGraphCsv('observation_date,ICSA\n2026-07-11,190000\n2026-07-18,187000\n');
        expect(out).toEqual([
            { date: '2026-07-11', value: 190000 },
            { date: '2026-07-18', value: 187000 },
        ]);
    });

    test('drops the "." missing-observation marker', () => {
        const out = parseFredGraphCsv('DATE,T10Y2Y\n2026-07-03,.\n2026-07-04,0.36\n');
        expect(out).toEqual([{ date: '2026-07-04', value: 0.36 }]);
    });

    test('throws when the payload is an HTML error page, not a CSV', () => {
        expect(() => parseFredGraphCsv('<!doctype html>\n<html>blocked</html>')).toThrow(/header/);
    });
});

// ─────────────────────────────────────────────────────────────────────────────
describe('resolveHorseman cascade', () => {
    test('returns the first healthy source', async () => {
        const r = await resolveHorseman([healthy('treasury', 0.36), healthy('fredcsv', 0.35)], new Set(), NOW);
        expect(r.source).toBe('treasury');
        expect(r.current).toBe(0.36);
    });

    test('"turn off option 1 → option 2 answers" (the owner\'s test)', async () => {
        const r = await resolveHorseman([healthy('treasury', 0.36), healthy('fredcsv', 0.35)], new Set(['hm_treasury']), NOW);
        expect(r.source).toBe('fredcsv');
        expect(r.tried).toEqual(['treasury:off', 'fredcsv:ok(2)']);
    });

    test('falls through a thrown error to the next provider', async () => {
        const r = await resolveHorseman([src('treasury', 7, new Error('502')), healthy('fredcsv', 0.35)], new Set(), NOW);
        expect(r.source).toBe('fredcsv');
        expect(r.tried).toEqual(['treasury:err', 'fredcsv:ok(2)']);
    });

    test('rejects a provider whose newest point is already stale', async () => {
        const r = await resolveHorseman([healthy('treasury', 0.36, 90), healthy('fredcsv', 0.35)], new Set(), NOW);
        expect(r.source).toBe('fredcsv');
        expect(r.tried[0]).toMatch(/^treasury:stale/);
    });

    test('rejects a single-point series (cannot draw a line or fit a trend)', async () => {
        const one = src('treasury', 7, [{ date: iso(0), value: 0.36 }]);
        const r = await resolveHorseman([one, healthy('fredcsv', 0.35)], new Set(), NOW);
        expect(r.tried).toContain('treasury:empty');
        expect(r.source).toBe('fredcsv');
    });

    test('every provider failing yields a null result rather than throwing', async () => {
        const r = await resolveHorseman([src('treasury', 7, new Error('x')), src('fredcsv', 7, new Error('y'))], new Set(), NOW);
        expect(r.source).toBeNull();
        expect(r.current).toBeNull();
        expect(r.history).toEqual([]);
    });
});

// ─────────────────────────────────────────────────────────────────────────────
describe('needsRepair / isUpgrade guards', () => {
    const good = { history: [{ date: '2026-07-18', value: 1 }, { date: '2026-07-25', value: 2 }], asOf: '2026-07-25', stale: false, unavailable: false };

    test('a healthy line is left alone (so the happy path costs no network)', () => {
        expect(needsRepair(good)).toBe(false);
    });

    test('repairs when empty, unavailable, or stale', () => {
        expect(needsRepair(undefined)).toBe(true);
        expect(needsRepair({ ...good, history: [] })).toBe(true);
        expect(needsRepair({ ...good, unavailable: true })).toBe(true);
        expect(needsRepair({ ...good, stale: true })).toBe(true);
    });

    test('a fallback is adopted only when genuinely NEWER than the primary', () => {
        expect(isUpgrade({ currentDate: '2026-07-26' }, good)).toBe(true);
        expect(isUpgrade({ currentDate: '2026-07-25' }, good)).toBe(false);
        expect(isUpgrade({ currentDate: '2026-06-01' }, good)).toBe(false); // lagging BLS print
        expect(isUpgrade({ currentDate: null }, good)).toBe(false);
        expect(isUpgrade({ currentDate: '2026-01-01' }, { asOf: null })).toBe(true);
    });
});

// ─────────────────────────────────────────────────────────────────────────────
describe('buildHorseman', () => {
    test('produces the freshness contract the card and health check expect', () => {
        const r = { current: 4.2, currentDate: iso(1), history: [{ date: iso(8), value: 4.1 }, { date: iso(1), value: 4.2 }], source: 'bls', tried: ['bls:ok(2)'] };
        const h = buildHorseman(r, 80, NOW);
        expect(h).toMatchObject({ current: 4.2, asOf: iso(1), stale: false, unavailable: false, source: 'bls' });
        expect(h.history).toHaveLength(2);
    });

    test('keeps a stale value visible (orange 🕐) rather than blanking it', () => {
        const r = { current: 4.2, currentDate: iso(200), history: [{ date: iso(400), value: 4.0 }, { date: iso(200), value: 4.2 }], source: 'bls', tried: [] };
        const h = buildHorseman(r, 80, NOW);
        expect(h.current).toBe(4.2);
        expect(h.stale).toBe(true);
        expect(h.staleDays).toBeGreaterThan(0);
    });
});

describe('splitCsvLine', () => {
    test('honors quoted fields containing commas', () => {
        expect(splitCsvLine('a,"b,c",d')).toEqual(['a', 'b,c', 'd']);
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// The scenario this whole module exists for: FRED is gone, but Treasury/BLS —
// who ORIGINATE these series — are fine. The live lines must survive, and must
// not be mistaken for a complete fresh payload.
describe('mergeHorsemenOverBase', () => {
    const base = {
        indicators: { sahmRule: { value: 0.03 } },
        horsemen: { claims: { current: 1, history: [] }, bankruptcies: { current: 25960 } },
        yieldCurve: { current: 0.30, asOf: '2026-07-01', history: [] },
        _meta: { source: 'cache', messages: ['served last-known-good'] },
    };
    const live = {
        horsemen: { claims: { current: 187000, asOf: '2026-07-18', source: 'fredcsv', history: [1, 2] } },
        yieldCurve: { current: 0.36, asOf: '2026-07-24', source: 'treasury', history: [1, 2] },
        messages: ['Horseman claims: fredcsv [fredcsv:ok(3107)]'],
    };

    test('live lines win, everything else is kept from the cache', () => {
        const m = mergeHorsemenOverBase(base, live);
        expect(m.horsemen.claims.current).toBe(187000);        // live
        expect(m.yieldCurve.current).toBe(0.36);               // live
        expect(m.horsemen.bankruptcies.current).toBe(25960);   // cached, preserved
        expect(m.indicators.sahmRule.value).toBe(0.03);        // cached, preserved
    });

    test('stays honestly labelled as degraded so the health check still alerts', () => {
        const m = mergeHorsemenOverBase(base, live);
        expect(m._meta.stale).toBe(true);
        expect(m._meta.hasErrors).toBe(true);
        expect(m._meta.messages.join(' ')).toMatch(/served LIVE from independent sources/);
    });

    // Found on prod: the merged payload inherited the CACHED snapshot's
    // source ("St. Louis Fed") and messages ("Loaded 17/17 series") verbatim, so it
    // read as fresh FRED data and flatly contradicted loadedCount: 0.
    test('never presents cached provenance as live data', () => {
        const m = mergeHorsemenOverBase(
            { ...base, _meta: { source: 'St. Louis Fed', messages: ['Loaded 17/17 series'] } },
            { ...live, baseLabel: 'cache (last-known-good 2026-07-25T04:00:00Z)' },
        );
        expect(m._meta.source).toBe('cache (last-known-good 2026-07-25T04:00:00Z) + live Horsemen (claims, spread)');
        expect(m._meta.source).not.toBe('St. Louis Fed');
        // The inherited message must not read as this response's own status.
        expect(m._meta.messages).toContain('cached: Loaded 17/17 series');
        expect(m._meta.messages).not.toContain('Loaded 17/17 series');
    });

    test('labels the base sensibly when there is no cache at all', () => {
        const m = mergeHorsemenOverBase(null, live);
        expect(m._meta.source).toMatch(/live Horsemen/);
    });

    test('carries the per-line cascade trail through, for prod tier verification', () => {
        const m = mergeHorsemenOverBase(base, live);
        expect(m._meta.messages).toContain('Horseman claims: fredcsv [fredcsv:ok(3107)]');
    });

    test('works with no cached base at all (cold instance, nothing stored)', () => {
        const m = mergeHorsemenOverBase(null, live);
        expect(m.horsemen.claims.current).toBe(187000);
        expect(m._meta.hasErrors).toBe(true);
    });

    test('returns null when there is nothing live to overlay', () => {
        expect(mergeHorsemenOverBase(base, { horsemen: {}, yieldCurve: null })).toBeNull();
        expect(mergeHorsemenOverBase(base, null)).toBeNull();
    });
});
