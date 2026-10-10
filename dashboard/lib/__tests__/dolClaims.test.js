import fs from 'fs';
import path from 'path';
import { parseDolClaimsXml, resolveHorseman, buildHorseman } from '../horsemen';

// Real DOL ETA national weekly-claims report, fetched 2026-10-09:
//   curl -d "level=nation&strtdate=2026&enddate=2026&filetype=xml" https://oui.doleta.gov/unemploy/wkclaims/report.asp
const XML = fs.readFileSync(path.join(__dirname, 'fixtures', 'dol-wkclaims-2026-10-09.xml'), 'utf8');

// FRED ICSA as served by prod /api/fred on 2026-10-09 (horsemen.claims.history).
const FRED_ICSA = {
    '2026-07-18': 189000, '2026-07-25': 198000, '2026-08-01': 200000, '2026-08-08': 212000,
    '2026-08-15': 207000, '2026-08-22': 204000, '2026-08-29': 207000, '2026-09-05': 207000,
    '2026-09-12': 198000,
};

describe('parseDolClaimsXml', () => {
    const series = parseDolClaimsXml(XML);

    test('parses every week, ascending, ISO dates', () => {
        expect(series).toHaveLength(37);
        expect(series[0]).toEqual({ date: '2026-01-03', value: 207000 });
        expect(series[series.length - 1]).toEqual({ date: '2026-09-12', value: 198000 });
        for (let i = 1; i < series.length; i++) expect(series[i].date > series[i - 1].date).toBe(true);
    });

    test('is the SEASONALLY ADJUSTED series — matches FRED ICSA on every overlapping week', () => {
        const byDate = Object.fromEntries(series.map((p) => [p.date, p.value]));
        for (const [date, value] of Object.entries(FRED_ICSA)) expect(byDate[date]).toBe(value);
    });

    test('never reads NSA or the continued-claims SA', () => {
        // Week of 01/03: NSA 298,700 · continued SA 1,875,000 · initial SA 207,000
        const first = series[0].value;
        expect(first).not.toBe(298700);
        expect(first).not.toBe(1875000);
        // Every value is in the initial-claims range, not the millions of continued claims.
        for (const p of series) expect(p.value).toBeLessThan(1000000);
    });

    test('a week with no SA initial claims is skipped, not zero-filled', () => {
        const xml = '<r><week><weekEnded>10/03/2026</weekEnded><InitialClaims><NSA>200,000</NSA><SA></SA></InitialClaims>'
            + '<ContinuedClaims><SA>1,900,000</SA></ContinuedClaims></week>'
            + '<week><weekEnded>09/26/2026</weekEnded><InitialClaims><SA>199,000</SA></InitialClaims></week></r>';
        expect(parseDolClaimsXml(xml)).toEqual([{ date: '2026-09-26', value: 199000 }]);
    });

    test('garbage / HTML error page → [] (no data, never wrong data)', () => {
        expect(parseDolClaimsXml('<html><body>Service Unavailable</body></html>')).toEqual([]);
        expect(parseDolClaimsXml('')).toEqual([]);
        expect(parseDolClaimsXml(null)).toEqual([]);
    });
});

describe('DOL as the claims tier 2', () => {
    const NOW = new Date('2026-10-09T12:00:00Z');
    const dolSrc = { name: 'dol', freshnessDays: 35, fetch: async () => parseDolClaimsXml(XML) };
    const deadFredCsv = { name: 'fredcsv', freshnessDays: 14, fetch: async () => { throw new Error('phantom'); } };

    test('resolves a ~4-week-lagged DOL week inside its own 35-day window', async () => {
        const r = await resolveHorseman([dolSrc, deadFredCsv], new Set(), NOW);
        expect(r.source).toBe('dol');
        expect(r.currentDate).toBe('2026-09-12');
        expect(r.current).toBe(198000);
        expect(r.tried).toEqual(['dol:ok(37)']);
    });

    test('is then stamped against ICSA\'s 14-day deadline: own as-of date + stale flag', async () => {
        const r = await resolveHorseman([dolSrc], new Set(), NOW);
        const h = buildHorseman(r, 14, NOW);
        expect(h.asOf).toBe('2026-09-12');
        expect(h.stale).toBe(true);           // never passes as this week's print
        expect(h.staleDays).toBeGreaterThan(0);
        expect(h.current).toBe(198000);       // kept (flagged), not blanked
        expect(h.source).toBe('dol');
    });

    test('?_fail=hm_dol switches the tier off and falls through', async () => {
        const r = await resolveHorseman([dolSrc, deadFredCsv], new Set(['hm_dol']), NOW);
        expect(r.source).toBeNull();
        expect(r.tried).toEqual(['dol:off', 'fredcsv:err']);
    });

    test('a DOL feed older than 35 days is rejected, not served', async () => {
        const later = new Date('2026-11-01T12:00:00Z'); // 2026-09-12 is 50 days old
        const r = await resolveHorseman([dolSrc], new Set(), later);
        expect(r.source).toBeNull();
        expect(r.tried).toEqual(['dol:stale(2026-09-12)']);
    });
});
