import { parseMultplEps, parseShillerCsv, toQuarterlyHistory, resolveSpEps } from '../spEps';

const NOW = new Date('2026-07-03T12:00:00Z');
const iso = (daysAgo) => new Date(NOW.getTime() - daysAgo * 86400000).toISOString().slice(0, 10);

// A source descriptor whose fetch resolves to `result` (or throws if it's an Error).
const src = (name, freshnessDays, result) => ({
    name,
    freshnessDays,
    fetch: async () => { if (result instanceof Error) throw result; return result; },
});

// Real structure from multpl.com/s-p-500-earnings/table/by-month (whitespace + &#x2002; included).
const MULTPL_HTML = `
<table id="datatable">
<tr> <th>Date</th> <th>Value</th> </tr>
<tr class="odd"> <td>Sep 30, 2025</td> <td> &#x2002; 241.50 </td> </tr>
<tr class="even"> <td>Aug 31, 2025</td> <td> &#x2002; 238.14 </td> </tr>
<tr class="odd"> <td>Jun 30, 2025</td> <td> &#x2002; 231.20 </td> </tr>
<tr class="even"> <td>Dec 31, 1946</td> <td> &#x2002; 1.06 </td> </tr>
<tr class="odd"> <td>Jan 31, 1871</td> <td> &#x2002; 0.40 </td> </tr>
</table>`;

// Real structure from the datasets/s-and-p-500 GitHub mirror of Shiller's data:
// recent months have price but 0.0 in every earnings-related column.
const SHILLER_CSV = `Date,SP500,Dividend,Earnings,Consumer Price Index,Long Interest Rate,Real Price,Real Dividend,Real Earnings,PE10
1871-01-01,4.44,0.26,0.4,12.46,5.32,109.05,6.39,9.82,0.0
2023-05-01,4146.17,67.29,179.17,304.13,3.57,4300.0,69.79,185.83,28.9
2023-06-01,4345.37,67.79,181.17,305.11,3.75,4491.0,70.06,187.24,30.1
2026-05-01,7412.55,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2026-06-01,7450.03,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0`;

describe('parseMultplEps', () => {
    test('parses rows to ascending {date, value}', () => {
        const rows = parseMultplEps(MULTPL_HTML);
        expect(rows[0]).toEqual({ date: '1871-01-31', value: 0.4 });
        expect(rows[rows.length - 1]).toEqual({ date: '2025-09-30', value: 241.5 });
        expect(rows).toHaveLength(5);
    });

    test('returns [] on garbage', () => {
        expect(parseMultplEps('<html>nope</html>')).toEqual([]);
        expect(parseMultplEps('')).toEqual([]);
    });
});

describe('parseShillerCsv', () => {
    test('keeps only rows with a genuine Real Earnings value, ascending', () => {
        const rows = parseShillerCsv(SHILLER_CSV);
        expect(rows[0]).toEqual({ date: '1871-01-01', value: 9.82 });
        expect(rows[rows.length - 1]).toEqual({ date: '2023-06-01', value: 187.24 });
        expect(rows).toHaveLength(3); // the 0.0 placeholder months are dropped
    });

    test('returns [] when the header is missing or malformed', () => {
        expect(parseShillerCsv('not,a,shiller,file\n1,2,3,4')).toEqual([]);
        expect(parseShillerCsv('')).toEqual([]);
    });
});

describe('toQuarterlyHistory', () => {
    test('keeps quarter-end months from 1947 on', () => {
        const hist = [
            { date: '1946-12-31', value: 1.06 },  // pre-1947 → dropped
            { date: '2025-06-30', value: 231.2 }, // Jun → kept
            { date: '2025-08-31', value: 238.14 },// Aug → dropped
            { date: '2025-09-30', value: 241.5 }, // Sep → kept
        ];
        expect(toQuarterlyHistory(hist)).toEqual([
            { date: '2025-06-30', value: 231.2 },
            { date: '2025-09-30', value: 241.5 },
        ]);
    });

    test('handles empty/missing input', () => {
        expect(toQuarterlyHistory([])).toEqual([]);
        expect(toQuarterlyHistory(undefined)).toEqual([]);
    });
});

describe('resolveSpEps cascade', () => {
    const multpl = (daysAgo = 276) => src('multpl', 400, {
        current: 241.5, currentDate: iso(daysAgo),
        historyAsc: [{ date: iso(daysAgo + 90), value: 238 }, { date: iso(daysAgo), value: 241.5 }],
    });
    const derived = () => src('derived', 7, { current: 243.1, currentDate: iso(1), historyAsc: [] });
    const datahub = () => src('datahub', 400, {
        current: 181.17, currentDate: iso(1100),
        historyAsc: [{ date: iso(1190), value: 179 }, { date: iso(1100), value: 181.17 }],
    });

    test('multpl healthy → level + history from multpl, no further fetches', async () => {
        let datahubHit = false;
        const spyDatahub = { name: 'datahub', freshnessDays: 400, fetch: async () => { datahubHit = true; return null; } };
        const r = await resolveSpEps([multpl(), derived(), spyDatahub], new Set(), NOW);
        expect(r.source).toBe('multpl');
        expect(r.current).toBe(241.5);
        expect(r.stale).toBe(false);
        expect(r.history).toHaveLength(2);
        expect(datahubHit).toBe(false); // polite: stops once level + history are in hand
    });

    test('multpl off → derived gives the fresh level, datahub fills the history', async () => {
        const r = await resolveSpEps([multpl(), derived(), datahub()], new Set(['eps_multpl']), NOW);
        expect(r.source).toBe('derived');
        expect(r.current).toBe(243.1);
        expect(r.stale).toBe(false);
        expect(r.historySource).toBe('datahub');
        expect(r.history).toHaveLength(2);
        expect(r.tried).toEqual(['multpl:off', 'derived:ok', expect.stringMatching(/^datahub:stale/)]);
    });

    test('multpl + derived off → datahub serves a STALE level (graceful staleness)', async () => {
        const r = await resolveSpEps([multpl(), derived(), datahub()], new Set(['eps_multpl', 'eps_derived']), NOW);
        expect(r.source).toBe('datahub');
        expect(r.current).toBe(181.17);
        expect(r.stale).toBe(true);
        expect(r.unavailable).toBe(false);
        expect(r.history).toHaveLength(2);
    });

    test('a fresh level from a later source beats an earlier stale one', async () => {
        const staleMultpl = src('multpl', 400, { current: 200, currentDate: iso(500), historyAsc: [{ date: iso(500), value: 200 }] });
        const r = await resolveSpEps([staleMultpl, derived(), datahub()], new Set(), NOW);
        expect(r.source).toBe('derived');
        expect(r.stale).toBe(false);
        expect(r.historySource).toBe('multpl'); // stale source still contributes history
    });

    test('errors and empty results fall through', async () => {
        const r = await resolveSpEps([
            src('multpl', 400, new Error('boom')),
            src('derived', 7, { current: NaN, currentDate: iso(0), historyAsc: [] }),
            datahub(),
        ], new Set(), NOW);
        expect(r.source).toBe('datahub');
        expect(r.tried).toEqual(['multpl:err', 'derived:empty', expect.stringMatching(/^datahub:stale/)]);
    });

    test('everything down → unavailable, never throws', async () => {
        const r = await resolveSpEps([src('multpl', 400, new Error('x'))], new Set(['eps_derived']), NOW);
        expect(r.current).toBeNull();
        expect(r.unavailable).toBe(true);
        expect(r.stale).toBe(false);
        expect(r.history).toEqual([]);
    });
});
