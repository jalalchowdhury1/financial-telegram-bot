import { resolveSpEps } from '../spEps';

// The live situation on 2026-10-09: multpl serves 295.36 (asOf 2026-06-30, normal
// as-reported lag); the datahub Shiller mirror froze at 181.77 (asOf 2023-06-01).
const NOW = new Date('2026-10-09T12:00:00Z');
const multplHist = [{ date: '2026-03-31', value: 280.1 }, { date: '2026-06-30', value: 295.36 }];
const datahubHist = [{ date: '2023-05-01', value: 179.17 }, { date: '2023-06-01', value: 181.77 }];

const multpl = (over = {}) => ({ name: 'multpl', freshnessDays: 400, fetch: async () => ({ current: 295.36, currentDate: '2026-06-30', historyAsc: multplHist }), ...over });
const derived = (over = {}) => ({ name: 'derived', freshnessDays: 7, fetch: async () => { throw new Error('no TTM P/E'); }, ...over });
// As wired in app/api/fred/route.js: 183-day hard ceiling.
const datahub = (over = {}) => ({ name: 'datahub', freshnessDays: 400, maxAgeDays: 183, fetch: async () => ({ current: 181.77, currentDate: '2023-06-01', historyAsc: datahubHist }), ...over });

describe('S&P EPS — datahub hard ceiling (maxAgeDays)', () => {
    test('happy path unchanged: multpl wins, datahub never fetched', async () => {
        const r = await resolveSpEps([multpl(), derived(), datahub()], new Set(), NOW);
        expect(r).toMatchObject({ current: 295.36, source: 'multpl', stale: false, historySource: 'multpl' });
    });

    test('multpl + derived down → a 3-year-old datahub is REJECTED: N/A, not 181.77', async () => {
        const r = await resolveSpEps([multpl(), derived(), datahub()], new Set(['eps_multpl']), NOW);
        expect(r.current).toBeNull();
        expect(r.unavailable).toBe(true);
        expect(r.source).toBeNull();
        // Its 2023-ending chart is not used either — it would look current.
        expect(r.history).toEqual([]);
        expect(r.historySource).toBeNull();
        expect(r.tried).toEqual(['multpl:off', 'derived:err', 'datahub:tooold(2023-06-01)']);
    });

    test('a datahub that is merely a few months behind is still a graceful-staleness fallback', async () => {
        const recent = datahub({ fetch: async () => ({ current: 290, currentDate: '2026-06-01', historyAsc: [{ date: '2026-06-01', value: 290 }] }) });
        const r = await resolveSpEps([multpl(), derived(), recent], new Set(['eps_multpl']), NOW);
        expect(r.current).toBe(290);
        expect(r.source).toBe('datahub');
    });

    test('a ceiling-declared source with no date is treated as too old', async () => {
        const undated = datahub({ fetch: async () => ({ current: 290, currentDate: null, historyAsc: [] }) });
        const r = await resolveSpEps([undated], new Set(), NOW);
        expect(r.unavailable).toBe(true);
        expect(r.tried).toEqual(['datahub:tooold(no date)']);
    });

    test('sources without maxAgeDays keep the old graceful-staleness behavior', async () => {
        const old = { name: 'multpl', freshnessDays: 400, fetch: async () => ({ current: 200, currentDate: '2024-06-30', historyAsc: [] }) };
        const r = await resolveSpEps([old], new Set(), NOW);
        expect(r).toMatchObject({ current: 200, stale: true, source: 'multpl' });
    });
});
