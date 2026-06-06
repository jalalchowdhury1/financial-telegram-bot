import { resolveLeg, buildCopperGold } from '../copperGold';

const NOW = new Date('2026-06-06T12:00:00Z');
const iso = (daysAgo) => new Date(NOW.getTime() - daysAgo * 86400000).toISOString().slice(0, 10);

// Build an ascending daily-ish history from [{daysAgo, price}] points.
const hist = (points) =>
    points.map((p) => ({ date: iso(p.daysAgo), price: p.price })).sort((a, b) => (a.date < b.date ? -1 : 1));

// A source descriptor whose fetch resolves to `result` (or throws if it's an Error).
const src = (name, freshnessDays, result) => ({
    name,
    freshnessDays,
    fetch: async () => { if (result instanceof Error) throw result; return result; },
});

const healthy = (name, current, daysAgoOfData = 0) =>
    src(name, 7, { current, currentDate: iso(daysAgoOfData), historyAsc: hist([{ daysAgo: daysAgoOfData, price: current }]) });

describe('resolveLeg cascade', () => {
    test('returns the first healthy source', async () => {
        const leg = await resolveLeg([healthy('cnbc', 6.2), healthy('fred', 6.0)], new Set(), NOW);
        expect(leg.source).toBe('cnbc');
        expect(leg.current).toBe(6.2);
    });

    test('skips a fault-injected source and uses the next', async () => {
        const leg = await resolveLeg([healthy('cnbc', 6.2), healthy('fred', 6.0)], new Set(['cg_cnbc']), NOW);
        expect(leg.source).toBe('fred');
        expect(leg.tried).toContain('cnbc:off');
    });

    test('"turn off option 1 AND 2 → option 3 answers" (the owner\'s test)', async () => {
        const sources = [healthy('cnbc', 6.2), healthy('polygon', 6.1), healthy('fred', 6.0), healthy('goldapi', 5.9)];
        const leg = await resolveLeg(sources, new Set(['cg_cnbc', 'cg_polygon']), NOW);
        expect(leg.source).toBe('fred');
        expect(leg.tried).toEqual(['cnbc:off', 'polygon:off', 'fred:ok']);
    });

    test('falls through on a thrown error', async () => {
        const leg = await resolveLeg([src('cnbc', 7, new Error('boom')), healthy('fred', 6.0)], new Set(), NOW);
        expect(leg.source).toBe('fred');
        expect(leg.tried).toContain('cnbc:err');
    });

    test('falls through on an empty (non-finite) price', async () => {
        const bad = src('cnbc', 7, { current: NaN, currentDate: iso(0), historyAsc: [] });
        const leg = await resolveLeg([bad, healthy('fred', 6.0)], new Set(), NOW);
        expect(leg.source).toBe('fred');
        expect(leg.tried).toContain('cnbc:empty');
    });

    test('rejects a stale source and falls through', async () => {
        const stale = src('cnbc', 7, { current: 6.2, currentDate: iso(40), historyAsc: [] }); // 40d old, deadline 7d
        const leg = await resolveLeg([stale, healthy('fred', 6.0)], new Set(), NOW);
        expect(leg.source).toBe('fred');
        expect(leg.tried.some((t) => t.startsWith('cnbc:stale'))).toBe(true);
    });

    test('monthly source is NOT rejected within its generous deadline', async () => {
        const monthly = src('fred', 80, { current: 6.0, currentDate: iso(36), historyAsc: hist([{ daysAgo: 36, price: 6.0 }]) });
        const leg = await resolveLeg([monthly], new Set(), NOW);
        expect(leg.source).toBe('fred'); // 36d old but 80d deadline → fresh
    });

    test('returns a null leg when every source fails', async () => {
        const leg = await resolveLeg([src('cnbc', 7, new Error('x')), src('fred', 7, new Error('y'))], new Set(), NOW);
        expect(leg.source).toBeNull();
        expect(leg.current).toBeNull();
        expect(leg.tried).toEqual(['cnbc:err', 'fred:err']);
    });
});

describe('buildCopperGold', () => {
    const copperLeg = {
        current: 6.0, currentDate: iso(0), source: 'cnbc',
        historyAsc: hist([{ daysAgo: 0, price: 6.0 }, { daysAgo: 30, price: 5.5 }, { daysAgo: 90, price: 5.0 }]),
        tried: ['cnbc'],
    };
    const goldFlat = {
        current: 4000, currentDate: iso(0), source: 'cnbc',
        historyAsc: hist([{ daysAgo: 0, price: 4000 }, { daysAgo: 30, price: 4000 }, { daysAgo: 90, price: 4000 }]),
        tried: ['cnbc'],
    };

    test('computes ratio + 1mo and 3mo change (rising)', () => {
        const cg = buildCopperGold(copperLeg, goldFlat);
        expect(cg.value).toBeCloseTo(1.5, 3);          // 6/4000*1000
        expect(cg.changePct).toBeCloseTo(9.09, 1);      // vs 5.5/4000 -> 1.375
        expect(cg.changePct3mo).toBeCloseTo(20.0, 1);   // vs 5.0/4000 -> 1.25
        expect(cg.status).toBe('rising');
        expect(cg.copperSource).toBe('cnbc');
        expect(cg.source).toContain('copper:cnbc');
    });

    test('status falling when the ratio dropped', () => {
        const dropping = { ...copperLeg, current: 4.5, historyAsc: hist([{ daysAgo: 0, price: 4.5 }, { daysAgo: 30, price: 5.5 }, { daysAgo: 90, price: 5.0 }]) };
        const cg = buildCopperGold(dropping, goldFlat);
        expect(cg.change).toBeLessThan(0);
        expect(cg.status).toBe('falling');
    });

    test('unavailable (N/A) when one leg is null', () => {
        const nullLeg = { current: null, currentDate: null, historyAsc: [], source: null, tried: ['cnbc:err'] };
        const cg = buildCopperGold(copperLeg, nullLeg);
        expect(cg.value).toBeNull();
        expect(cg.unavailable).toBe(true);
        expect(cg.status).toBe('unknown');
    });

    test('shows the ratio but null change when history is too short for the window', () => {
        const noHist = { current: 6.0, currentDate: iso(0), historyAsc: [], source: 'goldapi', tried: ['cnbc:err', 'fred:err', 'goldapi'] };
        const cg = buildCopperGold(noHist, { current: 4000, currentDate: iso(0), historyAsc: [], source: 'goldapi', tried: [] });
        expect(cg.value).toBeCloseTo(1.5, 3);
        expect(cg.change).toBeNull();
        expect(cg.change3mo).toBeNull();
        expect(cg.status).toBe('neutral');
    });

    test('asOf is the older of the two legs', () => {
        const olderCopper = { ...copperLeg, currentDate: iso(35) }; // monthly copper a bit old
        const cg = buildCopperGold(olderCopper, goldFlat);
        expect(cg.asOf).toBe(iso(35));
    });
});
