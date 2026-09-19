import { PAIRS, ratioSeries, pairStats } from '../breadth';

/**
 * Build an ascending daily history from [{daysAgo, price}].
 */
const hist = (points) =>
    points
        .map((p) => ({
            date: new Date(Date.now() - p.daysAgo * 86400000).toISOString().slice(0, 10),
            price: p.price,
        }))
        .sort((a, b) => (a.date < b.date ? -1 : 1));

describe('PAIRS', () => {
    it('defines the four expected pairs', () => {
        expect(PAIRS).toEqual({
            rspSpy: ['RSP', 'SPY'],
            iwmSpy: ['IWM', 'SPY'],
            xlkXlu: ['XLK', 'XLU'],
            hygLqd: ['HYG', 'LQD'],
        });
    });
});

describe('ratioSeries', () => {
    it('computes ratios on matching dates', () => {
        const a = hist([
            { daysAgo: 3, price: 200 },
            { daysAgo: 2, price: 210 },
            { daysAgo: 1, price: 220 },
        ]);
        const b = hist([
            { daysAgo: 3, price: 100 },
            { daysAgo: 2, price: 105 },
            { daysAgo: 1, price: 110 },
        ]);
        const ratios = ratioSeries(a, b);
        expect(ratios).toHaveLength(3);
        expect(ratios[0].ratio).toBeCloseTo(2.0, 4);
        expect(ratios[1].ratio).toBeCloseTo(2.0, 4);
        expect(ratios[2].ratio).toBeCloseTo(2.0, 4);
    });

    it('aligns on common dates when one series has extra dates', () => {
        // histA has an extra day that histB doesn't have
        const a = hist([
            { daysAgo: 4, price: 100 },
            { daysAgo: 3, price: 102 },
            { daysAgo: 2, price: 104 },
            { daysAgo: 1, price: 106 },
        ]);
        const b = hist([
            { daysAgo: 4, price: 50 },
            { daysAgo: 2, price: 52 },
            { daysAgo: 1, price: 53 },
        ]);
        const ratios = ratioSeries(a, b);
        expect(ratios).toHaveLength(3);
        // daysAgo=3 is missing from B → omitted
        expect(ratios[0].ratio).toBeCloseTo(2.0, 4); // daysAgo=4: 100/50
        expect(ratios[1].ratio).toBeCloseTo(2.0, 4); // daysAgo=2: 104/52
        expect(ratios[2].ratio).toBeCloseTo(106 / 53, 4); // daysAgo=1: 106/53
    });

    it('returns empty for empty arrays', () => {
        expect(ratioSeries([], [{ date: '2026-01-01', price: 100 }])).toEqual([]);
        expect(ratioSeries([{ date: '2026-01-01', price: 100 }], [])).toEqual([]);
        expect(ratioSeries(null, [{ date: '2026-01-01', price: 100 }])).toEqual([]);
        expect(ratioSeries([{ date: '2026-01-01', price: 100 }], undefined)).toEqual([]);
    });

    it('skips dates with zero or non-finite prices', () => {
        const a = [
            { date: '2026-01-01', price: 100 },
            { date: '2026-01-02', price: 0 },
            { date: '2026-01-03', price: 120 },
            { date: '2026-01-04', price: null },
        ];
        const b = [
            { date: '2026-01-01', price: 50 },
            { date: '2026-01-02', price: 55 },
            { date: '2026-01-03', price: 60 },
            { date: '2026-01-04', price: 65 },
        ];
        const ratios = ratioSeries(a, b);
        // Jan 2 (price=0 in A) and Jan 4 (null in A) are skipped; Jan 1 and Jan 3 survive
        expect(ratios).toHaveLength(2);
        expect(ratios[0].date).toBe('2026-01-01');
        expect(ratios[0].ratio).toBeCloseTo(2.0, 4);
        expect(ratios[1].date).toBe('2026-01-03');
        expect(ratios[1].ratio).toBeCloseTo(2.0, 4);
    });

    it('returns sorted ascending by date', () => {
        const a = hist([
            { daysAgo: 5, price: 50 },
            { daysAgo: 1, price: 90 },
            { daysAgo: 3, price: 70 },
        ]);
        const b = hist([
            { daysAgo: 5, price: 25 },
            { daysAgo: 1, price: 45 },
            { daysAgo: 3, price: 35 },
        ]);
        const ratios = ratioSeries(a, b);
        expect(ratios).toHaveLength(3);
        expect(ratios[0].ratio).toBeCloseTo(2.0, 4);
        expect(ratios[1].ratio).toBeCloseTo(2.0, 4);
        expect(ratios[2].ratio).toBeCloseTo(2.0, 4);
        // Check ascending date order
        for (let i = 1; i < ratios.length; i++) {
            expect(ratios[i].date >= ratios[i - 1].date).toBe(true);
        }
    });

    it('sorts ascending when given unsorted raw arrays (no helper sort)', () => {
        // Pass completely unsorted raw arrays — no helper sort, no ascending
        // guarantee. The function must sort before returning.
        const a = [
            { date: '2026-03-05', price: 90 },
            { date: '2026-03-01', price: 50 },
            { date: '2026-03-03', price: 70 },
        ];
        const b = [
            { date: '2026-03-03', price: 35 },
            { date: '2026-03-05', price: 45 },
            { date: '2026-03-01', price: 25 },
        ];
        const ratios = ratioSeries(a, b);
        expect(ratios).toHaveLength(3);
        expect(ratios[0].date).toBe('2026-03-01');
        expect(ratios[0].ratio).toBeCloseTo(2.0, 4);
        expect(ratios[1].date).toBe('2026-03-03');
        expect(ratios[1].ratio).toBeCloseTo(2.0, 4);
        expect(ratios[2].date).toBe('2026-03-05');
        expect(ratios[2].ratio).toBeCloseTo(2.0, 4);
        for (let i = 1; i < ratios.length; i++) {
            expect(ratios[i].date >= ratios[i - 1].date).toBe(true);
        }
    });
});

describe('pairStats', () => {
    it('returns nulls for an empty series', () => {
        const s = pairStats([]);
        expect(s.ratio).toBeNull();
        expect(s.chg20Pct).toBeNull();
        expect(s.chg60Pct).toBeNull();
        expect(s.vs50dPct).toBeNull();
        expect(s.asOf).toBeNull();
    });

    it('returns nulls for a too-short series (< 20 rows)', () => {
        const series = [];
        for (let i = 1; i <= 10; i++) {
            series.push({ date: `2026-01-${String(i).padStart(2, '0')}`, ratio: 1.0 });
        }
        const s = pairStats(series);
        expect(s.ratio).toBeCloseTo(1.0, 4);
        expect(s.chg20Pct).toBeNull();
        expect(s.chg60Pct).toBeNull();
        expect(s.vs50dPct).toBeNull();
        expect(s.asOf).toBe('2026-01-10');
    });

    it('returns nulls for chg60Pct when series is 21-60 rows', () => {
        const series = [];
        for (let i = 1; i <= 30; i++) {
            series.push({ date: `2026-01-${String(i).padStart(2, '0')}`, ratio: 1.0 });
        }
        const s = pairStats(series);
        expect(s.ratio).toBeCloseTo(1.0, 4);
        expect(s.chg20Pct).toBeCloseTo(0, 4);  // 21+ rows → computed
        expect(s.chg60Pct).toBeNull();          // < 61 rows → null
        expect(s.vs50dPct).toBeNull();           // < 50 rows → null
    });

    it('returns vs50dPct once series has 50+ rows', () => {
        const series = [];
        for (let i = 1; i <= 50; i++) {
            series.push({ date: `2026-01-${String(i).padStart(2, '0')}`, ratio: 1.0 });
        }
        const s = pairStats(series);
        expect(s.ratio).toBeCloseTo(1.0, 4);
        expect(s.chg20Pct).toBeCloseTo(0, 4);
        expect(s.chg60Pct).toBeNull();
        expect(s.vs50dPct).toBeCloseTo(0, 4);  // 50+ rows → computed, mean=1, ratio=1
    });

    it('computes correct values on a synthetic rising series', () => {
        // Build 65 entries where ratio linearly increases from 1.00 to 1.65
        const series = [];
        for (let i = 0; i < 65; i++) {
            const ratio = 1.0 + (i / 64) * 0.65; // from 1.0 to 1.65
            const day = String(i + 1).padStart(2, '0');
            series.push({ date: `2026-03-${day}`, ratio });
        }

        const s = pairStats(series);
        expect(s.asOf).toBe('2026-03-65');

        // ratio should be the last value: 1.0 + (64/64)*0.65 = 1.65
        expect(s.ratio).toBeCloseTo(1.65, 4);

        // chg20Pct: last ratio vs 20 rows ago (index 44, ratio = 1.0 + 44/64*0.65 = 1.446875)
        const ratio20Ago = 1.0 + (44 / 64) * 0.65;
        const expectedChg20 = ((1.65 / ratio20Ago) - 1) * 100;
        expect(s.chg20Pct).toBeCloseTo(expectedChg20, 4);

        // chg60Pct: last ratio vs 60 rows ago (index 4, ratio = 1.0 + 4/64*0.65 = 1.040625)
        const ratio60Ago = 1.0 + (4 / 64) * 0.65;
        const expectedChg60 = ((1.65 / ratio60Ago) - 1) * 100;
        expect(s.chg60Pct).toBeCloseTo(expectedChg60, 4);

        // vs50dPct: last ratio vs mean of last 50 ratios (indices 15..64)
        const last50 = series.slice(-50);
        const mean50 = last50.reduce((sum, r) => sum + r.ratio, 0) / 50;
        const expectedVs50 = ((1.65 / mean50) - 1) * 100;
        expect(s.vs50dPct).toBeCloseTo(expectedVs50, 4);
    });

    it('computes zero change when ratio is constant', () => {
        const series = [];
        for (let i = 0; i < 70; i++) {
            series.push({ date: `2026-03-${String(i + 1).padStart(2, '0')}`, ratio: 2.0 });
        }
        const s = pairStats(series);
        expect(s.ratio).toBeCloseTo(2.0, 4);
        expect(s.chg20Pct).toBeCloseTo(0, 4);
        expect(s.chg60Pct).toBeCloseTo(0, 4);
        expect(s.vs50dPct).toBeCloseTo(0, 4);
    });

    it('handles a declining ratio', () => {
        const series = [];
        for (let i = 0; i < 70; i++) {
            const ratio = 1.5 - (i / 69) * 0.4; // from 1.5 down to 1.1
            series.push({ date: `2026-04-${String(i + 1).padStart(2, '0')}`, ratio });
        }
        const s = pairStats(series);
        expect(s.ratio).toBeCloseTo(1.1, 4);
        expect(s.chg20Pct).toBeLessThan(0); // declining → negative
        expect(s.chg60Pct).toBeLessThan(0);
        expect(s.vs50dPct).toBeLessThan(0);
    });
});