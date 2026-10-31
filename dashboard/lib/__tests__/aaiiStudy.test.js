import { aaiiEpisodes, aaiiStudy } from '../aaiiStudy';

// SPY: 100 on Jan 1 2020, +1 every day (a trading day every day keeps the math readable)
const spy = Array.from({ length: 500 }, (_, i) => ({ date: new Date(Date.UTC(2020, 0, 1 + i)).toISOString().slice(0, 10), price: 100 + i }));

describe('aaiiEpisodes', () => {
    it('counts a run of 20+ readings once, and again only after 3 quiet months', () => {
        const eps = aaiiEpisodes([
            { date: '2020-01-02', value: 21 },
            { date: '2020-01-09', value: 30 }, // same run
            { date: '2020-02-06', value: 5 },
            { date: '2020-04-02', value: 22 }, // 84 days after the last 20+: same episode
            { date: '2020-08-06', value: 25 }, // 126 days quiet: new episode
        ]);
        expect(eps).toEqual([{ date: '2020-01-02', value: 21 }, { date: '2020-08-06', value: 25 }]);
    });
});

describe('aaiiStudy', () => {
    const aaii = [{ date: '2019-06-01', value: 40 }, { date: '2020-01-02', value: 21 }, { date: '2021-03-01', value: 25 }];
    const s = aaiiStudy(aaii, spy, 6);

    it('skips signals before SPY starts and measures SPY from the signal day to 6 months on', () => {
        expect(s.episodes.map((e) => e.date)).toEqual(['2020-01-02', '2021-03-01']);
        const [done] = s.episodes;
        expect(done.live).toBe(false);
        // Jan 2 (101) → Jul 2 (283)
        expect(done.ret).toBeCloseTo(283 / 101 - 1, 6);
        expect(done.path[0]).toEqual({ x: 0, r: 0 });
        expect(done.path[done.path.length - 1].x).toBeCloseTo(1, 6);
    });

    it('keeps a signal whose window has not ended as live, out of the stats', () => {
        const live = s.episodes[1];
        expect(live.live).toBe(true);
        expect(s.stats.n).toBe(1);
        expect(s.stats.win).toBe(1);
        expect(s.avgPath[s.avgPath.length - 1].r).toBeCloseTo(s.episodes[0].ret, 6);
    });

    it('compares against the same window started on every day', () => {
        // every 6-month window that fits: first one Jan 1 → Jul 1 (100 → 282)
        expect(s.base.n).toBeGreaterThan(300);
        expect(s.base.win).toBe(1);
        expect(s.base.avg).toBeGreaterThan(0);
    });

    it('returns null without data', () => {
        expect(aaiiStudy([], spy, 6)).toBeNull();
    });
});
