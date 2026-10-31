import { signalEpisodes, signalStudy, plusMonthsIso, comboSeries } from '../signalStudy';

// SPY: 100 on Jan 1 2020, +1 every day (a trading day every day keeps the math readable)
const spy = Array.from({ length: 500 }, (_, i) => ({ date: new Date(Date.UTC(2020, 0, 1 + i)).toISOString().slice(0, 10), price: 100 + i }));

describe('signalEpisodes', () => {
    it('counts a run of 20+ readings once, and again only after 3 quiet months', () => {
        const eps = signalEpisodes([
            { date: '2020-01-02', value: 21 },
            { date: '2020-01-09', value: 30 }, // same run
            { date: '2020-02-06', value: 5 },
            { date: '2020-04-02', value: 22 }, // 84 days after the last 20+: same episode
            { date: '2020-08-06', value: 25 }, // 126 days quiet: new episode
        ], 20);
        expect(eps).toEqual([{ date: '2020-01-02', value: 21 }, { date: '2020-08-06', value: 25 }]);
    });
});

describe('signalStudy', () => {
    const aaii = [{ date: '2019-06-01', value: 40 }, { date: '2020-01-02', value: 21 }, { date: '2021-03-01', value: 25 }];
    const s = signalStudy(aaii, spy, 6, 20);

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
        expect(signalStudy([], spy, 6, 20)).toBeNull();
    });
});

describe('plusMonthsIso', () => {
    it('stops at the month end instead of rolling into the next month', () => {
        expect(plusMonthsIso('2025-08-31', 6)).toBe('2026-02-28');
        expect(plusMonthsIso('2023-08-31', 6)).toBe('2024-02-29');
        expect(plusMonthsIso('2026-01-15', 6)).toBe('2026-07-15');
    });
});

describe('comboSeries', () => {
    const combo = { legs: [{ key: 'a', line: 20 }, { key: 'b', days: 365, pct: true, line: -15 }] };
    // leg a: weekly-ish readings; leg b: weekly through 2020 at 100, then daily in 2021 at 80 (oil's real spacing change)
    const a = [{ date: '2021-01-04', value: 25 }, { date: '2021-01-11', value: 10 }, { date: '2021-04-01', value: 30 }];
    const day = (y, m, d) => new Date(Date.UTC(y, m, d)).toISOString().slice(0, 10);
    const b = [...Array.from({ length: 53 }, (_, i) => ({ date: day(2020, 0, 1 + 7 * i), value: 100 })),
        ...Array.from({ length: 120 }, (_, i) => ({ date: day(2021, 0, 1 + i), value: 80 }))];
    const out = comboSeries(combo, a, b);
    const on = (d) => out.find((p) => p.date === d);

    it('takes a change against the reading a year earlier by date, never by row', () => {
        expect(out.every((p) => Math.abs(p.legs[1] + 20) < 1e-9)).toBe(true); // a row shift would compare daily 80 with daily 80
    });

    it('scores the weaker leg against its line, and starts when both legs have a reading', () => {
        expect(out[0]).toEqual({ date: '2021-01-04', value: 1.25, legs: [25, expect.any(Number)] });
        expect(on('2021-01-11').value).toBe(0.5);
        expect(on('2021-04-01').value).toBeCloseTo(4 / 3, 9);
    });

    it('drops days where a leg has not printed for 45+ days', () => {
        expect(on('2021-02-25')).toBeTruthy();
        expect(out.some((p) => p.date > '2021-02-25' && p.date < '2021-04-01')).toBe(false);
    });
});
