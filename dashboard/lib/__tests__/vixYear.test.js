import { vixYearGeo, vixZone } from '../vixYear';
import { realizedVol } from '../vol';

// Weekdays (holidays aside) ending on `end`, oldest first.
const weekdays = (end, n) => {
    const out = [];
    for (let d = new Date(`${end}T12:00:00Z`); out.length < n; d = new Date(d - 864e5)) if (d.getUTCDay() % 6) out.unshift(d.toISOString().slice(0, 10));
    return out;
};
const dates = weekdays('2026-10-30', 253);
const year = dates.map((_, i) => (i === 100 ? 31.2 : i === 252 ? 19.42 : 14 + (i % 10) / 2));
const spyHistory = weekdays('2026-10-30', 300).map((date, i) => ({ date, price: 600 + (i % 2 ? 3 : -3) + i * 0.1 }));
const t = { iv: 19.42, ivPctile1y: 77.8, live: false, asOf: '2026-10-30' };

describe('vixYearGeo', () => {
    it("dates each close and lines SPY's real moves up by day", () => {
        const g = vixYearGeo({ year, dates, t, spyHistory });
        expect(g.n).toBe(252);
        expect(g.rows.at(-1)).toEqual({ v: 19.42, d: '2026-10-30' });
        expect(g.real.at(-1)).toBeCloseTo(realizedVol(spyHistory.map((p) => p.price).slice(-22)), 9); // the table's RV21 formula
        expect(g.vix[g.peak]).toBe(31.2);
        expect(g.rows[g.peak].d).toBe(dates[100]);
        expect(g.zones.map((z) => z.name)).toEqual(['calm', 'normal', 'nervous', 'fear']);
        const nervous = g.zones.find((z) => z.name === 'nervous');
        expect([nervous.top, nervous.bottom]).toEqual([g.y(30), g.y(20)]);
    });

    it('an undated copy (older saved payload, or dates that do not match) draws VIX alone', () => {
        for (const d of [undefined, dates.slice(1)]) {
            const g = vixYearGeo({ year, dates: d, t, spyHistory });
            expect(g.rows.every((r) => r.d === null)).toBe(true);
            expect(g.real).toBeNull();
            expect(g.realLine).toBeNull();
        }
    });

    it("SPY days that don't line up with VIX's give no real line, not a stub", () => {
        const old = spyHistory.map((p, i) => ({ ...p, date: weekdays('2020-06-30', 300)[i] }));
        expect(vixYearGeo({ year, dates, t, spyHistory: old }).real).toBeNull();
        expect(vixYearGeo({ year, dates, t, spyHistory: null }).real).toBeNull();
    });

    it("a live quote adds today, or replaces its own day's close", () => {
        const next = vixYearGeo({ year, dates, t: { ...t, live: true, iv: 22, asOf: '2026-11-02' }, spyHistory });
        expect(next.n).toBe(253);
        expect(next.rows.at(-1)).toEqual({ v: 22, d: '2026-11-02' });
        const same = vixYearGeo({ year, dates, t: { ...t, live: true, iv: 22, asOf: '2026-10-30' }, spyHistory });
        expect(same.n).toBe(252);
        expect(same.rows.at(-1)).toEqual({ v: 22, d: '2026-10-30' });
        expect(same.rows.at(-2).d).toBe('2026-10-29');
    });

    it('the scale always spans 15–20, so a calm year still shows where nervous starts', () => {
        const g = vixYearGeo({ year: year.map(() => 13), dates, t, spyHistory: null });
        expect([g.lo, g.hi]).toEqual([13, 20]);
        expect(g.zones.map((z) => z.name)).toEqual(['calm', 'normal']);
    });

    it('too few closes: nothing to draw', () => {
        expect(vixYearGeo({ year: year.slice(-19), dates: dates.slice(-19), t, spyHistory })).toBeNull();
        expect(vixYearGeo({ year: null, t, spyHistory })).toBeNull();
    });
});

it('vixZone reads VIX the usual way', () => {
    expect([14.9, 15, 19.99, 20, 29.9, 30, 80].map((v) => vixZone(v).name)).toEqual(['calm', 'normal', 'normal', 'nervous', 'nervous', 'fear', 'fear']);
});

it("a VIX day SPY did not trade (CBOE's file has NYSE holidays) keeps the last move; a bar over 4 days old is not that day's", () => {
    const spyDays = spyHistory.filter((p) => p.date !== '2026-10-29'); // pretend Oct 29 was a holiday
    const g = vixYearGeo({ year, dates, t, spyHistory: spyDays });
    const i = g.rows.findIndex((r) => r.d === '2026-10-29');
    expect(g.real[i]).toBeCloseTo(g.real[i - 1], 9);
    const behind = vixYearGeo({ year, dates, t, spyHistory: spyHistory.filter((p) => p.date <= '2026-10-23') });
    expect(behind.real.at(-1)).toBeNull(); // Oct 30 vs a bar from Oct 23
    expect(behind.real.at(-6)).not.toBeNull(); // Oct 23 itself
});
