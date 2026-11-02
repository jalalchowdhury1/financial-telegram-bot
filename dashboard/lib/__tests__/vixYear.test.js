import { vixYearGeo, vixZone, moveDir } from '../vixYear';
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
            expect([g.move, g.ink]).toEqual([null, null]);
        }
    });

    it("SPY days that don't line up with VIX's give no real line, not a stub", () => {
        const old = spyHistory.map((p, i) => ({ ...p, date: weekdays('2020-06-30', 300)[i] }));
        expect(vixYearGeo({ year, dates, t, spyHistory: old }).real).toBeNull();
        expect(vixYearGeo({ year, dates, t, spyHistory: old }).move).toBeNull();
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

    it('which way SPY went over the same 21 sessions inks the dotted line: up, down, or flat within ±1%', () => {
        // SPY climbs, rolls over, slides, then goes nowhere
        const climb = weekdays('2026-10-30', 300).map((date, i) => ({ date, price: i < 120 ? 500 + 2 * i : i < 200 ? 738 - 3 * (i - 119) : 498 }));
        const g = vixYearGeo({ year, dates, t, spyHistory: climb });
        expect(g.move[0]).toBeCloseTo(climb[48].price / climb[27].price - 1, 12); // row 0 is SPY day 48, 21 sessions after day 27
        expect(g.move.at(-1)).toBe(0);
        expect(g.ink.map((r) => r.dir)).toEqual(['up', 'flat', 'down', 'flat']);
        expect([g.ink[0].from, g.ink.at(-1).to]).toEqual([0, 100]);
        g.ink.slice(1).forEach((r, k) => expect(r.from).toBe(g.ink[k].to)); // hard switches, no gaps
        g.move.forEach((m, i) => expect(g.ink.find((r) => r.from <= g.x(i) && g.x(i) <= r.to).dir).toBe(moveDir(m)));
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
    expect(g.move[i]).toBeCloseTo(g.move[i - 1], 12);
    const behind = vixYearGeo({ year, dates, t, spyHistory: spyHistory.filter((p) => p.date <= '2026-10-23') });
    expect(behind.real.at(-1)).toBeNull(); // Oct 30 vs a bar from Oct 23
    expect(behind.move.at(-1)).toBeNull();
    expect(behind.real.at(-6)).not.toBeNull(); // Oct 23 itself
});

it('moveDir: a month within ±1% is flat', () => {
    expect([0.0101, 0.01, 0, -0.01, -0.0101, null, NaN].map(moveDir)).toEqual(['up', 'flat', 'flat', 'flat', 'down', 'flat', 'flat']);
});
