/**
 * 🧬 Factor row math + cascade (lib/factors.js) and its KV store (lib/factorStore.js).
 */
import {
    windowStart, weeklyToFriday, thinToWeekly, validSeries, mergeSeries, align,
    computeWindow, resolveTicker, buildPayload, isGoodPayload, isStorablePayload,
    FACTORS, TICKERS, WINDOWS, daysBetween, todayET,
} from '../factors';
import { loadFactorsKV, saveFactorsKV, KV_KEY } from '../factorStore';
import baked from '../data/factorsBaked.json';

const DAY = 864e5;
const iso = (ms) => new Date(ms).toISOString().slice(0, 10);

/** Weekday series from `startIso` to `endIso` inclusive; price(i) supplies closes. */
function weekdays(startIso, endIso, price = (i) => 100 + i * 0.1) {
    const out = [];
    let i = 0;
    for (let t = Date.parse(`${startIso}T00:00:00Z`); t <= Date.parse(`${endIso}T00:00:00Z`); t += DAY) {
        const dow = new Date(t).getUTCDay();
        if (dow === 0 || dow === 6) continue;
        out.push({ date: iso(t), price: price(i++) });
    }
    return out;
}

describe('windowStart', () => {
    it('subtracts calendar months and clamps the day', () => {
        expect(windowStart('2026-03-31', '1M')).toBe('2026-02-28');
        expect(windowStart('2024-03-31', '1M')).toBe('2024-02-29');
        expect(windowStart('2026-09-25', '6M')).toBe('2026-03-25');
        expect(windowStart('2026-09-25', '1Y')).toBe('2025-09-25');
        expect(windowStart('2026-09-25', '10Y')).toBe('2016-09-25');
        expect(windowStart('2026-01-15', '3M')).toBe('2025-10-15');
    });
    it('YTD starts at the prior year-end', () => {
        expect(windowStart('2026-09-25', 'YTD')).toBe('2025-12-31');
    });
    it('unknown window → null', () => {
        expect(windowStart('2026-09-25', '7Y')).toBeNull();
    });
});

describe('weeklyToFriday', () => {
    it('shifts CNBC Sunday-dated bars to their Friday and drops any bar from the last 7 days', () => {
        const h = [
            { date: '2026-09-06', price: 1 },
            { date: '2026-09-13', price: 2 },
            { date: '2026-09-20', price: 3 }, // Friday = 09-25: unreliable snapshot, even on Saturday
        ];
        expect(weeklyToFriday(h, '2026-09-26')).toEqual([
            { date: '2026-09-11', price: 1 },
            { date: '2026-09-18', price: 2 },
        ]);
        expect(weeklyToFriday(h, '2026-10-02')).toHaveLength(3);
    });
});

describe('thinToWeekly', () => {
    it('keeps the last trading day of each week', () => {
        const h = weekdays('2026-09-07', '2026-09-18');
        const w = thinToWeekly(h);
        expect(w.map((p) => p.date)).toEqual(['2026-09-11', '2026-09-18']);
    });
});

describe('validSeries', () => {
    const good = weekdays('2026-01-01', '2026-03-01');
    it('accepts a clean ascending series', () => expect(validSeries(good)).toBe(true));
    it('rejects too few points', () => expect(validSeries(good.slice(0, 5))).toBe(false));
    it('rejects a split-sized cliff', () => {
        const bad = good.map((p, i) => (i > 20 ? { ...p, price: p.price / 2 } : p));
        expect(validSeries(bad)).toBe(false);
    });
    it('rejects unsorted / duplicate dates and junk prices', () => {
        expect(validSeries([...good].reverse())).toBe(false);
        expect(validSeries(good.map((p, i) => (i === 3 ? { ...p, price: NaN } : p)))).toBe(false);
        expect(validSeries(null)).toBe(false);
    });
});

describe('mergeSeries / align', () => {
    it('uses only the long history OLDER than the recent series', () => {
        const long = [{ date: '2020-01-03', price: 1 }, { date: '2024-01-05', price: 2 }, { date: '2025-01-03', price: 99 }];
        const recent = [{ date: '2024-06-03', price: 3 }, { date: '2025-01-03', price: 4 }];
        expect(mergeSeries(long, recent)).toEqual([
            { date: '2020-01-03', price: 1 }, { date: '2024-01-05', price: 2 },
            { date: '2024-06-03', price: 3 }, { date: '2025-01-03', price: 4 },
        ]);
        expect(mergeSeries(null, recent)).toEqual(recent);
        expect(mergeSeries(long, [])).toEqual(long);
    });
    it('aligns on common dates only', () => {
        const f = [{ date: 'a', price: 1 }, { date: 'b', price: 2 }];
        const b = [{ date: 'b', price: 4 }, { date: 'c', price: 5 }];
        expect(align(f, b)).toEqual([{ date: 'b', f: 2, b: 4 }]);
    });
});

describe('computeWindow', () => {
    // Factor doubles, SPY +50% over the window: $1 in the factor ends 33.3% ahead of $1 in SPY.
    const series = [
        { date: '2026-03-20', f: 10, b: 100 },
        { date: '2026-03-25', f: 10, b: 100 },
        { date: '2026-06-25', f: 15, b: 120 },
        { date: '2026-09-25', f: 20, b: 150 },
    ];
    it('computes relative, factor and benchmark returns from the last bar on/before the start', () => {
        const w = computeWindow(series, '6M');
        expect(w.from).toBe('2026-03-25');
        expect(w.to).toBe('2026-09-25');
        expect(w.f).toBe(100);
        expect(w.b).toBe(50);
        expect(w.rel).toBeCloseTo(33.33, 2);
        expect(w.spark[0]).toBe(0);
        expect(w.spark[w.spark.length - 1]).toBe(w.rel);
    });
    it('returns null when history does not reach the window start', () => {
        expect(computeWindow(series, '1Y')).toBeNull();
    });
    it('allows a small offset between nominal start and first bar (weekly bars)', () => {
        const s = [{ date: '2016-09-30', f: 1, b: 1 }, { date: '2020-01-03', f: 1, b: 1 }, { date: '2026-09-25', f: 2, b: 1 }];
        expect(computeWindow(s, '10Y')?.rel).toBe(100);
    });
    it('resamples long windows by TIME to 60 points, ending exactly on the headline number', () => {
        const f = weekdays('2023-01-02', '2026-09-25', (i) => 50 + i * 0.05);
        const b = weekdays('2023-01-02', '2026-09-25', (i) => 100 + i * 0.03);
        const w = computeWindow(align(f, b), '3Y');
        expect(w.spark).toHaveLength(60);
        expect(w.spark[59]).toBe(w.rel);
    });
    it('short windows keep every bar', () => {
        const f = weekdays('2026-08-01', '2026-09-25');
        const w = computeWindow(align(f, f), '1M');
        expect(w.spark.length).toBeGreaterThan(15);
        expect(w.rel).toBe(0);
    });
});

describe('resolveTicker cascade', () => {
    const today = '2026-09-26';
    const fresh = weekdays('2024-09-27', '2026-09-25');
    const stale = weekdays('2024-09-01', '2026-09-10');
    const tenYears = weekdays('2016-09-26', '2026-09-25');
    const weekly = thinToWeekly(weekdays('2016-09-26', '2026-09-18'));
    const ok = (h) => jest.fn().mockResolvedValue(h);
    const fail = () => jest.fn().mockRejectedValue(new Error('boom'));

    it('primary fresh daily + primary weekly: stops at tier 1', async () => {
        const second = ok(fresh);
        const r = await resolveTicker('SPY', {
            recent: [{ name: 'cnbc', fn: ok(fresh) }, { name: 'nasdaq', fn: second }],
            long: [{ name: 'cnbc-weekly', fn: ok(weekly) }],
            today,
        });
        expect(second).not.toHaveBeenCalled();
        expect(r.recentSource).toBe('cnbc');
        expect(r.longSource).toBe('cnbc-weekly');
        expect(r.stale).toBe(false);
        expect(r.history[0].date < '2017-01-01').toBe(true);
        expect(r.asOf).toBe('2026-09-25');
    });

    it('a STALE primary is replaced by a fresher later tier', async () => {
        const r = await resolveTicker('SPY', {
            recent: [{ name: 'cnbc', fn: ok(stale) }, { name: 'nasdaq', fn: ok(fresh) }],
            long: [{ name: 'cnbc-weekly', fn: ok(weekly) }],
            today,
        });
        expect(r.recentSource).toBe('nasdaq');
        expect(r.tried).toContain('cnbc:stale');
        expect(r.stale).toBe(false);
    });

    it('keeps the freshest stale candidate when nothing fresh answers', async () => {
        const r = await resolveTicker('SPY', {
            recent: [{ name: 'cnbc', fn: ok(stale) }, { name: 'nasdaq', fn: fail() }],
            long: [],
            today,
        });
        expect(r.recentSource).toBe('cnbc');
        expect(r.stale).toBe(true);
    });

    it('a 10y daily recent tier needs no long history', async () => {
        const longFn = ok(weekly);
        const r = await resolveTicker('SPY', {
            recent: [{ name: 'cnbc', fn: fail() }, { name: 'nasdaq', fn: ok(tenYears) }],
            long: [{ name: 'cnbc-weekly', fn: longFn }],
            today,
        });
        expect(longFn).not.toHaveBeenCalled();
        expect(r.recentSource).toBe('nasdaq');
        expect(r.longSource).toBeNull();
    });

    it('invalid data (split cliff) falls through to the next tier', async () => {
        const cliff = fresh.map((p, i) => (i > 100 ? { ...p, price: p.price * 3 } : p));
        const r = await resolveTicker('SPY', {
            recent: [{ name: 'cnbc', fn: ok(cliff) }, { name: 'polygon', fn: ok(fresh) }],
            long: [],
            today,
        });
        expect(r.tried).toContain('cnbc:invalid');
        expect(r.recentSource).toBe('polygon');
    });

    it('every live tier down → the baked weekly floor, flagged stale', async () => {
        const r = await resolveTicker('SPY', {
            recent: [{ name: 'cnbc', fn: fail() }],
            long: [{ name: 'cnbc-weekly', fn: fail() }],
            baked: () => weekdays('2016-09-26', '2026-06-30').filter((_, i) => i % 5 === 0),
            today,
        });
        expect(r.recentSource).toBe('baked(only)');
        expect(r.longSource).toBe('baked');
        expect(r.stale).toBe(true);
        expect(r.history.length).toBeGreaterThan(400);
    });

    it('skips remaining tiers once past the deadline', async () => {
        const later = ok(fresh);
        const r = await resolveTicker('SPY', {
            recent: [{ name: 'cnbc', fn: ok(fresh) }, { name: 'nasdaq', fn: later }],
            long: [],
            today,
            deadline: Date.now() - 1,
        });
        expect(later).not.toHaveBeenCalled();
        expect(r.tried).toContain('cnbc:deadline');
        expect(r.history).toEqual([]);
    });
});

describe('buildPayload', () => {
    const today = '2026-09-26';
    const mk = (ticker, drift, src = 'cnbc', long = 'cnbc-weekly') => ({
        ticker,
        history: weekdays('2016-09-26', '2026-09-25', (i) => 100 * (1 + drift) ** i),
        asOf: '2026-09-25',
        stale: false,
        recentSource: src,
        longSource: long,
        tried: [],
    });
    const all = () => Object.fromEntries(TICKERS.map((t, i) => [t, mk(t, 0.0002 * (i + 1))]));

    it('builds all 5 factors × 8 windows with the primary path flagged clean', () => {
        const p = buildPayload(all());
        expect(p.factors.map((f) => f.ticker)).toEqual(FACTORS.map((f) => f.ticker));
        for (const f of p.factors) for (const w of WINDOWS) expect(f.windows[w]).not.toBeNull();
        expect(p._meta.fallback).toBe(false);
        expect(p._meta.hasErrors).toBe(false);
        expect(p.basis).toBe('price');
        expect(isGoodPayload(p)).toBe(true);
        expect(isStorablePayload(p)).toBe(true);
        // Faster-drifting factor than SPY → positive relative return.
        expect(p.factors[0].windows['1Y'].rel).toBeGreaterThan(0);
    });

    it('a missing factor is reported, the rest still ship', () => {
        const r = all();
        r.MTUM = { ticker: 'MTUM', history: [], tried: ['cnbc:err', 'nasdaq:err'] };
        const p = buildPayload(r);
        expect(p.factors).toHaveLength(4);
        expect(p._meta.hasErrors).toBe(true);
        expect(p._meta.messages.join(' ')).toMatch(/MTUM: unavailable \(cnbc:err, nasdaq:err\)/);
    });

    it('a fallback tier anywhere flips _meta.fallback (health check: primary must have run)', () => {
        const r = all();
        r.QUAL = mk('QUAL', 0.0001, 'nasdaq', null);
        expect(buildPayload(r)._meta.fallback).toBe(true);
    });

    it('stale benchmark → stale payload that is served but never stored', () => {
        const r = all();
        r.SPY = { ...r.SPY, stale: true };
        const p = buildPayload(r);
        expect(p._meta.stale).toBe(true);
        expect(isGoodPayload(p)).toBe(true);
        expect(isStorablePayload(p)).toBe(false);
    });

    it('all-baked is not "good" (a warmer cache should win), and no benchmark throws', () => {
        const r = all();
        for (const t of TICKERS) r[t] = { ...r[t], recentSource: 'baked(only)', longSource: 'baked' };
        expect(isGoodPayload(buildPayload(r))).toBe(false);
        expect(() => buildPayload({ VLUE: r.VLUE })).toThrow(/benchmark/);
    });

    it('the committed bake is complete and renders every window', () => {
        expect(Object.keys(baked.tickers).sort()).toEqual([...TICKERS].sort());
        const resolved = {};
        for (const t of TICKERS) {
            const h = baked.tickers[t].map(([date, price]) => ({ date, price }));
            expect(validSeries(h, 400)).toBe(true);
            resolved[t] = { ticker: t, history: h, asOf: h[h.length - 1].date, stale: false, recentSource: 'baked(only)', longSource: 'baked', tried: [] };
        }
        const p = buildPayload(resolved);
        expect(p.factors).toHaveLength(5);
        for (const f of p.factors) expect(f.windows['10Y']).not.toBeNull();
    });
});

describe('helpers', () => {
    it('daysBetween / todayET', () => {
        expect(daysBetween('2026-09-25', '2026-09-30')).toBe(5);
        // 03:00 UTC on the 27th is still the 26th in New York.
        expect(todayET(new Date('2026-09-27T03:00:00Z'))).toBe('2026-09-26');
    });
});

describe('factorStore (KV last-good)', () => {
    const payload = { asOf: '2026-09-25', factors: [{}, {}, {}], _meta: { source: 'SPY:cnbc', stale: false, messages: ['m'] } };

    it('load relabels provenance so a cached copy never reads as live', async () => {
        const kv = { get: jest.fn().mockResolvedValue(JSON.stringify({ data: payload, savedAt: new Date().toISOString() })) };
        const got = await loadFactorsKV({ kv });
        expect(kv.get).toHaveBeenCalledWith(KV_KEY);
        expect(got._meta.source).toMatch(/^KV last-good .* ← SPY:cnbc$/);
        expect(got._meta.messages).toEqual(['cached: m']);
    });

    it('load ignores copies older than 30 days, junk, and KV errors', async () => {
        const old = { get: async () => ({ data: payload, savedAt: '2026-01-01T00:00:00Z' }) };
        expect(await loadFactorsKV({ kv: old, now: Date.parse('2026-09-26') })).toBeNull();
        expect(await loadFactorsKV({ kv: { get: async () => 'not json' } })).toBeNull();
        expect(await loadFactorsKV({ kv: { get: async () => { throw new Error('x'); } } })).toBeNull();
    });

    it('save writes once per asOf per 6h, never a stale payload', async () => {
        let mark = null;
        const tmp = { load: () => mark, save: (_k, d) => { mark = { data: d }; } };
        const kv = { set: jest.fn().mockResolvedValue(true) };
        const now = Date.parse('2026-09-26T12:00:00Z');
        expect(await saveFactorsKV(payload, { kv, tmp, now })).toBe(true);
        expect(await saveFactorsKV(payload, { kv, tmp, now: now + 3600e3 })).toBe(false);
        expect(await saveFactorsKV(payload, { kv, tmp, now: now + 7 * 3600e3 })).toBe(true);
        expect(await saveFactorsKV({ ...payload, asOf: '2026-09-28' }, { kv, tmp, now: now + 7.5 * 3600e3 })).toBe(true);
        expect(await saveFactorsKV({ ...payload, _meta: { stale: true } }, { kv, tmp, now })).toBe(false);
        expect(kv.set).toHaveBeenCalledTimes(3);
    });
});
