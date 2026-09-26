import {
    CURVE_KV_KEY,
    CURVE_MAX_AGE_MS,
    buildRegime,
    buildTermStructure,
    curveDegraded,
    curveState,
    expectedMovePct,
    leveragedDecayPct,
    loadCurveKV,
    saveCurveKV,
    staleCurve,
} from '../volRegime';

// CBOE closes on 2026-09-25 (verified against the CBOE CDN CSVs on 2026-09-26).
const eod = {
    VIX9D: { value: 12.76, date: '2026-09-25', source: 'cboe' },
    VIX: { value: 14.87, date: '2026-09-25', source: 'cboe' },
    VIX3M: { value: 17.93, date: '2026-09-25', source: 'cboe' },
    VIX6M: { value: 20.01, date: '2026-09-25', source: 'cboe' },
};

const fakeKv = () => {
    const store = new Map();
    return {
        store,
        get: jest.fn(async (k) => (store.has(k) ? store.get(k) : null)),
        set: jest.fn(async (k, v) => { store.set(k, v); return true; }),
    };
};

describe('curveState', () => {
    it('splits at 0.90 and 1.00', () => {
        expect(curveState(0.829)).toBe('calm');
        expect(curveState(0.899)).toBe('calm');
        expect(curveState(0.9)).toBe('watch');
        expect(curveState(0.999)).toBe('watch');
        expect(curveState(1.0)).toBe('stress');
        expect(curveState(1.33)).toBe('stress');
    });
    it('makes no call on junk', () => {
        for (const v of [null, undefined, NaN, 0, -1, Infinity]) expect(curveState(v)).toBeNull();
    });
});

describe('buildTermStructure', () => {
    it('builds four CBOE points, ratio and a calm call', () => {
        const c = buildTermStructure(eod, {});
        expect(c.points.map((p) => p.tenor)).toEqual(['9D', '1M', '3M', '6M']);
        expect(c.points.map((p) => p.value)).toEqual([12.76, 14.87, 17.93, 20.01]);
        expect(c.ratio).toBe(0.829);
        expect(c.state).toBe('calm');
        expect(c.frontInverted).toBe(false);
        expect(c.complete).toBe(true);
        expect(c.live).toBe(false);
        expect(c.asOf).toBe('2026-09-25');
        expect(curveDegraded(c)).toBe(false);
    });

    it('goes live only when BOTH VIX and VIX3M have a newer quote', () => {
        const q = (value) => ({ value, date: '2026-09-28', lastTime: '2026-09-28T11:00:00.000-0400' });
        const onlyVix = buildTermStructure(eod, { VIX: q(19) });
        expect(onlyVix.live).toBe(false);
        expect(onlyVix.ratio).toBe(0.829); // never intraday VIX ÷ yesterday's VIX3M

        const both = buildTermStructure(eod, { VIX: q(19), VIX3M: q(18), VIX9D: q(21) });
        expect(both.live).toBe(true);
        expect(both.ratio).toBe(1.056);
        expect(both.state).toBe('stress');
        expect(both.frontInverted).toBe(true);
        const nine = both.points.find((p) => p.index === 'VIX9D');
        expect(nine).toMatchObject({ value: 21, source: 'cboe+live', live: true, asOf: '2026-09-28' });
        // VIX6M had no quote: keeps its close, so the curve's asOf is the OLDEST point
        expect(both.points.find((p) => p.index === 'VIX6M').live).toBe(false);
        expect(both.asOf).toBe('2026-09-25');
    });

    it('ignores quotes that are not newer, not ISO, or not positive', () => {
        const bad = [
            { value: 19, date: '2026-09-25' },
            { value: 19, date: '09/28/2026' },
            { value: 0, date: '2026-09-28' },
            { value: NaN, date: '2026-09-28' },
        ];
        for (const b of bad) expect(buildTermStructure(eod, { VIX: b, VIX3M: b }).live).toBe(false);
    });

    it('uses a bare quote as the last tier when a point has no close', () => {
        const c = buildTermStructure({ ...eod, VIX3M: null }, { VIX3M: { value: 18, date: '2026-09-25' } });
        expect(c.points.find((p) => p.index === 'VIX3M')).toMatchObject({ value: 18, source: 'cnbc-quote', live: false });
        expect(c.state).toBe('calm');
        expect(curveDegraded(c)).toBe(true); // a quote-only point is a backup, not CBOE
    });

    it('makes no call without VIX3M, and is incomplete with a missing tenor', () => {
        const noV3 = buildTermStructure({ ...eod, VIX3M: null }, {});
        expect(noV3.state).toBeNull();
        expect(noV3.ratio).toBeNull();
        expect(noV3.points).toHaveLength(3);
        const no9 = buildTermStructure({ ...eod, VIX9D: null }, {});
        expect(no9.state).toBe('calm');
        expect(no9.complete).toBe(false);
        expect(no9.frontInverted).toBeNull();
        expect(curveDegraded(no9)).toBe(true);
    });

    it('never throws on empty input', () => {
        expect(buildTermStructure().state).toBeNull();
        expect(buildTermStructure(null, null).points).toEqual([]);
    });
});

describe('curveDegraded', () => {
    it('flags a non-CBOE point, a stale copy, and no curve', () => {
        const c = buildTermStructure(eod, {});
        expect(curveDegraded({ ...c, points: c.points.map((p, i) => (i ? p : { ...p, source: 'cnbc' })) })).toBe(true);
        expect(curveDegraded({ ...c, stale: true })).toBe(true);
        expect(curveDegraded(null)).toBe(true);
    });
});

describe('decay and expected move', () => {
    it('TQQQ decay = 1 − e^(−3σ²)', () => {
        expect(leveragedDecayPct(15.8)).toBeCloseTo(7.22, 2);
        expect(leveragedDecayPct(20.9)).toBeCloseTo(12.28, 2);
        expect(leveragedDecayPct(20, 2)).toBeCloseTo((1 - Math.exp(-0.04)) * 100, 6); // (4−2)/2 = 1
        expect(leveragedDecayPct(null)).toBeNull();
        expect(leveragedDecayPct(0)).toBeNull();
    });
    it('±1σ over 5 trading days = IV × √(5/252)', () => {
        expect(expectedMovePct(14.9)).toBeCloseTo(2.1, 2);
        expect(expectedMovePct(20.9)).toBeCloseTo(2.94, 2);
        expect(expectedMovePct(-3)).toBeNull();
    });
});

describe('buildRegime', () => {
    const tickers = [
        { ticker: 'SPY', iv: 14.87, rv21: 10 },
        { ticker: 'QQQ', iv: 20.9, rv21: 15.8 },
        { ticker: 'TQQQ', iv: 62.7, rv21: 47 },
    ];
    it('reads QQQ for decay and SPY/QQQ IV for moves', () => {
        const r = buildRegime(tickers, { state: 'calm' });
        expect(r.curve).toEqual({ state: 'calm' });
        expect(r.decay).toMatchObject({ leverage: 3, realizedVol: 15.8, impliedVol: 20.9, realized: 7.22, implied: 12.28 });
        expect(r.moves).toEqual({ days: 5, SPY: 2.09, QQQ: 2.94 });
    });
    it('nulls only the missing numbers', () => {
        const r = buildRegime([{ ticker: 'SPY', iv: null }], null);
        expect(r.curve).toBeNull();
        expect(r.decay.realized).toBeNull();
        expect(r.moves).toEqual({ days: 5, SPY: null, QQQ: null });
        expect(buildRegime(undefined, undefined).moves.SPY).toBeNull();
    });
});

describe('KV last-good', () => {
    const now = Date.parse('2026-09-26T14:00:00Z');
    const good = buildTermStructure(eod, {});

    it('saves a clean curve once per close date, and never a degraded one', async () => {
        const kv = fakeKv();
        expect(await saveCurveKV({ ...good, stale: true }, { kv, now })).toBe(false);
        expect(await saveCurveKV({ ...good, asOf: '2031-01-01' }, { kv, now })).toBe(true);
        expect(await saveCurveKV({ ...good, asOf: '2031-01-01' }, { kv, now })).toBe(false); // same date, same instance
        expect(kv.set).toHaveBeenCalledTimes(1);
        expect(kv.set.mock.calls[0][0]).toBe(CURVE_KV_KEY);
        expect(kv.set.mock.calls[0][1].savedAt).toBe('2026-09-26T14:00:00.000Z');
    });

    it('loads it back relabelled as a stale backup (object or JSON string)', async () => {
        const kv = fakeKv();
        await saveCurveKV(good, { kv, now, force: true });
        const back = await loadCurveKV({ kv, now: now + 3600e3 });
        expect(back).toMatchObject({ state: 'calm', stale: true, live: false, backup: 'KV 2026-09-26T14:00Z' });
        expect(curveDegraded(back)).toBe(true); // a backup never gets edge-cached

        kv.store.set(CURVE_KV_KEY, JSON.stringify(kv.store.get(CURVE_KV_KEY)));
        expect((await loadCurveKV({ kv, now })).backup).toBe('KV 2026-09-26T14:00Z');
    });

    it('drops copies older than 5 days, and never throws', async () => {
        const kv = fakeKv();
        await saveCurveKV(good, { kv, now, force: true });
        expect(await loadCurveKV({ kv, now: now + CURVE_MAX_AGE_MS + 1 })).toBeNull();
        const broken = { get: async () => { throw new Error('down'); }, set: async () => { throw new Error('down'); } };
        expect(await loadCurveKV({ kv: broken, now })).toBeNull();
        expect(await saveCurveKV(good, { kv: broken, now, force: true })).toBe(false);
        kv.store.set(CURVE_KV_KEY, '{not json');
        expect(await loadCurveKV({ kv, now })).toBeNull();
    });

    it('staleCurve refuses a curve with no call or no timestamp', () => {
        expect(staleCurve({ ...good, state: null }, '2026-09-26T14:00:00Z', now)).toBeNull();
        expect(staleCurve(good, 'garbage', now)).toBeNull();
        expect(staleCurve(good, '2026-09-26T14:00:00Z', now, 'last-good').backup).toBe('last-good 2026-09-26T14:00Z');
    });
});
