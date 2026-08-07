import { parseCboeCsv, realizedVol, ivRank, ivPercentile, buildVolMetrics, VOL_PROXIES, resolveVolSeries, volIncompleteTickers } from '../vol';

describe('parseCboeCsv', () => {
    it('parses the OHLC schema (VIX/VXN) taking CLOSE', () => {
        const csv = 'DATE,OPEN,HIGH,LOW,CLOSE\n07/02/2026,15.1,16.0,15.0,15.5\n07/03/2026,15.7,16.0,15.7,15.9\n';
        expect(parseCboeCsv(csv)).toEqual([
            { date: '2026-07-02', value: 15.5 },
            { date: '2026-07-03', value: 15.9 },
        ]);
    });

    it('parses the two-column schema (VVIX)', () => {
        const csv = 'DATE,VVIX\n07/01/2026,90.2\n07/02/2026,88.8\n';
        expect(parseCboeCsv(csv)).toEqual([
            { date: '2026-07-01', value: 90.2 },
            { date: '2026-07-02', value: 88.8 },
        ]);
    });

    it('skips malformed rows and sorts ascending', () => {
        const csv = 'DATE,OPEN,HIGH,LOW,CLOSE\n07/03/2026,1,1,1,12\nbad,row,,,\n07/01/2026,1,1,1,10\n07/02/2026,1,1,1,notanumber\n';
        expect(parseCboeCsv(csv)).toEqual([
            { date: '2026-07-01', value: 10 },
            { date: '2026-07-03', value: 12 },
        ]);
    });

    it('returns [] on garbage', () => {
        expect(parseCboeCsv('')).toEqual([]);
        expect(parseCboeCsv('just one line')).toEqual([]);
    });
});

describe('realizedVol', () => {
    it('is ~0 for constant prices', () => {
        expect(realizedVol(Array(30).fill(100))).toBeCloseTo(0, 6);
    });

    it('matches a hand-computed 2-return case', () => {
        // closes 100 → 105 → 100: log returns ±ln(1.05), mean 0,
        // sample std = ln(1.05)·√2, annualized ×√252×100.
        const r = Math.log(1.05);
        const expected = r * Math.sqrt(2) * Math.sqrt(252) * 100;
        expect(realizedVol([100, 105, 100], 2)).toBeCloseTo(expected, 6);
    });

    it('returns null with too little history', () => {
        expect(realizedVol([100, 101], 21)).toBeNull();
        expect(realizedVol(null)).toBeNull();
    });

    it('uses only the last N+1 closes', () => {
        const flatThenMove = [...Array(50).fill(50), ...Array(22).fill(100)];
        expect(realizedVol(flatThenMove, 21)).toBeCloseTo(0, 6);
    });
});

describe('ivRank / ivPercentile', () => {
    const win = [10, 12, 14, 16, 18, 20];

    it('rank: midpoint of min..max is 50', () => {
        expect(ivRank(win, 15)).toBeCloseTo(50);
        expect(ivRank(win, 10)).toBeCloseTo(0);
        expect(ivRank(win, 20)).toBeCloseTo(100);
    });

    it('rank clamps outside the window and nulls a flat window', () => {
        expect(ivRank(win, 25)).toBe(100);
        expect(ivRank([5, 5, 5], 5)).toBeNull();
    });

    it('percentile: share of days at-or-below', () => {
        expect(ivPercentile(win, 15)).toBeCloseTo(50); // 3 of 6
        expect(ivPercentile(win, 20)).toBeCloseTo(100);
        expect(ivPercentile(win, 9)).toBeCloseTo(0);
    });

    it('null on empty/invalid inputs', () => {
        expect(ivRank([], 10)).toBeNull();
        expect(ivPercentile(win, NaN)).toBeNull();
    });
});

describe('buildVolMetrics', () => {
    const mkSeries = (values) => values.map((v, i) => ({ date: `2026-01-${String((i % 28) + 1).padStart(2, '0')}`, value: v }));

    it('scales IV by the proxy multiplier but keeps rank/pctile from the raw index', () => {
        const vxn = mkSeries([...Array(260)].map((_, i) => 20 + (i % 10))); // last value 20+(259%10)=29
        const out = buildVolMetrics({ VXN: vxn }, {});
        const qqq = out.tickers.find((t) => t.ticker === 'QQQ');
        const tqqq = out.tickers.find((t) => t.ticker === 'TQQQ');
        expect(qqq.iv).toBeCloseTo(29);
        expect(tqqq.iv).toBeCloseTo(87); // 3×29
        expect(tqqq.ivRank1y).toBeCloseTo(qqq.ivRank1y); // multiplier cancels
        expect(tqqq.ivPctile1y).toBeCloseTo(qqq.ivPctile1y);
        expect(tqqq.proxy).toBe('3×VXN');
    });

    it('degrades gracefully: missing index nulls IV but keeps RV; VRP needs both', () => {
        const closes = [...Array(30)].map((_, i) => 100 * Math.exp(0.01 * (i % 2 ? 1 : -1)));
        const out = buildVolMetrics({}, { UVXY: closes });
        const uvxy = out.tickers.find((t) => t.ticker === 'UVXY');
        expect(uvxy.iv).toBeNull();
        expect(uvxy.rv21).not.toBeNull();
        expect(uvxy.vrp).toBeNull();
    });

    it('computes VRP = IV − RV and reports every proxy ticker', () => {
        const vix = mkSeries(Array(252).fill(16));
        vix[vix.length - 1] = { date: '2026-07-03', value: 20 };
        const closes = Array(40).fill(100); // RV 0
        const out = buildVolMetrics({ VIX: vix }, { SPY: closes });
        const spy = out.tickers.find((t) => t.ticker === 'SPY');
        expect(spy.vrp).toBeCloseTo(20);
        expect(out.updated_at).toBe('2026-07-03');
        expect(out.tickers.map((t) => t.ticker)).toEqual(Object.keys(VOL_PROXIES));
    });
});

describe('buildVolMetrics — live intraday overrides', () => {
    const mkSeries = (values) => values.map((v, i) => ({ date: `2026-01-${String((i % 28) + 1).padStart(2, '0')}`, value: v }));
    // 1y window: 251 EOD days at 16, last EOD close 20 on 2026-07-14 → window min 16, max 20.
    const vixWithLast = () => {
        const s = mkSeries(Array(252).fill(16));
        s[s.length - 1] = { date: '2026-07-14', value: 20 };
        return s;
    };
    const liveVix = { value: 18, date: '2026-07-15', lastTime: '2026-07-15T13:42:31.000-0400' };

    it('replaces the last EOD close when the quote date is strictly newer', () => {
        const out = buildVolMetrics({ VIX: vixWithLast() }, { SPY: Array(40).fill(100) }, { VIX: liveVix });
        const spy = out.tickers.find((t) => t.ticker === 'SPY');
        expect(spy.iv).toBeCloseTo(18);
        expect(spy.live).toBe(true);
        expect(spy.asOf).toBe('2026-07-15');
        expect(spy.ivRank1y).toBeCloseTo(50);   // (18−16)/(20−16) against the EOD window
        expect(spy.ivPctile1y).toBeCloseTo((251 / 252) * 100, 1); // 251 of 252 window days ≤ 18
        expect(spy.rv21).toBeCloseTo(0, 6);     // RV stays EOD-only (flat closes)
        expect(spy.vrp).toBeCloseTo(18);        // live IV − EOD RV
        expect(out.updated_at).toBe('2026-07-15');
        expect(out.live_at).toBe('2026-07-15T13:42:31.000-0400');
    });

    it('scales the live level by the proxy multiplier but ranks on the raw index', () => {
        const vxn = mkSeries(Array(252).fill(20));
        vxn[vxn.length - 1] = { date: '2026-07-14', value: 30 };
        const out = buildVolMetrics({ VXN: vxn }, {}, { VXN: { value: 25, date: '2026-07-15', lastTime: '2026-07-15T10:00:00.000-0400' } });
        const qqq = out.tickers.find((t) => t.ticker === 'QQQ');
        const tqqq = out.tickers.find((t) => t.ticker === 'TQQQ');
        expect(qqq.iv).toBeCloseTo(25);
        expect(tqqq.iv).toBeCloseTo(75); // 3×25
        expect(tqqq.ivRank1y).toBeCloseTo(qqq.ivRank1y); // multiplier cancels
        expect(tqqq.live).toBe(true);
    });

    it('skips the override when the quote is not strictly newer (evenings/weekends)', () => {
        const sameDay = buildVolMetrics({ VIX: vixWithLast() }, {}, { VIX: { ...liveVix, date: '2026-07-14' } });
        const older = buildVolMetrics({ VIX: vixWithLast() }, {}, { VIX: { ...liveVix, date: '2026-07-11' } });
        for (const out of [sameDay, older]) {
            const spy = out.tickers.find((t) => t.ticker === 'SPY');
            expect(spy.iv).toBeCloseTo(20);
            expect(spy.live).toBe(false);
            expect(spy.asOf).toBe('2026-07-14');
            expect(out.live_at).toBeNull();
        }
    });

    it('never lets a garbage quote replace good EOD math', () => {
        const cases = [
            { value: NaN, date: '2026-07-15' },
            { value: -3, date: '2026-07-15' },
            { value: 0, date: '2026-07-15' },
            { value: 18, date: null },
            { value: 18 }, // no date at all
            { value: 18, date: 'not-a-date' },
            { value: 18, date: '2026-7-15' }, // non-padded month — not a safe string compare
        ];
        for (const quote of cases) {
            const out = buildVolMetrics({ VIX: vixWithLast() }, {}, { VIX: quote });
            const spy = out.tickers.find((t) => t.ticker === 'SPY');
            expect(spy.iv).toBeCloseTo(20);
            expect(spy.live).toBe(false);
        }
    });

    it('a live quote alone (no EOD series) produces nothing — no window, no metrics', () => {
        const out = buildVolMetrics({}, {}, { VIX: liveVix });
        const spy = out.tickers.find((t) => t.ticker === 'SPY');
        expect(spy.iv).toBeNull();
        expect(spy.live).toBe(false);
    });

    it('a date-only lastTime never becomes live_at (avoids UTC-midnight misformatting)', () => {
        const out = buildVolMetrics({ VIX: vixWithLast() }, {}, { VIX: { value: 18, date: '2026-07-15', lastTime: '2026-07-15' } });
        const spy = out.tickers.find((t) => t.ticker === 'SPY');
        expect(spy.live).toBe(true);        // the override still applies
        expect(out.live_at).toBeNull();     // but the footnote falls back to updated_at
        expect(out.updated_at).toBe('2026-07-15');
    });

    it('is fully backward-compatible when liveQuotes is omitted', () => {
        const out = buildVolMetrics({ VIX: vixWithLast() }, { SPY: Array(40).fill(100) });
        const spy = out.tickers.find((t) => t.ticker === 'SPY');
        expect(spy.iv).toBeCloseTo(20);
        expect(spy.live).toBe(false);
        expect(out.live_at).toBeNull();
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// Staleness gate (added 2026-08-06). /api/vol used to accept ANY series with
// length > 0, so a frozen CBOE/CNBC feed served months-old vol as today's
// number. Every sibling cascade (copperGold.resolveLeg, horsemen) rejects a
// stale source and falls through; this one now does too.
// ─────────────────────────────────────────────────────────────────────────────
describe('resolveVolSeries', () => {
    const NOW = new Date('2026-08-06T12:00:00Z');
    const pts = (date, value = 15) => [{ date: '2026-01-01', value: 10 }, { date, value }];
    const src = (name, fetch, extra = {}) => ({ name, gate: `vol_${name}`, fetch, ...extra });

    it('takes the first source whose newest point is fresh', async () => {
        const r = await resolveVolSeries([src('cboe', async () => pts('2026-08-05'))], null, NOW);
        expect(r.source).toBe('cboe');
        expect(r.asOf).toBe('2026-08-05');
        expect(r.tried).toEqual(['cboe:ok']);
    });

    it('REJECTS a stale source and falls through to a fresh one', async () => {
        const r = await resolveVolSeries([
            src('cboe', async () => pts('2026-05-01', 99)),   // frozen months ago
            src('cnbc', async () => pts('2026-08-05', 15)),
        ], null, NOW);
        expect(r.source).toBe('cnbc');
        expect(r.points[r.points.length - 1].value).toBe(15);
        expect(r.tried).toEqual(['cboe:stale(2026-05-01)', 'cnbc:ok']);
    });

    it('returns a null series when every source is stale/empty/erroring', async () => {
        const r = await resolveVolSeries([
            src('cboe', async () => pts('2026-05-01')),
            src('cnbc', async () => []),
            src('yahoo', async () => { throw new Error('blocked'); }),
        ], null, NOW);
        expect(r.points).toBeNull();
        expect(r.source).toBeNull();
        expect(r.tried).toEqual(['cboe:stale(2026-05-01)', 'cnbc:empty', 'yahoo:err']);
    });

    it('honors the per-source fault gate like cg_* does', async () => {
        const faults = new Set(['vol_cboe']);
        const r = await resolveVolSeries([
            src('cboe', async () => pts('2026-08-05', 99)),
            src('cnbc', async () => pts('2026-08-05', 15)),
        ], faults, NOW);
        expect(r.source).toBe('cnbc');
        expect(r.tried).toEqual(['cboe:off', 'cnbc:ok']);
    });

    it('allows a per-source freshness window (a day-delayed tier is still fresh)', async () => {
        const r = await resolveVolSeries(
            [src('polygon', async () => pts('2026-07-31'), { freshnessDays: 10 })], null, NOW);
        expect(r.source).toBe('polygon');
    });
});

describe('volIncompleteTickers', () => {
    it('names every ticker missing iv or rv21', () => {
        const tickers = [
            { ticker: 'SPY', iv: 15, rv21: 14 },
            { ticker: 'UVXY', iv: null, rv21: 77 },   // dead VVIX cascade (no FRED tier)
            { ticker: 'QQQ', iv: 24, rv21: null },
        ];
        expect(volIncompleteTickers(tickers)).toEqual(['UVXY', 'QQQ']);
    });

    it('is empty on a fully healthy table', () => {
        expect(volIncompleteTickers([{ ticker: 'SPY', iv: 15, rv21: 14 }])).toEqual([]);
    });

    it('treats a missing/empty table as incomplete-free but the caller sees no rows', () => {
        expect(volIncompleteTickers(null)).toEqual([]);
    });
});
