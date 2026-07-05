import { parseCboeCsv, realizedVol, ivRank, ivPercentile, buildVolMetrics, VOL_PROXIES } from '../vol';

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
