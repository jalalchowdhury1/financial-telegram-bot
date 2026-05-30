import { yahooChart, stooqDaily, coingeckoPrice, dbnomicsFred, dailyChange, dxyFromUsdRates, polygonDaily, fredObservations, krakenSpot, fawazRates } from '../sources';

// Mock the network: proxyFetch -> global.fetch. Return ok + json()/text().
function mockFetch(payload, { text = false } = {}) {
    global.fetch = jest.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => payload,
        text: async () => (text ? payload : JSON.stringify(payload)),
    }));
}

afterEach(() => { jest.restoreAllMocks(); });

describe('dailyChange', () => {
    test('computes value and pct', () => {
        expect(dailyChange(110, 100)).toEqual({ value: 10, pct: 10 });
    });
    test('zero prev -> pct 0', () => {
        expect(dailyChange(5, 0)).toEqual({ value: 5, pct: 0 });
    });
});

describe('dbnomicsFred', () => {
    test('parses and returns newest-first, drops NA', async () => {
        mockFetch({ series: { docs: [{ period: ['2020-01-01', '2020-02-01', '2020-03-01'], value: [1.5, 'NA', 1.7] }] } });
        const out = await dbnomicsFred('M2SL');
        expect(out[0]).toEqual({ date: '2020-03-01', value: 1.7 });
        expect(out[1]).toEqual({ date: '2020-01-01', value: 1.5 });
        expect(out).toHaveLength(2);
    });
    test('throws on empty', async () => {
        mockFetch({ series: { docs: [{ period: [], value: [] }] } });
        await expect(dbnomicsFred('X')).rejects.toThrow();
    });
});

describe('stooqDaily', () => {
    test('parses CSV oldest->newest with current/prev', async () => {
        mockFetch('Date,Open,High,Low,Close,Volume\n2026-05-28,1,2,0.5,1.5,100\n2026-05-29,1,2,0.5,1.7,100', { text: true });
        const out = await stooqDaily('spy.us');
        expect(out.current).toBe(1.7);
        expect(out.prevClose).toBe(1.5);
        expect(out.history).toHaveLength(2);
        expect(out.history[0]).toEqual({ date: '2026-05-28', price: 1.5 });
    });
    test('rejects bad CSV', async () => {
        mockFetch('garbage', { text: true });
        await expect(stooqDaily('x')).rejects.toThrow();
    });
});

describe('yahooChart', () => {
    test('uses regularMarketPrice + builds history', async () => {
        mockFetch({ chart: { result: [{
            timestamp: [1700000000, 1700086400],
            indicators: { quote: [{ close: [10, 11] }] },
            meta: { regularMarketPrice: 11.5, chartPreviousClose: 11 },
        }] } });
        const out = await yahooChart('SPY', { range: '1mo' });
        expect(out.current).toBe(11.5);
        expect(out.prevClose).toBe(11);
        expect(out.history).toHaveLength(2);
    });
    test('throws when no result', async () => {
        mockFetch({ chart: { result: [] } });
        await expect(yahooChart('SPY')).rejects.toThrow();
    });
});

describe('coingeckoPrice', () => {
    test('derives prevClose from 24h change', async () => {
        mockFetch({ bitcoin: { usd: 70000, usd_24h_change: 2 } });
        const out = await coingeckoPrice('bitcoin');
        expect(out.current).toBe(70000);
        expect(out.prevClose).toBeCloseTo(70000 / 1.02, 2);
    });
});

describe('dxyFromUsdRates', () => {
    test('computes a realistic DXY (~98-100) from typical USD rates', () => {
        // Rates are "currency per USD"
        const dxy = dxyFromUsdRates({ EUR: 0.92, JPY: 150, GBP: 0.79, CAD: 1.38, SEK: 9.5, CHF: 0.83 });
        expect(dxy).toBeGreaterThan(90);
        expect(dxy).toBeLessThan(110);
    });
    test('returns null when a currency is missing', () => {
        expect(dxyFromUsdRates({ EUR: 0.92 })).toBeNull();
    });
});

describe('polygonDaily', () => {
    test('parses aggregates into history + current/prev', async () => {
        mockFetch({ status: 'OK', results: [
            { t: 1700000000000, c: 100 }, { t: 1700086400000, c: 101 }, { t: 1700172800000, c: 102 },
        ] });
        const out = await polygonDaily('SPY', 'KEY');
        expect(out.current).toBe(102);
        expect(out.prevClose).toBe(101);
        expect(out.history).toHaveLength(3);
    });
    test('throws without a key', async () => {
        await expect(polygonDaily('SPY', '')).rejects.toThrow('no API key');
    });
    test('throws on Polygon ERROR status', async () => {
        mockFetch({ status: 'ERROR', error: 'unknown ticker' });
        await expect(polygonDaily('BAD', 'KEY')).rejects.toThrow();
    });
});

describe('krakenSpot', () => {
    test('parses last-trade price from result', async () => {
        mockFetch({ result: { XXBTZUSD: { c: ['73000.5', '0.01'] } } });
        const out = await krakenSpot('XBTUSD');
        expect(out.current).toBe(73000.5);
    });
    test('throws on empty result', async () => {
        mockFetch({ result: {} });
        await expect(krakenSpot('XBTUSD')).rejects.toThrow();
    });
});

describe('fawazRates', () => {
    test('uppercases the rate keys', async () => {
        mockFetch({ date: '2026-05-30', usd: { cad: 1.38, inr: 95, bdt: 122 } });
        const out = await fawazRates('usd');
        expect(out.CAD).toBe(1.38);
        expect(out.BDT).toBe(122);
    });
});

describe('fredObservations', () => {
    test('parses, drops dots, newest-first', async () => {
        mockFetch({ observations: [
            { date: '2026-05-28', value: '4.45' }, { date: '2026-05-27', value: '.' }, { date: '2026-05-26', value: '4.40' },
        ] });
        const out = await fredObservations('DGS10', 'KEY');
        expect(out[0]).toEqual({ date: '2026-05-28', value: 4.45 });
        expect(out).toHaveLength(2);
    });
});
