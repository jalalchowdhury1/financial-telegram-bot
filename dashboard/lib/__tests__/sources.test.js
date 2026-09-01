import { yahooChart, coingeckoPrice, dbnomicsFred, dailyChange, dxyFromUsdRates, polygonDaily, fredObservations, krakenSpot, fawazRates, cnbcQuotes, cnbcHistory, goldApiSpot } from '../sources';

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

describe('cnbcQuotes', () => {
    test('parses string last/change/change_pct + slices date from last_time (array form)', async () => {
        mockFetch({ QuickQuoteResult: { QuickQuote: [
            { symbol: '@HG.1', last: '6.2755', change: '-0.2595', change_pct: '-3.97', last_time: '2026-06-06T18:23:33.000-0400', currencyCode: 'USD' },
            { symbol: '@GC.1', last: '4353.90', change: '-151.10', change_pct: '-3.35', last_time: '2026-06-06T18:23:04.000-0400' },
        ] } });
        const out = await cnbcQuotes(['@HG.1', '@GC.1']);
        expect(out['@HG.1'].price).toBeCloseTo(6.2755, 4);
        expect(out['@HG.1'].changePct).toBeCloseTo(-3.97, 2);
        expect(out['@HG.1'].asOf).toBe('2026-06-06');
        expect(out['@GC.1'].price).toBeCloseTo(4353.9, 1);
    });
    test('handles a single (non-array) QuickQuote object', async () => {
        mockFetch({ QuickQuoteResult: { QuickQuote: { symbol: '@HG.1', last: '6.10', change: '0.05', change_pct: '0.8', last_time: '2026-06-05T20:00:00.000-0400' } } });
        const out = await cnbcQuotes(['@HG.1']);
        expect(out['@HG.1'].price).toBe(6.10);
    });
    test('returns the full last_time as lastTime alongside the sliced asOf', async () => {
        mockFetch({ QuickQuoteResult: { QuickQuote: [
            { symbol: '.VIX', last: '16.39', change: '-0.11', change_pct: '-0.67', last_time: '2026-07-15T13:42:31.000-0400' },
            { symbol: '.VXN', last: '26.28', change: '0.00', change_pct: '0.00', last_time: '2026-07-14' },
        ] } });
        const out = await cnbcQuotes(['.VIX', '.VXN']);
        expect(out['.VIX'].lastTime).toBe('2026-07-15T13:42:31.000-0400');
        expect(out['.VIX'].asOf).toBe('2026-07-15');
        expect(out['.VXN'].lastTime).toBe('2026-07-14'); // CNBC sends date-only off-hours
        expect(out['.VXN'].asOf).toBe('2026-07-14');
    });
    test('throws when no parseable quotes', async () => {
        mockFetch({ QuickQuoteResult: { QuickQuote: [{ symbol: '@HG.1', last: 'N/A' }] } });
        await expect(cnbcQuotes(['@HG.1'])).rejects.toThrow();
    });
});

describe('cnbcHistory', () => {
    test('parses tradeTime+close into ascending {date,price}', async () => {
        mockFetch({ barData: { priceBars: [
            { tradeTime: '20260605000000', close: '6.28' },
            { tradeTime: '20260601000000', close: '6.10' }, // out of order on purpose
        ] } });
        const out = await cnbcHistory('@HG.1');
        expect(out).toHaveLength(2);
        expect(out[0]).toEqual({ date: '2026-06-01', price: 6.10 }); // sorted oldest-first
        expect(out[1]).toEqual({ date: '2026-06-05', price: 6.28 });
    });
    test('throws on empty bars', async () => {
        mockFetch({ barData: { priceBars: [] } });
        await expect(cnbcHistory('@HG.1')).rejects.toThrow();
    });
});

describe('goldApiSpot', () => {
    test('returns current + asOf date', async () => {
        mockFetch({ symbol: 'XAU', price: 4330.0, updatedAt: '2026-06-06T13:55:25Z' });
        const out = await goldApiSpot('XAU');
        expect(out.current).toBe(4330.0);
        expect(out.asOf).toBe('2026-06-06');
    });
    test('throws when price missing', async () => {
        mockFetch({ symbol: 'XAU' });
        await expect(goldApiSpot('XAU')).rejects.toThrow();
    });
});
