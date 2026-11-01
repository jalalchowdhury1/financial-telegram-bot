/**
 * @jest-environment node
 *
 * /api/spy: a Lambda answer goes first only in SPY dollars, on one scale, for the latest
 * session. Audit 1 Nov 2026: its FRED layer + a SPY spot read "−89% vs 200-day" and was
 * served + saved as last-good; a Polygon-only answer held yesterday's bars as live.
 */
jest.mock('../store', () => ({
    serve: jest.fn(async (key, produce, opts) => ({ payload: await produce().catch((e) => ({ thrown: e.message })), opts })),
}));
jest.mock('../spyTiers', () => ({ ...jest.requireActual('../spyTiers'), fallbackSpy: jest.fn() }));

import { GET, lambdaDoubt } from '../../app/api/spy/route';
const { fallbackSpy } = require('../spyTiers');

const NOW = Date.parse('2026-11-01T15:00:00Z'); // Sunday → latest session Fri 2026-10-30
const lam = ({ source = 'yfinance + Finnhub Spot', current = 755.72, ma200 = 728.85, last = '2026-10-30', price = current } = {}) => ({
    current, ma200: { value: ma200, pct: 3.7 }, week52High: { value: 779.09, pct: -3 }, rsi: 35.7, return3y: 81.8,
    chartHistory: [{ date: '2026-10-28', price: price - 1 }, { date: last, price }],
    _meta: { source, hasErrors: false, messages: [`Loaded from ${source}`] },
});
const fb = (meta = {}) => ({ current: 755.7, ma200: { value: 728 }, week52High: { value: 779 }, _meta: { source: 'Polygon + Finnhub (fallback)', messages: [], ...meta } });
const realFetch = global.fetch;
let lambdaSays;
beforeEach(() => {
    jest.spyOn(Date, 'now').mockReturnValue(NOW);
    process.env.LAMBDA_URL = 'https://lambda.example';
    global.fetch = jest.fn(async () => ({ ok: true, json: async () => lambdaSays }));
    fallbackSpy.mockReset();
});
afterEach(() => { global.fetch = realFetch; delete process.env.LAMBDA_URL; jest.restoreAllMocks(); });
const get = (q = '') => GET(new Request(`https://dash.example/api/spy${q}`));

describe('lambdaDoubt', () => {
    test('the normal answer (yfinance + Finnhub, newest bar = latest session) goes first', () => {
        expect(lambdaDoubt(lam(), NOW)).toBeNull();
    });
    test('FRED index points, with or without a SPY spot on top, are never served', () => {
        expect(lambdaDoubt(lam({ source: 'FRED S&P 500 Index + Finnhub Spot', current: 680, ma200: 6500, price: 6800 }), NOW).never).toBe(true);
        expect(lambdaDoubt(lam({ source: 'FRED S&P 500 Index', current: 6800, ma200: 6500 }), NOW).never).toBe(true);
    });
    test('price, 200-day and chart on different scales are never served', () => {
        expect(lambdaDoubt(lam({ ma200: 6500 }), NOW).never).toBe(true);
        expect(lambdaDoubt(lam({ price: 6800 }), NOW).never).toBe(true);
    });
    test('the sheet layer waits (its chart is FRED index points by design)', () => {
        const d = lambdaDoubt(lam({ source: 'Google Sheet', price: 6800 }), NOW);
        expect(d.never).toBeUndefined();
        expect(d.asOf).toBeUndefined();
    });
    test('an older session waits, dated by its newest bar', () => {
        expect(lambdaDoubt(lam({ last: '2026-10-29' }), NOW)).toMatchObject({ asOf: '2026-10-29' });
    });
});

describe('GET /api/spy', () => {
    test('mixed scales + every tier down → nothing from the Lambda (serve() falls to last-good)', async () => {
        lambdaSays = lam({ source: 'FRED S&P 500 Index + Finnhub Spot', current: 680, ma200: 6500, price: 6800 });
        fallbackSpy.mockRejectedValue(new Error('all SPY tiers failed'));
        const { payload } = await get();
        expect(payload.thrown).toMatch(/all SPY tiers failed/);
    });
    test('mixed scales + a working tier → the tier', async () => {
        lambdaSays = lam({ source: 'FRED S&P 500 Index + Finnhub Spot', current: 680, ma200: 6500, price: 6800 });
        fallbackSpy.mockResolvedValue(fb());
        const { payload } = await get();
        expect(payload._meta.source).toBe('Polygon + Finnhub (fallback)');
        expect(payload._meta.messages.join(' ')).toMatch(/index points/);
    });
    test('an older session: our fresh tier first; served only when tiers fail, flagged stale', async () => {
        lambdaSays = lam({ last: '2026-10-29' });
        fallbackSpy.mockResolvedValue(fb());
        expect((await get()).payload._meta.source).toBe('Polygon + Finnhub (fallback)');
        fallbackSpy.mockRejectedValue(new Error('all SPY tiers failed'));
        const { payload, opts } = await get();
        expect(payload._meta).toMatchObject({ source: 'yfinance + Finnhub Spot', stale: true, hasErrors: true, asOf: '2026-10-29' });
        expect(opts.isGood(payload)).toBe(false); // never saved as last-good
    });
    test('an older session vs a stale tier: the newer close wins, a tie goes to our tier', async () => {
        lambdaSays = lam({ last: '2026-10-29' });
        fallbackSpy.mockResolvedValue(fb({ stale: true, asOf: '2026-10-28' }));
        expect((await get()).payload._meta.asOf).toBe('2026-10-29');
        expect((await get()).payload._meta.source).toBe('yfinance + Finnhub Spot');
        fallbackSpy.mockResolvedValue(fb({ stale: true, asOf: '2026-10-29' }));
        expect((await get()).payload._meta.source).toBe('Polygon + Finnhub (fallback)');
    });
    test('the sheet answer still beats a stale-only tier (unchanged)', async () => {
        lambdaSays = lam({ source: 'Google Sheet', price: 6800 });
        fallbackSpy.mockResolvedValue(fb({ stale: true, asOf: '2026-10-29' }));
        expect((await get()).payload._meta.source).toBe('Google Sheet');
    });
    test('?debug=compare without the secret is the normal answer (no Polygon spend)', async () => {
        lambdaSays = lam();
        const res = await get('?debug=compare');
        expect(res.payload._meta.source).toBe('yfinance + Finnhub Spot');
        expect(fallbackSpy).not.toHaveBeenCalled();
    });
});
