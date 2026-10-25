/**
 * @jest-environment node
 *
 * /api/spy-daily-move: the Lambda reads Sheet B12 raw, so a GOOGLEFINANCE hiccup ('#N/A',
 * 'Loading...', a blank cell) must fall through to the direct tiers, never be served,
 * edge-cached or saved as last-good.
 */
jest.mock('../store', () => ({
    serve: jest.fn(async (key, produce, opts) => {
        const payload = await produce();
        return { payload, isGood: opts.isGood };
    }),
}));
jest.mock('../spyTiers', () => ({
    fallbackMove: jest.fn(async () => ({ value: '+0.40%', source: 'Finnhub (fallback)' })),
}));

import { GET } from '../../app/api/spy-daily-move/route';

const realFetch = global.fetch;
const lambdaSays = (body) => { global.fetch = jest.fn(async () => ({ ok: true, json: async () => body })); };
beforeEach(() => { process.env.LAMBDA_URL = 'https://lambda.example'; });
afterEach(() => { global.fetch = realFetch; delete process.env.LAMBDA_URL; });
const get = () => GET(new Request('https://dash.example/api/spy-daily-move'));

test.each(['#N/A', 'Loading...', '', '  ', 'N/A', '#REF!'])('Lambda cell %p falls through to Finnhub', async (value) => {
    lambdaSays({ value, source: 'Google Sheets' });
    const { payload, isGood } = await get();
    expect(payload.source).toBe('Finnhub (fallback)');
    expect(isGood({ value, source: 'Google Sheets' })).toBe(false);
});

test('a real Sheet move is served as the Lambda sent it', async () => {
    lambdaSays({ value: '-0.52%', source: 'Google Sheets' });
    const { payload, isGood } = await get();
    expect(payload).toEqual({ value: '-0.52%', source: 'Google Sheets' });
    expect(isGood(payload)).toBe(true);
    expect(isGood({ value: '+0.40%' })).toBe(true);
    expect(isGood({ value: null })).toBe(false);
});
