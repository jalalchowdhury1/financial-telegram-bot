/**
 * @jest-environment node
 *
 * /api/spy and /api/spy-daily-move bound the Lambda hop with an AbortSignal, so a
 * hung Lambda can't hold the direct tiers back until API Gateway's ~29 s cap.
 */
jest.mock('../store', () => ({ serve: jest.fn((key, produce) => produce()) }));
jest.mock('../spyTiers', () => ({
    fallbackSpy: jest.fn(async () => ({ current: 1, _meta: { messages: [] } })),
    fallbackMove: jest.fn(async () => ({ value: 0.5 })),
    spyPreferNewer: jest.fn(),
}));

import { GET as spyGET } from '../../app/api/spy/route';
import { GET as moveGET } from '../../app/api/spy-daily-move/route';

const realFetch = global.fetch;
beforeEach(() => {
    process.env.LAMBDA_URL = 'https://lambda.example';
    global.fetch = jest.fn().mockRejectedValue(new Error('The operation was aborted due to timeout'));
});
afterEach(() => { global.fetch = realFetch; delete process.env.LAMBDA_URL; });

test.each([
    ['/api/spy', spyGET],
    ['/api/spy-daily-move', moveGET],
])('%s passes an AbortSignal to the Lambda fetch and falls through on abort', async (path, GET) => {
    const out = await GET(new Request(`https://dash.example${path}`));
    const [url, opts] = global.fetch.mock.calls[0];
    expect(url).toBe(`https://lambda.example${path}`);
    expect(opts.signal).toBeInstanceOf(AbortSignal);
    expect(out).toBeTruthy(); // fallback tier answered
});
