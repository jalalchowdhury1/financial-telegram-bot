/**
 * @jest-environment node
 *
 * /api/fred last-good: a partial FRED load (some series failed, N/A tiles) is served but
 * never stored, so it cannot overwrite a complete last-known-good copy (audit 25 Oct 2026).
 */
let opts;
jest.mock('../store', () => ({
    serve: jest.fn(async (key, build, o) => { opts = o; return new Response('{}'); }),
    loadLastGood: jest.fn(() => null),
}));

const { GET } = require('../../app/api/fred/route');

test('only a complete live load becomes the new last-known-good', async () => {
    await GET(new Request('https://x.test/api/fred', { headers: { 'user-agent': 'jest' } }));
    const meta = (loadedCount, horsemenLive = 0) => ({ _meta: { loadedCount, horsemenLive } });
    expect(opts.shouldStore(meta(17))).toBe(true);
    expect(opts.shouldStore(meta(5))).toBe(false);          // 12 of 17 failed: served, not stored
    expect(opts.isGood(meta(5))).toBe(true);
    expect(opts.shouldStore(meta(0, 4))).toBe(false);       // merged over the cache
});
