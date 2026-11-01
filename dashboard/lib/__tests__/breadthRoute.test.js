/**
 * @jest-environment node
 *
 * /api/breadth staleness gate (audit, 1 Nov 2026): a pair more than one session behind the
 * newest close still shows, but labelled (pair `stale`, `_meta.stale` + `asOf`, hasErrors) and
 * is never saved as last-good. Calls carry `?_fail=breadth_polygon` (test mode: no /tmp or KV
 * writes) so only the mocked CNBC bars answer.
 */
let lastBar = {};
jest.mock('../sources', () => ({
    polygonDaily: jest.fn(async () => { throw new Error('unused'); }),
    cnbcHistory: jest.fn(async (ticker) => {
        const end = Date.parse(`${lastBar[ticker] || lastBar['*']}T12:00:00Z`);
        return Array.from({ length: 80 }, (_, i) => ({ date: new Date(end - (79 - i) * 864e5).toISOString().slice(0, 10), price: 100 + i }));
    }),
}));

const { GET } = require('../../app/api/breadth/route');
const get = async () => (await GET(new Request('https://x.test/api/breadth?_fail=breadth_polygon,tmplg,kvlg', { headers: { 'user-agent': 'jest' } }))).json();

beforeAll(() => jest.useFakeTimers({ now: new Date('2026-11-01T21:22:00Z'), doNotFake: ['setTimeout', 'clearTimeout', 'setImmediate', 'nextTick', 'queueMicrotask'] }));
afterAll(() => jest.useRealTimers());

test('live 1 Nov shape (Friday bars on Sunday) is not stale', async () => {
    lastBar = { '*': '2026-10-30' };
    const b = await get();
    expect(Object.keys(b.pairs)).toHaveLength(4);
    expect(b._meta.stale).toBeFalsy();
    expect(b._meta.hasErrors).toBe(false);
});

test('one pair stuck two sessions back -> labelled stale, dated by its own last day', async () => {
    lastBar = { '*': '2026-10-30', HYG: '2026-10-28', LQD: '2026-10-28' };
    const b = await get();
    expect(b.pairs.hygLqd).toMatchObject({ asOf: '2026-10-28', stale: true });
    expect(b.pairs.rspSpy.stale).toBeUndefined();
    expect(b._meta).toMatchObject({ stale: true, asOf: '2026-10-28', hasErrors: true });
    expect(b._meta.messages.join(' ')).toMatch(/hygLqd: stale/);
});
