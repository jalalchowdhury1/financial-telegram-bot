/**
 * @jest-environment node
 *
 * /api/fred passes P/E's stale flag on: lib/peRatio serves a frozen multpl page last, stale with its
 * own date (audit B6), but the route dropped the flag and the tile hard-coded stale: false.
 */
const boom = async () => { throw new Error('down'); };
const obs = () => ({ observations: Array.from({ length: 30 }, (_, i) => ({ date: new Date(Date.UTC(2026, 9, 8) - i * 864e5).toISOString().slice(0, 10), value: String(3 + i * 0.01) })) });

jest.mock('../fetcher', () => ({ fetchJson: jest.fn(async () => obs()), fetchText: jest.fn(boom), proxyFetch: jest.fn(boom) }));
jest.mock('../sources', () => new Proxy({}, { get: (_, k) => (k === '__esModule' ? true : jest.fn(boom)) }));
jest.mock('../bankruptcies', () => ({ resolveBankruptcies: jest.fn(boom) }));
jest.mock('../sheetLkg', () => ({ fetchSheetLkg: jest.fn(async () => null) }));
jest.mock('../peRatio', () => ({
    ...jest.requireActual('../peRatio'),
    resolvePeRatio: jest.fn(async () => ({ peRatio: 25.86, peSource: 'multpl', peIsCape: false, peAsOf: '2026-10-22', stale: true, messages: [] })),
}));

const { GET } = require('../../app/api/fred/route');

beforeAll(() => { process.env.FRED_API_KEY = 'k'; });
afterAll(() => { delete process.env.FRED_API_KEY; jest.useRealTimers(); });

test('a frozen multpl page served last reaches the tile stale, with its own date', async () => {
    jest.useFakeTimers({ doNotFake: ['nextTick', 'queueMicrotask', 'setImmediate'] });
    // `?_fail=none` = fault test mode → nothing is written to /tmp or KV.
    const p = GET(new Request('https://x.test/api/fred?_fail=none', { headers: { 'user-agent': 'jest' } }));
    await jest.advanceTimersByTimeAsync(45000);
    const b = await (await p).json();
    expect(b).toMatchObject({ peRatio: 25.86, peRatioAsOf: '2026-10-22', peStale: true });
});
