/**
 * @jest-environment node
 *
 * /api/fred time budget: the optional tiers (P/E, Horsemen repair, copper/gold, S&P EPS,
 * bankruptcies) share FRED_BUDGET_MS from request start. A hung tier is cut, its card is
 * left N/A with a message, and the route still answers inside maxDuration (no 504).
 */
const hang = () => new Promise(() => {});
const obs = () => ({ observations: Array.from({ length: 30 }, (_, i) => ({ date: new Date(Date.UTC(2026, 9, 8) - i * 864e5).toISOString().slice(0, 10), value: String(3 + i * 0.01) })) });

jest.mock('../fetcher', () => ({
    fetchJson: jest.fn(async () => obs()),
    fetchText: jest.fn(hang),
    proxyFetch: jest.fn(hang),
}));
jest.mock('../sources', () => new Proxy({}, { get: (_, k) => (k === '__esModule' ? true : jest.fn(hang)) }));
jest.mock('../bankruptcies', () => ({ resolveBankruptcies: jest.fn(hang) }));
jest.mock('../sheetLkg', () => ({ fetchSheetLkg: jest.fn(async () => null) }));

const { GET, maxDuration } = require('../../app/api/fred/route');

beforeAll(() => { process.env.FRED_API_KEY = 'k'; });
afterAll(() => { delete process.env.FRED_API_KEY; jest.useRealTimers(); });

test('maxDuration is declared (Hobby: <= 60 s valid with or without Fluid)', () => {
    expect(maxDuration).toBe(60);
});

test('hung optional tiers are cut at the budget; FRED series still served', async () => {
    jest.useFakeTimers({ doNotFake: ['nextTick', 'queueMicrotask', 'setImmediate'] });
    // `?_fail=none` = fault test mode → nothing is written to /tmp or KV.
    const p = GET(new Request('https://x.test/api/fred?_fail=none', { headers: { 'user-agent': 'jest' } }));
    await jest.advanceTimersByTimeAsync(45000);
    const res = await p;
    const b = await res.json();
    expect(res.status).toBe(200);
    expect(b._meta.loadedCount).toBeGreaterThan(0);
    const msgs = b._meta.messages.join(' | ');
    for (const what of ['P/E', 'Copper/Gold', 'Bankruptcies']) expect(msgs).toMatch(new RegExp(`${what.replace('/', '\\/')} skipped: request time budget`));
    expect(b.indicators.copperGold.unavailable).toBe(true);
});
