/**
 * @jest-environment node
 *
 * /api/fred when FRED itself is down or hanging (audit B, 1 Nov 2026): the merge base,
 * the time budget, per-series fills and the Horsemen-derived indicators.
 */
const day = (n) => new Date(Date.now() - n * 864e5).toISOString().slice(0, 10);
// Newest-first, one point per `step` days (the shape FRED's sort_order=desc gives).
const fredObs = (n, step, v = (i) => 3 + i * 0.01) => ({ observations: Array.from({ length: n }, (_, i) => ({ date: day(3 + i * step), value: String(v(i)) })) });

const mockSrc = {};
jest.mock('../fetcher', () => ({ fetchJson: jest.fn(), fetchText: jest.fn(), proxyFetch: jest.fn(async () => { throw new Error('down'); }) }));
jest.mock('../sources', () => new Proxy({}, { get: (_, k) => (k === '__esModule' ? true : (mockSrc[k] ??= jest.fn(async () => { throw new Error(`${String(k)} down`); }))) }));
jest.mock('../bankruptcies', () => ({ resolveBankruptcies: jest.fn(async () => { throw new Error('down'); }) }));
jest.mock('../sheetLkg', () => ({ fetchSheetLkg: jest.fn(async () => null) }));
jest.mock('../kv', () => ({ ...jest.requireActual('../kv'), defaultKv: { get: jest.fn(async () => null), read: jest.fn(), set: jest.fn(async () => true) } }));
// BLS answers: a fresh monthly UNRATE history (oldest first, as parseBlsSeries returns).
jest.mock('../horsemen', () => ({
    ...jest.requireActual('../horsemen'),
    parseBlsSeries: jest.fn(() => Array.from({ length: 24 }, (_, i) => ({ date: new Date(Date.now() - (24 - i) * 30 * 864e5).toISOString().slice(0, 10), value: 4.1 }))),
    parseDolClaimsXml: jest.fn(() => Array.from({ length: 10 }, (_, i) => ({ date: new Date(Date.now() - (10 - i) * 7 * 864e5).toISOString().slice(0, 10), value: 220000 }))),
}));

const { fetchJson } = require('../fetcher');
const { defaultKv } = require('../kv');
const { GET } = require('../../app/api/fred/route');

const get = (fail = 'none') => GET(new Request(`https://x.test/api/fred?_fail=${fail}`, { headers: { 'user-agent': 'jest' } }));

beforeAll(() => { process.env.FRED_API_KEY = 'k'; });
afterAll(() => { delete process.env.FRED_API_KEY; });
beforeEach(() => {
    jest.clearAllMocks();
    mockSrc.blsSeries = jest.fn(async () => ({}));
    mockSrc.dolWeeklyClaimsXml = jest.fn(async () => '<x/>');
    defaultKv.get.mockResolvedValue(null);
});

test('B1: FRED down on a cold instance, no /tmp, no Sheet → the KV last-good is the merge base (other cards keep values)', async () => {
    fetchJson.mockRejectedValue(new Error('FRED 503'));
    const kvPayload = {
        checklist: { nfci: { value: -0.5, asOf: day(5) } },
        profitMargin: { value: 12, current: 12 },
        indicators: { sahmRule: { value: 0.2 } },
        horsemen: {},
        _meta: { source: 'St. Louis Fed', loadedCount: 17, messages: ['Loaded 17/17 series'] },
    };
    defaultKv.get.mockResolvedValue(JSON.stringify({ data: kvPayload, savedAt: new Date(Date.now() - 3600e3).toISOString() }));
    const b = await (await get()).json();
    expect(b.checklist.nfci.value).toBe(-0.5);
    expect(b.profitMargin.value).toBe(12);
    expect(b.horsemen.unemployment.source).toBe('bls');
    expect(b._meta.stale).toBe(true);
    expect(b._meta.source).toMatch(/KV last-good/);
});

test('B1: FRED down and no base anywhere → full payload shape (cards N/A, labelled), never a horsemen-only object', async () => {
    fetchJson.mockRejectedValue(new Error('FRED 503'));
    const b = await (await get()).json();
    expect(b.checklist.nfci.unavailable).toBe(true);
    expect(b.indicators.sahmRule).toBeDefined();
    expect(b.horsemen.unemployment.source).toBe('bls');
    expect(b._meta.hasErrors).toBe(true);
});
