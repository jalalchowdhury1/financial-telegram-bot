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

test('B2: FRED hanging (each call times out at 8 s) no longer eats the Horsemen repair budget', async () => {
    jest.useFakeTimers({ doNotFake: ['nextTick', 'queueMicrotask', 'setImmediate'] });
    try {
        fetchJson.mockImplementation(() => new Promise((_, rej) => setTimeout(() => rej(new Error('timeout 8000ms')), 8000)));
        mockSrc.blsSeries = jest.fn(() => new Promise((r) => setTimeout(() => r({}), 3000))); // BLS takes 3 s
        const p = get();
        await jest.advanceTimersByTimeAsync(60000);
        const b = await (await p).json();
        expect(b.horsemen.unemployment.source).toBe('bls');
        expect(b._meta.messages.join(' | ')).not.toMatch(/Horsemen repair skipped/);
    } finally { jest.useRealTimers(); }
});

// FRED answers every series except the ones named (those throw like a 500).
const fredExcept = (...down) => fetchJson.mockImplementation(async (url) => {
    const id = /series_id=([A-Z0-9]+)/.exec(url)[1];
    if (down.includes(id)) throw new Error(`${id} 500`);
    return fredObs(30, 30);
});

test('B5: FRED ICSA failing + DOL ok → indicators.claims is recomputed from the repaired line', async () => {
    fredExcept('ICSA');
    const b = await (await get()).json();
    expect(b.horsemen.claims.source).toBe('dol');
    expect(b.indicators.claims.value).toBe(220);
});

test('B5: one failed FRED series is filled from the KV last-good, flagged stale (not N/A)', async () => {
    fredExcept('NFCI');
    const kvPayload = { checklist: { nfci: { value: -0.4, asOf: day(9), stale: false, unavailable: false, label: 'Financial Conditions' } }, _meta: { loadedCount: 17 } };
    defaultKv.get.mockResolvedValue(JSON.stringify({ data: kvPayload, savedAt: new Date(Date.now() - 3600e3).toISOString() }));
    const b = await (await get()).json();
    expect(b.checklist.nfci).toMatchObject({ value: -0.4, stale: true, asOf: day(9) });
    expect(b.checklist.m2.unavailable).toBe(false);   // healthy series stay live
    expect(b._meta.hasErrors).toBe(true);
});

test('B5: total FRED outage merged over a cached base → the claims indicator comes from the live DOL line', async () => {
    fetchJson.mockRejectedValue(new Error('FRED 503'));
    defaultKv.get.mockResolvedValue(JSON.stringify({ data: { indicators: { claims: { value: 999 } }, _meta: {} }, savedAt: new Date().toISOString() }));
    const b = await (await get()).json();
    expect(b.indicators.claims.value).toBe(220);
});

test('B10: 4 housing points → no 6-month average (not 0, which made any print > 1300 "bullish")', async () => {
    fetchJson.mockImplementation(async (url) => (/series_id=HOUST/.test(url) ? fredObs(4, 30, () => 1350) : fredObs(30, 30)));
    const b = await (await get()).json();
    expect(b.checklist.housing.value).toBe(1350);
    expect(b.checklist.housing.bullish).toBe(false);
});

test('B10: M2 with only 5 monthly points → no year-ago match (nearest point 4 months off is not "a year ago")', async () => {
    fetchJson.mockImplementation(async (url) => (/series_id=M2SL/.test(url) ? fredObs(5, 30, (i) => 100 - i) : fredObs(30, 30)));
    const b = await (await get()).json();
    expect(b.checklist.m2.value).toBeNull();
    expect(b.checklist.retail.value).not.toBeNull();   // a real 3-month match still works
});
