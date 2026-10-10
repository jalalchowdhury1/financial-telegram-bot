/**
 * @jest-environment node
 *
 * /api/market-extra backup layers: the keyless tiers in lib/marketExtraTiers.js
 * (Treasury 10Y/2Y, Freddie Mac PMMS, CNBC oil + real DXY, Fawaz USD/BDT), the
 * `?_fail=` switches that prove each one, and the per-metric last-known-good merge
 * (the bug: a lone gold-api print passed isGood -> 12 blanks instead of yesterday).
 */

// In-memory /tmp last-good store (the real lib/store.js runs on top of it).
jest.mock('fs', () => {
    const actual = jest.requireActual('fs');
    const mem = new Map();
    const lg = (p) => String(p).startsWith('/tmp/lg-');
    return {
        ...actual,
        __mem: mem,
        existsSync: (p) => (lg(p) ? mem.has(p) : actual.existsSync(p)),
        readFileSync: (p, ...a) => (lg(p) ? mem.get(p) : actual.readFileSync(p, ...a)),
        writeFileSync: (p, d, ...a) => (lg(p) ? void mem.set(p, d) : actual.writeFileSync(p, d, ...a)),
    };
});

// In-memory Upstash KV (lib/kv.js defaultKv) so the durable `ftb:lg:market-extra` tier is real.
jest.mock('../kv', () => {
    const actual = jest.requireActual('../kv');
    const m = new Map();
    return {
        ...actual,
        __m: m,
        defaultKv: {
            get: jest.fn(async (k) => (m.has(k) ? JSON.stringify(m.get(k)) : null)),
            set: jest.fn(async (k, v) => { m.set(k, JSON.parse(JSON.stringify(v))); return true; }),
        },
    };
});

const NOW = new Date('2026-10-09T20:00:00Z');
const down = new Set();     // hosts/keywords whose calls fail
const seen = [];

const TREASURY_2026 = [
    'Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"',
    '10/09/2026,4.13,4.13,4.13,4.25,4.29,4.32,4.47,4.80,4.89,5.02,5.13,5.24,5.65,5.60',
    '10/08/2026,4.14,4.14,4.13,4.23,4.29,4.30,4.44,4.75,4.85,4.99,5.11,5.22,5.64,5.60',
].join('\n');
const TREASURY_2025 = [
    'Date,"1 Mo","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"',
    '12/31/2025,4.0,4.0,4.0,4.0,4.0,4.0,4.10,4.2,4.3,4.4,4.50,4.9,4.8',
].join('\n');
const PMMS = 'date,pmms30,pmms30p,pmms15,pmms15p,pmms51,pmms51p,pmms51m,pmms51spread\n4/2/1971,7.33, ,,,,,,\n9/24/2026,7.03,,6.42,,,,,\n10/1/2026,7.28,,6.6,,,,,\n10/8/2026,7.4,,6.73,,,,,\n';
const QUOTES = { QuickQuoteResult: { QuickQuote: [
    { symbol: '@CL.1', last: '91.66', change: '0.17', change_pct: '0.1858', last_time: '2026-10-09T16:59:58.000-0400' },
    { symbol: '.DXY', last: '102.231', change: '0.00', change_pct: '0.00', last_time: '2026-10-09' },
] } };
const bars = (rows) => ({ barData: { priceBars: rows.map(([d, c]) => ({ tradeTime: `${d.replace(/-/g, '')}000000`, close: String(c) })) } });
const CL_BARS = bars([['2026-10-07', 88.28], ['2026-10-08', 91.49]]);          // lags the quote a session
const DXY_BARS = bars([['2026-10-08', 102.115], ['2026-10-09', 102.231]]);
const ER = { result: 'success', rates: { CAD: 1.37, INR: 84.1, BDT: 123.22, EUR: 0.86, JPY: 150, GBP: 0.75, SEK: 10.5, CHF: 0.8 } };
const FAWAZ = { date: '2026-10-09', usd: { bdt: 123.05, cad: 1.371, inr: 84.0, eur: 0.861, jpy: 150.1, gbp: 0.751, sek: 10.51, chf: 0.801 } };
const fredObs = (v) => ({ observations: [{ date: '2026-10-08', value: String(v) }, { date: '2026-10-07', value: String(v - 0.1) }] });

const hit = (url, ...keys) => keys.some((k) => url.includes(k));
const failIf = (url, key) => { if (down.has(key)) throw new Error(`HTTP 404 for ${url}`); }; // not "fetch failed" (withRetry would back off and retry)

jest.mock('../fetcher', () => ({
    fetchJson: jest.fn(async (url) => {
        seen.push(url);
        if (hit(url, 'open.er-api.com')) { failIf(url, 'erapi'); return ER; }
        if (hit(url, 'frankfurter')) { failIf(url, 'frankfurter'); return { rates: { CAD: 1.37, INR: 84.1, EUR: 0.86, JPY: 150, GBP: 0.75, SEK: 10.5, CHF: 0.8 } }; }
        if (hit(url, 'cdn.jsdelivr.net')) { failIf(url, 'fawaz'); failIf(url, 'jsdelivr'); return FAWAZ; }
        if (hit(url, 'currency-api.pages.dev')) { failIf(url, 'fawaz'); if (!url.includes('latest.currency-api')) throw new Error('bad mirror'); return FAWAZ; }
        if (hit(url, 'api.stlouisfed.org')) {
            failIf(url, 'fred');
            if (/api_key=&/.test(url)) throw new Error(`HTTP 400 for ${url}`);
            const id = /series_id=([A-Z0-9]+)/.exec(url)[1];
            return fredObs({ DGS10: 5.22, DGS2: 4.75, MORTGAGE30US: 7.28, DCOILWTICO: 90.5 }[id] ?? 1);
        }
        if (hit(url, 'gold-api.com')) { failIf(url, 'goldapi'); return { price: 2650.5, updatedAt: '2026-10-09T19:00:00Z' }; }
        if (hit(url, 'coinbase')) { failIf(url, 'btc'); return { data: { amount: '62000' } }; }
        if (hit(url, 'coingecko')) { failIf(url, 'btc'); return { bitcoin: { usd: 62000, usd_24h_change: 1 } }; }
        if (hit(url, 'kraken')) { failIf(url, 'btc'); return { result: { XXBTZUSD: { c: ['62000'] } } }; }
        if (hit(url, 'quote.cnbc.com')) { failIf(url, 'cnbc'); return QUOTES; }
        if (hit(url, 'ts-api.cnbc.com')) { failIf(url, 'cnbc'); return url.includes('DXY') ? DXY_BARS : CL_BARS; }
        throw new Error(`unexpected ${url}`);
    }),
    fetchText: jest.fn(async (url) => {
        seen.push(url);
        if (hit(url, 'home.treasury.gov')) { failIf(url, 'treasury'); return url.includes('/2026/') ? TREASURY_2026 : TREASURY_2025; }
        if (hit(url, 'freddiemac.com')) { failIf(url, 'pmms'); return PMMS; }
        throw new Error(`unexpected ${url}`);
    }),
}));

const fs = require('fs');
const kvMod = require('../kv');
const flush = () => new Promise((r) => setImmediate(r)); // let serve()'s background KV write settle
const tiers = require('../marketExtraTiers');
const { GET } = require('../../app/api/market-extra/route');

const req = (q = '') => new Request(`https://x.test/api/market-extra${q}`, { headers: { 'user-agent': 'jest' } });
const get = async (q) => (await GET(req(q))).json();
const ALL_DOWN = ['erapi', 'frankfurter', 'fawaz', 'fred', 'btc', 'cnbc', 'treasury', 'pmms'];

beforeAll(() => jest.useFakeTimers({ now: NOW, doNotFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'setImmediate', 'nextTick', 'queueMicrotask'] }));
afterAll(() => jest.useRealTimers());
beforeEach(() => { down.clear(); seen.length = 0; fs.__mem.clear(); kvMod.__m.clear(); kvMod.defaultKv.set.mockClear(); delete process.env.LAMBDA_URL; delete process.env.POLYGON_KEY; process.env.FRED_API_KEY = 'k'; });

describe('parsers + freshness', () => {
    test('Treasury: 2 Yr / 10 Yr by header name (never "20 Yr"), ascending across years', () => {
        const t = tiers.parseTreasuryTenors(TREASURY_2025, TREASURY_2026);
        expect(t.tnx).toEqual([{ date: '2025-12-31', price: 4.5 }, { date: '2026-10-08', price: 5.22 }, { date: '2026-10-09', price: 5.24 }]);
        expect(t.t2y.map((p) => p.price)).toEqual([4.1, 4.75, 4.8]);
    });

    test('PMMS: weekly 30Y, ascending, blank cells skipped', () => {
        const s = tiers.parsePmmsCsv(PMMS);
        expect(s[0]).toEqual({ date: '1971-04-02', price: 7.33 });
        expect(s[s.length - 1]).toEqual({ date: '2026-10-08', price: 7.4 });
    });

    test('seriesMetric refuses a print older than its freshness window (no stale-as-live)', () => {
        const asc = [{ date: '2026-09-20', price: 5 }, { date: '2026-09-21', price: 5.1 }];
        expect(tiers.seriesMetric(asc, { freshDays: 6, now: NOW.getTime() })).toBeNull();
        expect(tiers.seriesMetric(asc, { freshDays: 30, now: NOW.getTime() })).toMatchObject({ current: 5.1, lastDate: '2026-09-21' });
    });

    test('CNBC: quote appended as the last bar; change from the prior bar, not CNBC\'s 0.00 field', () => {
        const m = tiers.cnbcMetricFrom({ price: 91.66, asOf: '2026-10-09' }, [{ date: '2026-10-07', price: 88.28 }, { date: '2026-10-08', price: 91.49 }], { now: NOW.getTime() });
        expect(m.current).toBe(91.66);
        expect(m.lastDate).toBe('2026-10-09');
        expect(m.history[m.history.length - 1]).toEqual({ date: '2026-10-09', price: 91.66 });
        expect(m.dailyChange.value).toBeCloseTo(0.17, 6);
    });

    test('Fawaz: a stale date is refused', () => {
        expect(tiers.fawazPair(FAWAZ, 'bdt', { now: NOW.getTime() })).toEqual({ current: 123.05, asOf: '2026-10-09' });
        expect(() => tiers.fawazPair({ ...FAWAZ, date: '2026-09-01' }, 'bdt', { now: NOW.getTime() })).toThrow(/stale/);
    });
});

describe('mergeLastGood', () => {
    const save = (data, savedAt) => fs.__mem.set('/tmp/lg-market-extra.json', JSON.stringify({ data, savedAt }));
    const paths = ['fx.usdbdt', 'rates.tnx', 'commodities.gc'];

    test('fills only missing metrics, stamped stale + savedAt; never overwrites live', async () => {
        save({ fx: { usdbdt: { current: 120 } }, rates: { tnx: { current: 5 } }, commodities: { gc: { current: 1 } } }, '2026-10-08T12:00:00.000Z');
        const out = { fx: {}, rates: {}, commodities: { gc: { current: 2650 } }, _meta: { sourceLog: {} } };
        const filled = await tiers.mergeLastGood(out, { key: 'market-extra', paths, now: NOW.getTime() });
        expect(filled).toEqual(['usdbdt', 'tnx']);
        expect(out.fx.usdbdt).toEqual({ current: 120, stale: true, savedAt: '2026-10-08T12:00:00.000Z' });
        expect(out.commodities.gc).toEqual({ current: 2650 });
        expect(out._meta.sourceLog.tnx).toBe('last-known-good 2026-10-08T12:00:00.000Z');
    });

    test('a borrowed metric keeps its ORIGINAL savedAt through a re-save; >7 days is dropped', async () => {
        save({ fx: { usdbdt: { current: 120, stale: true, savedAt: '2026-10-05T00:00:00.000Z' } }, rates: { tnx: { current: 5, stale: true, savedAt: '2026-09-30T00:00:00.000Z' } } }, '2026-10-09T10:00:00.000Z');
        const out = { fx: {}, rates: {} };
        expect(await tiers.mergeLastGood(out, { key: 'market-extra', paths, now: NOW.getTime() })).toEqual(['usdbdt']);
        expect(out.fx.usdbdt.savedAt).toBe('2026-10-05T00:00:00.000Z');
        expect(out.rates.tnx).toBeUndefined();
    });

    test('never fills in fault-test mode with `lastgood`', async () => {
        save({ fx: { usdbdt: { current: 120 } } }, '2026-10-09T10:00:00.000Z');
        const out = { fx: {} };
        expect(await tiers.mergeLastGood(out, { key: 'market-extra', paths, faults: new Set(['lastgood']), now: NOW.getTime() })).toEqual([]);
        expect(out.fx.usdbdt).toBeUndefined();
    });

    const kvSeed = (data, savedAt) => kvMod.__m.set('ftb:lg:market-extra', { data, savedAt });

    test('cold instance (/tmp empty) -> fills from the KV copy, stale + original savedAt', async () => {
        kvSeed({ fx: { usdbdt: { current: 121 } }, rates: { tnx: { current: 5.1, stale: true, savedAt: '2026-10-07T00:00:00.000Z' } } }, '2026-10-08T15:00:00.000Z');
        const out = { fx: {}, rates: {}, commodities: { gc: { current: 2650 } } };
        expect(await tiers.mergeLastGood(out, { key: 'market-extra', paths, now: NOW.getTime() })).toEqual(['usdbdt', 'tnx']);
        expect(out.fx.usdbdt).toEqual({ current: 121, stale: true, savedAt: '2026-10-08T15:00:00.000Z' });
        expect(out.rates.tnx.savedAt).toBe('2026-10-07T00:00:00.000Z');
        expect(out._meta.sourceLog.usdbdt).toBe('KV last-known-good 2026-10-08T15:00:00.000Z');
        expect(out.commodities.gc).toEqual({ current: 2650 });
    });

    test('/tmp first; KV only for what /tmp lacks', async () => {
        save({ fx: { usdbdt: { current: 120 } } }, '2026-10-09T10:00:00.000Z');
        kvSeed({ fx: { usdbdt: { current: 999 } }, rates: { tnx: { current: 5.1 } } }, '2026-10-08T15:00:00.000Z');
        const out = { fx: {}, rates: {} };
        expect(await tiers.mergeLastGood(out, { key: 'market-extra', paths, now: NOW.getTime() })).toEqual(['usdbdt', 'tnx']);
        expect(out.fx.usdbdt.current).toBe(120);
        expect(out._meta.sourceLog.tnx).toBe('KV last-known-good 2026-10-08T15:00:00.000Z');
    });

    test('`kvlg` disables only the KV copy; `lastgood` both; KV past maxStaleMs ignored', async () => {
        kvSeed({ fx: { usdbdt: { current: 121 } } }, '2026-10-08T15:00:00.000Z');
        expect(await tiers.mergeLastGood({ fx: {} }, { key: 'market-extra', paths, faults: new Set(['kvlg']), now: NOW.getTime() })).toEqual([]);
        expect(await tiers.mergeLastGood({ fx: {} }, { key: 'market-extra', paths, faults: new Set(['lastgood']), now: NOW.getTime() })).toEqual([]);
        kvSeed({ fx: { usdbdt: { current: 121 } } }, '2026-09-01T00:00:00.000Z');
        expect(await tiers.mergeLastGood({ fx: {} }, { key: 'market-extra', paths, now: NOW.getTime() })).toEqual([]);
    });

    test('a throwing KV client cannot break the merge', async () => {
        const kv = { get: async () => { throw new Error('kv down'); } };
        expect(await tiers.mergeLastGood({ fx: {} }, { key: 'market-extra', paths, kv, now: NOW.getTime() })).toEqual([]);
    });
});

describe('/api/market-extra direct tiers (Lambda down)', () => {
    test('healthy: real DXY from CNBC, FRED rates, full build, no stale', async () => {
        const b = await get('?_fail=lambda');
        expect(b._meta.sourceLog).toMatchObject({ dxy: 'CNBC', tnx: 'US Treasury', t2y: 'US Treasury', mortgageRate: 'Freddie Mac PMMS', cl: 'CNBC', usdbdt: 'ER-API', gc: 'gold-api' });
        expect(b.fx.dxy.current).toBe(102.231);
        expect(b._meta.stale).toBeUndefined();
        expect(b._meta.hasErrors).toBe(false);
    });

    test('FRED dead: 10Y/2Y from Treasury, 30Y from Freddie Mac, oil from CNBC', async () => {
        const b = await get('?_fail=lambda,fred');
        expect(b._meta.sourceLog).toMatchObject({ tnx: 'US Treasury', t2y: 'US Treasury', mortgageRate: 'Freddie Mac PMMS', cl: 'CNBC' });
        expect(b.rates.tnx).toMatchObject({ current: 5.24, lastDate: '2026-10-09' });
        expect(b.rates.t2y.current).toBe(4.8);
        expect(b.rates.mortgageRate).toMatchObject({ current: 7.4, lastDate: '2026-10-08' });
        expect(b.commodities.cl).toMatchObject({ current: 91.66, lastDate: '2026-10-09' });
        // 10Y + 2Y share ONE Treasury download per year.
        expect(seen.filter((u) => u.includes('home.treasury.gov') && u.includes('/2026/'))).toHaveLength(1);
    });

    test.each([
        ['treasury', ['tnx', 't2y']],
        ['pmms', ['mortgageRate']],
        ['cnbc_cl', ['cl']],
    ])('?_fail=lambda,fred,%s,lastgood blanks exactly those metrics', async (name, gone) => {
        const b = await get(`?_fail=lambda,fred,${name},lastgood`);
        for (const k of gone) expect(b._meta.sourceLog[k]).toBeUndefined();
        expect(b._meta.hasErrors).toBe(true);
    });

    test('DXY: CNBC -> computed basket (labelled) -> blank', async () => {
        expect((await get('?_fail=lambda,cnbc_dxy'))._meta.sourceLog.dxy).toBe('computed (FX basket)');
        expect((await get('?_fail=lambda,cnbc'))._meta.sourceLog.dxy).toBe('computed (FX basket)');
        const b = await get('?_fail=lambda,cnbc,dxy_computed,lastgood');
        expect(b.fx.dxy).toBeUndefined();
    });

    test('USD/BDT: ER-API dead -> Frankfurter has no BDT -> Fawaz fills it', async () => {
        const b = await get('?_fail=lambda,erapi');
        expect(b._meta.sourceLog.usdbdt).toBe('Fawaz');
        expect(b.fx.usdbdt).toMatchObject({ current: 123.05, lastDate: '2026-10-09' });
        expect(b.fx.cadbdt.current).toBeCloseTo(123.05 / 1.37, 6);
        expect((await get('?_fail=lambda,erapi,fawaz_bdt,lastgood')).fx.usdbdt).toBeUndefined();
    });

    test('Fawaz mirror: jsDelivr down -> latest.currency-api.pages.dev', async () => {
        down.add('jsdelivr');
        const b = await get('?_fail=lambda,erapi');
        expect(b.fx.usdbdt.current).toBe(123.05);
        expect(seen.some((u) => u.startsWith('https://latest.currency-api.pages.dev/'))).toBe(true);
    });

    test('gold-api has a switch: ?_fail=gold_api', async () => {
        const b = await get('?_fail=lambda,gold_api,lastgood');
        expect(b.commodities.gc).toBeUndefined();
    });

    test('fault-test mode never writes the last-good store', async () => {
        await get('?_fail=lambda,fred');
        await flush();
        expect(fs.__mem.size).toBe(0);
        expect(kvMod.defaultKv.set).not.toHaveBeenCalled();
    });
});

describe('per-metric last-known-good (the 12-blanks bug)', () => {
    test('only gold-api alive -> gold live + every other metric from last good, flagged stale', async () => {
        const first = await get();                 // healthy, non-test: saves last good
        expect(first._meta.stale).toBeUndefined();
        expect(fs.__mem.has('/tmp/lg-market-extra.json')).toBe(true);
        const savedAt = JSON.parse(fs.__mem.get('/tmp/lg-market-extra.json')).savedAt;

        ALL_DOWN.forEach((k) => down.add(k));
        const b = await get();
        expect(b._meta.sourceLog.gc).toBe('gold-api');
        expect(b.commodities.gc.stale).toBeUndefined();
        for (const [grp, k] of [['fx', 'usdbdt'], ['fx', 'dxy'], ['fx', 'usdcad'], ['fx', 'cadbdt'], ['rates', 'tnx'], ['rates', 't2y'], ['rates', 'mortgageRate'], ['commodities', 'cl'], ['commodities', 'btc']]) {
            expect(b[grp][k]).toMatchObject({ stale: true, savedAt, current: first[grp][k].current });
            expect(b._meta.sourceLog[k]).toBe(`last-known-good ${savedAt}`);
        }
        expect(b._meta.stale).toBe(true);
        expect(b._meta.hasErrors).toBe(true);
        expect(b._meta.messages.some((m) => /^unavailable:/.test(m))).toBe(false);
        expect(b._meta.messages.some((m) => /Stale \(last-known-good\)/.test(m))).toBe(true);
        // degraded -> never edge-cached
        const res = await GET(req());
        expect(res.headers.get('vercel-cdn-cache-control')).toBeNull();

        // The re-saved copy keeps each borrowed metric's ORIGINAL savedAt.
        const stored = JSON.parse(fs.__mem.get('/tmp/lg-market-extra.json'));
        expect(stored.data.rates.tnx.savedAt).toBe(savedAt);
    });

    test('cold instance + only gold-api alive -> every other metric from the KV copy (not 12 blanks)', async () => {
        const first = await get();
        await flush();
        const kvCopy = kvMod.__m.get('ftb:lg:market-extra');
        expect(kvCopy.data.rates.tnx.current).toBe(first.rates.tnx.current);
        fs.__mem.clear();                          // new instance: /tmp is empty
        ALL_DOWN.forEach((k) => down.add(k));
        const b = await get();
        expect(b._meta.sourceLog.gc).toBe('gold-api');
        for (const [grp, k] of [['fx', 'usdbdt'], ['fx', 'dxy'], ['rates', 'tnx'], ['rates', 'mortgageRate'], ['commodities', 'cl'], ['commodities', 'btc']]) {
            expect(b[grp][k]).toMatchObject({ stale: true, savedAt: kvCopy.savedAt, current: first[grp][k].current });
            expect(b._meta.sourceLog[k]).toBe(`KV last-known-good ${kvCopy.savedAt}`);
        }
        expect(b._meta.stale).toBe(true);
        expect(b._meta.hasErrors).toBe(true);
        // ...and `?_fail=kvlg` proves the KV step alone: without it those metrics are blank.
        fs.__mem.clear();
        const noKv = await get('?_fail=kvlg');
        expect(noKv.rates.tnx).toBeUndefined();
    });

    test('a partial run never overwrites the full KV copy', async () => {
        await get();
        await flush();
        const full = JSON.stringify(kvMod.__m.get('ftb:lg:market-extra'));
        kvMod.defaultKv.set.mockClear();
        fs.__mem.clear();                          // no /tmp copy -> nothing to fill from /tmp
        kvMod.__m.clear();                         // ...nor KV: the partial run stays partial (hasErrors)
        ALL_DOWN.forEach((k) => down.add(k));
        const partial = await get();
        await flush();
        expect(partial._meta.hasErrors).toBe(true);
        expect(kvMod.defaultKv.set).not.toHaveBeenCalled();
        // and with the full copy in KV, the stale-filled run does not rewrite it either
        kvMod.__m.set('ftb:lg:market-extra', JSON.parse(full));
        fs.__mem.clear();
        await get();
        await flush();
        expect(kvMod.defaultKv.set).not.toHaveBeenCalled();
        expect(JSON.stringify(kvMod.__m.get('ftb:lg:market-extra'))).toBe(full);
    });

    test('Lambda up but missing real-estate prints -> filled stale from last good', async () => {
        const re = { current: 2100, dailyChange: { value: 0, pct: 0 }, history: [], lastDate: '2026-09-01' };
        fs.__mem.set('/tmp/lg-market-extra.json', JSON.stringify({ data: { realEstate: { rentIndex: re, atnhpi: { ...re, current: 400 } } }, savedAt: '2026-10-08T00:00:00.000Z' }));
        process.env.LAMBDA_URL = 'https://lambda.test';
        const realFetch = global.fetch;
        const m =(v) => ({ current: v, dailyChange: { value: 0, pct: 0 }, history: [], lastDate: '2026-10-09' });
        global.fetch = jest.fn(async () => ({ ok: true, json: async () => ({
            fx: { usdcad: m(1.37), usdinr: m(84.1), usdbdt: m(123.2), dxy: m(102.2) },
            commodities: { gc: m(2650), cl: m(91.6), btc: m(62000) },
            rates: { tnx: m(5.24), t2y: m(4.8), mortgageRate: m(7.4) },
            realEstate: { mortgagePayment: m(2500) },
            _meta: { source: 'Lambda', hasErrors: true, messages: ['unavailable: 2 metrics'] },
        }) }));
        try {
            const b = await get('?_fail=polygon'); // test mode: reads last good, never writes
            expect(b.realEstate.mortgagePayment.stale).toBeUndefined();
            expect(b.realEstate.rentIndex).toMatchObject({ current: 2100, stale: true, savedAt: '2026-10-08T00:00:00.000Z' });
            expect(b.realEstate.atnhpi.current).toBe(400);
            expect(b._meta.staleMetrics).toEqual(['rentIndex', 'atnhpi']);
            expect(b._meta.messages).not.toContain('unavailable: 2 metrics');
        } finally { global.fetch = realFetch; }
    });
});

describe('/api/market-extra timing budget', () => {
    test('Lambda hop carries a <=12 s abort signal and runs in parallel with the FX rates; a timeout degrades to direct', async () => {
        process.env.LAMBDA_URL = 'https://lambda.test';
        const realFetch = global.fetch;
        const timeouts = [];
        const spy = jest.spyOn(AbortSignal, 'timeout').mockImplementation((ms) => {
            timeouts.push(ms);
            const c = new AbortController();
            setTimeout(() => c.abort(new DOMException('timed out', 'TimeoutError')), 5);
            return c.signal;
        });
        let erStartedBeforeLambdaSettled = false, lambdaSettled = false;
        const fetcher = require('../fetcher');
        const origJson = fetcher.fetchJson.getMockImplementation();
        fetcher.fetchJson.mockImplementation(async (url, ...a) => {
            if (url.includes('open.er-api.com') && !lambdaSettled) erStartedBeforeLambdaSettled = true;
            return origJson(url, ...a);
        });
        global.fetch = jest.fn((url, init) => new Promise((_, rej) => {
            init.signal.addEventListener('abort', () => { lambdaSettled = true; rej(init.signal.reason); });
        }));
        try {
            const b = await get();
            expect(timeouts).toHaveLength(1);
            expect(timeouts[0]).toBeLessThanOrEqual(12000);
            expect(erStartedBeforeLambdaSettled).toBe(true);
            expect(b._meta.messages.join(' ')).toMatch(/Lambda timed out after 12 s/);
            expect(b._meta.sourceLog.tnx).toBe('US Treasury');
        } finally {
            global.fetch = realFetch; spy.mockRestore(); fetcher.fetchJson.mockImplementation(origJson);
        }
    });

    test('the route keeps maxDuration 30 and its comment matches the budget', () => {
        const src = require('fs').readFileSync(require('path').join(__dirname, '../../app/api/market-extra/route.js'), 'utf8');
        expect(src).toMatch(/export const maxDuration = 30;/);
        const n = (name) => Number(src.match(new RegExp(`const ${name} = (\\d+);`))[1]);
        // fill deadline + the KV GET (3 s) + headroom must fit in 30 s
        expect(n('FILL_DEADLINE_MS') + 3000 + 2000).toBeLessThanOrEqual(30000);
        expect(n('LAMBDA_TIMEOUT_MS')).toBeLessThan(n('FILL_DEADLINE_MS'));
    });
});


describe('staleIfOld: a lagging FRED copy is flagged, a fresh one passes', () => {
    const { staleIfOld } = require('../../app/api/market-extra/route');
    const NOW = Date.parse('2026-10-09T22:00:00Z');
    test('the live case: Oct 6 oil seen Oct 9 is flagged at oil\'s 3-day limit; Oct 8 yields pass at 4', () => {
        expect(staleIfOld({ current: 96.24, lastDate: '2026-10-06' }, 3, NOW).stale).toBe(true);
        expect(staleIfOld({ current: 5.22, lastDate: '2026-10-08' }, 4, NOW).stale).toBeUndefined();
        const old = staleIfOld({ current: 90, lastDate: '2026-10-02' }, 4, NOW);
        expect(old.stale).toBe(true);
        expect(old.savedAt).toBe('2026-10-02T12:00:00Z');
    });
});
