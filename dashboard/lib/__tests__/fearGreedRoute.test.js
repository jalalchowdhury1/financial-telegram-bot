/**
 * @jest-environment node
 *
 * /api/fear-greed layers, each behind a `?_fail=` switch:
 * cnn → rapidapi → fg_yahoo / fg_cboe / fg_fred (VIX proxies) → fg_cache (/tmp) → fg_kvlg (KV) → N/A.
 */

let mode;
const NOW = new Date('2026-10-09T22:00:00Z');
const day = (offset) => Math.floor((Date.parse('2026-10-09T20:00:00Z') - offset * 864e5) / 1000);
const okJson = (body) => ({ ok: true, status: 200, json: async () => body });

jest.mock('../fetcher', () => ({
    proxyFetch: jest.fn(async (url) => {
        if (url.includes('dataviz.cnn.io')) {
            if (mode.cnn === 'down') throw new Error('CNN 418');
            return okJson({ fear_and_greed: { score: 45, rating: 'fear', timestamp: mode.cnnTs || '2026-10-09T21:59:54+00:00', previous_close: 37.9, previous_1_week: 40, previous_1_month: 38.2, previous_1_year: 48.6 } });
        }
        if (url.includes('rapidapi')) {
            if (mode.rapid === 'down') throw new Error('RapidAPI 403');
            return okJson({ fgi: { now: { value: 44, valueText: 'Fear' }, previousClose: { value: 38 }, oneWeekAgo: { value: 40 }, oneMonthAgo: { value: 38 }, oneYearAgo: { value: 49 } } });
        }
        if (url.includes('finance.yahoo.com')) {
            if (mode.yahoo === 'down') throw new Error('Yahoo 429');
            return okJson({ chart: { result: [{ timestamp: [day(2), day(1), day(0)], indicators: { quote: [{ close: [15.08, 15.41, null] }] } }] } });
        }
        throw new Error(`offline ${url}`);
    }),
    fetchText: jest.fn(async (url) => {
        if (!url.includes('cboe.com') || mode.cboe === 'down') throw new Error('CBOE offline');
        const m = mode.cboeFrozen ? '08' : '10';
        const rows = ['DATE,OPEN,HIGH,LOW,CLOSE', `${m}/07/2026,0,0,0,15.08`, `${m}/08/2026,0,0,0,15.41`, `${m}/09/2026,0,0,0,14.84`];
        return rows.join('\n');
    }),
    fetchJson: jest.fn(async () => {
        if (mode.fred === 'down') throw new Error('FRED 429');
        return { observations: [{ date: '2026-10-08', value: '15.41' }, { date: '2026-10-07', value: '15.08' }] };
    }),
}));
jest.mock('../store', () => {
    const tmp = new Map();
    const kv = new Map();
    const fresh = (r, maxAge) => (!r || (maxAge && Date.now() - Date.parse(r.savedAt) > maxAge) ? null : r);
    return {
        __tmp: tmp,
        __kv: kv,
        loadLastGood: (k, maxAge) => fresh(tmp.get(k), maxAge),
        saveLastGood: (k, data) => tmp.set(k, { data, savedAt: new Date().toISOString() }),
        loadLastGoodKV: jest.fn(async (k, maxAge) => fresh(kv.get(k), maxAge)),
        saveLastGoodKV: jest.fn(async (k, data) => { kv.set(k, { data, savedAt: new Date().toISOString() }); return true; }),
    };
});

const store = require('../store');
const { GET } = require('../../app/api/fear-greed/route');
const call = async (q = '') => {
    const res = await GET(new Request(`https://x.test/api/fear-greed${q}`, { headers: { 'user-agent': 'jest' } }));
    return { res, b: await res.json() };
};

beforeAll(() => {
    process.env.RAPIDAPI_KEY = 'rk'; process.env.FRED_API_KEY = 'fk';
    jest.useFakeTimers({ now: NOW, doNotFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'setImmediate', 'nextTick', 'queueMicrotask'] });
});
afterAll(() => { jest.useRealTimers(); delete process.env.RAPIDAPI_KEY; delete process.env.FRED_API_KEY; });
beforeEach(() => {
    mode = {}; store.__tmp.clear(); store.__kv.clear();
    store.saveLastGoodKV.mockClear(); store.loadLastGoodKV.mockClear();
});

describe('/api/fear-greed', () => {
    test('CNN healthy: served, edge-cached, saved to /tmp + KV', async () => {
        const { res, b } = await call();
        expect(b).toMatchObject({ score: 45, rating: 'FEAR', _meta: { source: 'CNN', hasErrors: false } });
        expect(res.headers.get('vercel-cdn-cache-control')).toMatch(/max-age/);
        expect(store.__tmp.get('fear-greed').data.score).toBe(45);
        expect(store.saveLastGoodKV).toHaveBeenCalledWith('fear-greed', expect.objectContaining({ score: 45 }));
    });

    test('?_fail=cnn → RapidAPI (the CNN index via a reseller), saved as last-good', async () => {
        const { b } = await call('?_fail=cnn');
        expect(b.score).toBe(44);
        expect(b._meta.source).toBe('RapidAPI');
        expect(b._meta.messages.join(' ')).toMatch(/injected fault: cnn/);
    });

    test('?_fail=cnn,rapidapi → Yahoo VIX proxy, labelled as a proxy (not CNN), null bar skipped', async () => {
        const { b } = await call('?_fail=cnn,rapidapi');
        expect(b._meta.source).toBe('Yahoo ^VIX Proxy');
        expect(b._meta.source).not.toMatch(/CNN/);
        expect(b._meta.proxy).toBe(true);
        expect(b._meta.note).toMatch(/NOT the CNN Fear & Greed/);
        expect(b.score).toBeCloseTo(100 - ((15.41 - 10) / 25) * 100);
        expect(b.asOf).toBe('2026-10-08');
    });

    test('a proxy answer is never saved as last-good', async () => {
        mode = { cnn: 'down', rapid: 'down' };
        const { b } = await call();
        expect(b._meta.proxy).toBe(true);
        expect(store.__tmp.size).toBe(0);
        expect(store.saveLastGoodKV).not.toHaveBeenCalled();
    });

    test('?_fail=cnn,rapidapi,fg_yahoo → CBOE VIX proxy (same-day), labelled proxy, not stale', async () => {
        const { b } = await call('?_fail=cnn,rapidapi,fg_yahoo');
        expect(b._meta.source).toBe('CBOE VIX Proxy');
        expect(b._meta.proxy).toBe(true);
        expect(b._meta.stale).toBeUndefined();
        expect(b.asOf).toBe('2026-10-09');
        expect(b.score).toBeCloseTo(100 - ((14.84 - 10) / 25) * 100);
    });

    test('a frozen CBOE CSV is rejected → FRED', async () => {
        mode.cboeFrozen = true;
        const { b } = await call('?_fail=cnn,rapidapi,fg_yahoo');
        expect(b._meta.source).toBe('FRED VIXCLS Proxy');
        expect(b._meta.messages.join(' ')).toMatch(/CBOE VIX frozen/);
    });

    test('?_fail=cnn,rapidapi,fg_yahoo,fg_cboe → FRED VIXCLS proxy, flagged proxy + stale', async () => {
        const { b } = await call('?_fail=cnn,rapidapi,fg_yahoo,fg_cboe');
        expect(b._meta.source).toBe('FRED VIXCLS Proxy');
        expect(b._meta.proxy).toBe(true);
        expect(b._meta.stale).toBe(true);
        expect(b.asOf).toBe('2026-10-08');
    });

    test('all live layers off → /tmp copy of the real index, relabelled + stale', async () => {
        await call(); // healthy load saves it
        const { b } = await call('?_fail=cnn,rapidapi,fg_yahoo,fg_cboe,fg_fred');
        expect(b.score).toBe(45);
        expect(b._meta.source).toMatch(/^Stale cache \(2026-10-09T22:00:00.000Z\) ← CNN$/);
        expect(b._meta.stale).toBe(true);
        expect(b._meta.lastGoodAt).toBe('2026-10-09T22:00:00.000Z');
    });

    test('?_fail=…,fg_cache → KV copy, labelled "KV last-good (<savedAt>) ← CNN"', async () => {
        await call();
        const { b } = await call('?_fail=cnn,rapidapi,fg_yahoo,fg_cboe,fg_fred,fg_cache');
        expect(b._meta.source).toBe('Stale KV last-good (2026-10-09T22:00:00.000Z) ← CNN');
        expect(b._meta.stale).toBe(true);
    });

    test('lastgood / kvlg / fg_kvlg switches reach the N/A floor (500 + error, no-store)', async () => {
        await call();
        for (const q of ['?_fail=cnn,rapidapi,fg_yahoo,fg_cboe,fg_fred,lastgood', '?_fail=cnn,rapidapi,fg_yahoo,fg_cboe,fg_fred,fg_cache,kvlg', '?_fail=cnn,rapidapi,fg_yahoo,fg_cboe,fg_fred,fg_cache,fg_kvlg']) {
            const { res, b } = await call(q);
            expect(res.status).toBe(500);
            expect(b.score).toBe('N/A');
            expect(b.error).toMatch(/unavailable/);
            expect(b._meta.source).toBe('Failed');
            expect(res.headers.get('cache-control')).toBe('no-store');
        }
    });

    test('fault test mode never writes /tmp or KV, even when CNN is healthy', async () => {
        const { b } = await call('?_fail=fg_yahoo');
        expect(b._meta.source).toBe('CNN');
        expect(store.__tmp.size).toBe(0);
        expect(store.saveLastGoodKV).not.toHaveBeenCalled();
    });

    test('a frozen CNN payload (timestamp a week old) is rejected, not served as live', async () => {
        mode.cnnTs = '2026-10-01T20:00:00+00:00';
        const { b } = await call();
        expect(b._meta.source).toBe('RapidAPI');
        expect(b._meta.messages.join(' ')).toMatch(/CNN frozen/);
    });

    test('a last-good copy older than 3 days is not served', async () => {
        await call();
        jest.setSystemTime(new Date(NOW.getTime() + 4 * 864e5));
        mode = { cnn: 'down', rapid: 'down', yahoo: 'down', cboe: 'down', fred: 'down' };
        const { b } = await call();
        expect(b._meta.source).toBe('Failed');
        jest.setSystemTime(NOW);
    });
});
