/**
 * @jest-environment node
 *
 * /api/sheets per-field cascade (lib/sheetsCascade.js + app/api/sheets/route.js):
 * sheet → alt → CBOE → FRED (stale) → /tmp last-good → KV last-good → 'N/A', every
 * tier behind a `?_fail=` switch, and the frozen sheet-C2 fear/greed tag never served.
 */

let mode;
const seen = [];
const day = (iso) => { const [y, m, d] = iso.split('-'); return `${m}/${d}/${y}`; };
// 60 daily closes ending `end`; the last one is `last` (so the 50d tag is computable).
function cboeCsv(last, end = '2026-10-09', base = 16) {
    const rows = [];
    const t = Date.parse(`${end}T12:00:00Z`);
    for (let i = 59; i >= 0; i--) {
        const iso = new Date(t - i * 864e5).toISOString().slice(0, 10);
        rows.push(`${day(iso)},0,0,0,${i === 0 ? last : base}`);
    }
    return ['DATE,OPEN,HIGH,LOW,CLOSE', ...rows].join('\n');
}

jest.mock('../data/aaiiNewest.json', () => ({}));
jest.mock('../fetcher', () => ({
    fetchText: jest.fn(async (url) => {
        seen.push(url);
        if (url.includes('docs.google.com')) {
            const alt = url.includes('output=csv');
            if (mode.sheets === 'down' || (alt && mode.alt === 'down') || (!alt && mode.main === 'down')) throw new Error('Fetch failed: 500');
            if (url.includes('10Y8Jus8')) return 'h1,h2\nx,y\nz,ON\n';
            if (url.includes('gid=1668420064')) return 'h\nBIL (T-Bill ETF)1\n';
            if (url.includes('gid=790638481')) return `VIX,VIX 3M,Fear and Greed\n${mode.vixCell || '14.84,17.77'},GREED13\n`;
        }
        if (url.includes('cboe.com')) {
            if (mode.cboe === 'down') throw new Error('CBOE offline');
            if (url.includes('VIX3M_History')) return cboeCsv(17.77, mode.cboeEnd);
            if (url.includes('VIX_History')) return cboeCsv(15.2, mode.cboeEnd);
        }
        throw new Error(`offline ${url}`);
    }),
    fetchJson: jest.fn(async (url) => {
        seen.push(url);
        if (mode.fred === 'down') throw new Error('FRED 429');
        if (url.includes('series_id=VXVCLS')) return { observations: [{ date: '2026-10-08', value: '18.08' }] };
        if (url.includes('series_id=VIXCLS')) return { observations: [{ date: '2026-10-08', value: '15.41' }, { date: '2026-10-07', value: '.' }] };
        throw new Error('offline');
    }),
}));
jest.mock('../store', () => {
    const m = new Map();
    return {
        __m: m,
        loadLastGood: (k) => m.get(k) || null,
        saveLastGood: (k, data) => m.set(k, { data, savedAt: new Date().toISOString() }),
    };
});
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

const store = require('../store');
const kv = require('../kv');
const { KV_KEY, TMP_KEY } = require('../sheetsCascade');
const { GET } = require('../../app/api/sheets/route');
// The KV SET runs in the background (lib/background.js); let it settle before asserting.
const get = async (q = '') => {
    const b = await (await GET(new Request(`https://x.test/api/sheets${q}`, { headers: { 'user-agent': 'jest' } }))).json();
    await new Promise((r) => setImmediate(r));
    return b;
};
const kvSets = () => kv.defaultKv.set.mock.calls.filter(([k]) => k === KV_KEY).length;

const NOW = new Date('2026-10-09T20:00:00Z');
beforeAll(() => {
    process.env.FRED_API_KEY = 'test-key';
    jest.useFakeTimers({ now: NOW, doNotFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'setImmediate', 'nextTick', 'queueMicrotask'] });
});
afterAll(() => { jest.useRealTimers(); delete process.env.FRED_API_KEY; });
beforeEach(() => {
    mode = {}; seen.length = 0; store.__m.clear(); kv.__m.clear();
    kv.defaultKv.set.mockClear(); kv.defaultKv.get.mockClear();
    jest.setSystemTime(NOW);
});

describe('/api/sheets healthy path', () => {
    test('all pills from the primary sheet; tag computed from CBOE, never the sheet C2', async () => {
        const b = await get();
        expect(b.NotSoBoring).toBe('ON');
        expect(b.FrontRunner).toBe('BIL (T-Bill ETF)');
        expect(b.VIX.current).toBe('14.84');
        expect(b.VIX.threeMonth).toBe('17.77');
        expect(b.VIX.fearGreed).not.toBe('GREED13');
        expect(b.VIX.fearGreed).toBe('GREED05'); // 15.2 vs a 50d mean of 15.98
        expect(b._meta.source).toBe('Google Sheets (Live)');
        expect(b._meta.stale).toBe(false);
        expect(b._meta.staleFields).toEqual([]);
        expect(b._meta.fields.vixFearGreed.source).toBe('CBOE-computed');
        expect(b._meta.messages[0]).toBe('Live data loaded');
    });

    test('live fields are saved per field to /tmp and KV', async () => {
        await get();
        const tmp = store.__m.get(TMP_KEY).data;
        expect(tmp.NotSoBoring).toMatchObject({ value: 'ON', source: 'Google Sheets (Live)' });
        expect(tmp.vixFearGreed.value).toBe('GREED05');
        const rec = kv.__m.get(KV_KEY);
        expect(rec.data.FrontRunner.value).toBe('BIL (T-Bill ETF)');
        expect(typeof rec.savedAt).toBe('string');
    });

    test('KV write is throttled: an unchanged second load does not SET again', async () => {
        await get(); await get();
        expect(kvSets()).toBe(1);
        jest.setSystemTime(new Date(NOW.getTime() + 31 * 60e3));
        await get();
        expect(kvSets()).toBe(1);                    // a last-good copy is refreshed at most hourly
        jest.setSystemTime(new Date(NOW.getTime() + 61 * 60e3));
        await get();
        await new Promise((r) => setImmediate(r));
        expect(kvSets()).toBe(2);
    });

    test('the CBOE VIX CSV is downloaded once per request (tag + level share it)', async () => {
        await get('?_fail=sheets_main,sheets_alt');
        expect(seen.filter((u) => u.includes('/VIX_History.csv')).length).toBe(1);
    });
});

describe('/api/sheets fault switches', () => {
    test('?_fail=sheets_main → alt URL', async () => {
        const b = await get('?_fail=sheets_main');
        expect(b.NotSoBoring).toBe('ON');
        expect(b._meta.source).toBe('Google Sheets (Alt URL)');
        expect(b._meta.hasErrors).toBe(true);
        expect(seen.some((u) => u.includes('output=csv'))).toBe(true);
    });

    test('?_fail=sheets_main,sheets_alt → VIX levels from CBOE; NotSoBoring/FrontRunner N/A with no copy', async () => {
        const b = await get('?_fail=sheets_main,sheets_alt');
        expect(b.VIX.current).toBe('15.20');
        expect(b.VIX.threeMonth).toBe('17.77');
        expect(b._meta.fields.vixCurrent.source).toBe('CBOE VIX_History.csv (close 2026-10-09)');
        expect(b._meta.fields.vixThreeMonth.source).toBe('CBOE VIX3M_History.csv (close 2026-10-09)');
        expect(b.NotSoBoring).toBe('N/A');
        expect(b.FrontRunner).toBe('N/A');
        expect(b._meta.source).toBe('CBOE (some N/A)');
    });

    test('?_fail=sheets_main,sheets_alt,sheets_cboe → FRED VIXCLS/VXVCLS, flagged stale (lags a day)', async () => {
        const b = await get('?_fail=sheets_main,sheets_alt,sheets_cboe');
        expect(b.VIX.current).toBe('15.41');
        expect(b.VIX.threeMonth).toBe('18.08');
        expect(b._meta.staleFields).toEqual(expect.arrayContaining(['vixCurrent', 'vixThreeMonth']));
        expect(b._meta.stale).toBe(true);
        expect(b._meta.source).toMatch(/^Stale: FRED Proxy \(VIX\)/);
    });

    test('every VIX tier off → VIX N/A', async () => {
        const b = await get('?_fail=sheets_main,sheets_alt,sheets_cboe,sheets_fred,vix_cboe,vix_fred');
        expect(b.VIX).toEqual({ current: 'N/A', threeMonth: 'N/A', fearGreed: 'N/A' });
        expect(b._meta.source).toBe('Static Defaults');
    });

    test('?_fail=vix_cboe,vix_fred → fear/greed N/A, never the frozen sheet GREED13', async () => {
        const b = await get('?_fail=vix_cboe,vix_fred');
        expect(b.VIX.fearGreed).toBe('N/A');
        expect(b.VIX.current).toBe('14.84');
        expect(b._meta.messages.join(' ')).toMatch(/never used/);
    });

    test('fault test mode never writes /tmp or KV', async () => {
        await get('?_fail=sheets_alt');
        expect(store.__m.size).toBe(0);
        expect(kv.defaultKv.set).not.toHaveBeenCalled();
    });
});

describe('/api/sheets per-field last-good', () => {
    test('sheet down → yesterday\'s NotSoBoring/FrontRunner from /tmp, relabelled + stale', async () => {
        await get(); // healthy load saves the copies
        jest.setSystemTime(new Date(NOW.getTime() + 864e5));
        mode.cboeEnd = '2026-10-10';
        const b = await get('?_fail=sheets_main,sheets_alt');
        expect(b.NotSoBoring).toBe('ON');
        expect(b._meta.fields.NotSoBoring.source).toBe(`/tmp last-good (${NOW.toISOString()}) ← Google Sheets (Live)`);
        expect(b._meta.fields.NotSoBoring.savedAt).toBe(NOW.toISOString());
        expect(b._meta.staleFields).toEqual(expect.arrayContaining(['NotSoBoring', 'FrontRunner']));
        expect(b._meta.staleFields).not.toContain('vixCurrent'); // CBOE is live
        expect(b._meta.source).toMatch(/^Stale: /);
    });

    test('cold instance (/tmp gone) → KV copy, labelled "KV last-good (<savedAt>) ← <orig>"', async () => {
        await get();
        store.__m.clear();
        const b = await get('?_fail=sheets_main,sheets_alt');
        expect(b.FrontRunner).toBe('BIL (T-Bill ETF)');
        expect(b._meta.fields.FrontRunner.source).toBe(`KV last-good (${NOW.toISOString()}) ← Google Sheets (Live)`);
        expect(b._meta.fields.FrontRunner.stale).toBe(true);
    });

    test('tag outage → yesterday\'s CBOE tag from the copy, flagged stale (never the sheet)', async () => {
        await get();
        const b = await get('?_fail=vix_cboe,vix_fred');
        expect(b.VIX.fearGreed).toBe('GREED05');
        expect(b._meta.staleFields).toContain('vixFearGreed');
    });

    test('?_fail=sheets_cache skips /tmp; ?_fail=kvlg / sheets_kvlg skip KV; lastgood skips both', async () => {
        await get();
        expect((await get('?_fail=sheets_main,sheets_alt,sheets_cache'))._meta.fields.NotSoBoring.source).toMatch(/^KV last-good/);
        expect((await get('?_fail=sheets_main,sheets_alt,kvlg'))._meta.fields.NotSoBoring.source).toMatch(/^\/tmp last-good/);
        expect((await get('?_fail=sheets_main,sheets_alt,sheets_cache,sheets_kvlg')).NotSoBoring).toBe('N/A');
        expect((await get('?_fail=sheets_main,sheets_alt,lastgood')).NotSoBoring).toBe('N/A');
    });

    test('copies past their max age are ignored (VIX: 4 days, NotSoBoring: 7 days)', async () => {
        await get();
        jest.setSystemTime(new Date(NOW.getTime() + 5 * 864e5));
        mode.cboe = 'down'; mode.fred = 'down';
        const b = await get('?_fail=sheets_main,sheets_alt');
        expect(b.VIX.current).toBe('N/A');
        expect(b.NotSoBoring).toBe('ON');
        jest.setSystemTime(new Date(NOW.getTime() + 8 * 864e5));
        expect((await get('?_fail=sheets_main,sheets_alt')).NotSoBoring).toBe('N/A');
    });
});

describe('/api/sheets value guards', () => {
    test('a sheet error cell (#N/A) in VIX is not served as live — CBOE fills it', async () => {
        mode.vixCell = '#N/A,#N/A';
        const b = await get();
        expect(b.VIX.current).toBe('15.20');
        expect(b._meta.fields.vixCurrent.source).toMatch(/^CBOE/);
        expect(b._meta.hasErrors).toBe(true);
    });

    test('a frozen CBOE CSV (newest print weeks old) is rejected → FRED', async () => {
        mode.cboeEnd = '2026-09-01';
        const b = await get('?_fail=sheets_main,sheets_alt');
        expect(b.VIX.current).toBe('15.41');
        expect(b._meta.messages.join(' ')).toMatch(/frozen \(newest 2026-09-01\)/);
    });

    test('everything down + no copies → 200 with N/A, never throws', async () => {
        mode = { sheets: 'down', cboe: 'down', fred: 'down' };
        const res = await GET(new Request('https://x.test/api/sheets', { headers: { 'user-agent': 'jest' } }));
        expect(res.status).toBe(200);
        const b = await res.json();
        expect(b.NotSoBoring).toBe('N/A');
        expect(b.VIX.fearGreed).toBe('N/A');
        expect(b._meta.hasErrors).toBe(true);
        expect(res.headers.get('vercel-cdn-cache-control')).toBeNull();
    });
});

describe('/api/sheets CBOE VIX tier is live only when its close is current', () => {
    // Fri 9 Oct 2026. 13:00 ET = in session; 08:00 ET = pre-open; 17:00 ET = after the close.
    const at = (iso) => jest.setSystemTime(new Date(iso));

    test('intraday: the CSV\'s newest row is yesterday\'s close → stale, in staleFields, never persisted', async () => {
        at('2026-10-09T17:00:00Z');
        mode = { sheets: 'down', cboeEnd: '2026-10-08' };
        const b = await get();
        expect(b.VIX.current).toBe('15.20');
        expect(b._meta.fields.vixCurrent).toMatchObject({ stale: true });
        expect(b._meta.fields.vixCurrent.source).toMatch(/close 2026-10-08; market open/);
        expect(b._meta.staleFields).toEqual(expect.arrayContaining(['vixCurrent', 'vixThreeMonth']));
        expect(b._meta.source).toMatch(/^Stale: /);
        const tmp = store.__m.get(TMP_KEY)?.data || {};
        expect(tmp.vixCurrent).toBeUndefined();
        expect(tmp.vixThreeMonth).toBeUndefined();
    });

    test('pre-open: yesterday\'s close IS the current level → live, saved with savedAt = that close', async () => {
        at('2026-10-09T12:00:00Z');
        mode = { sheets: 'down', cboeEnd: '2026-10-08' };
        const b = await get();
        expect(b._meta.fields.vixCurrent.stale).toBe(false);
        expect(b._meta.staleFields).not.toContain('vixCurrent');
        const tmp = store.__m.get(TMP_KEY).data;
        expect(tmp.vixCurrent.savedAt).toBe('2026-10-08T20:00:00.000Z'); // 16:00 ET close, not "now"
        expect(kv.__m.get(KV_KEY).data.vixThreeMonth.savedAt).toBe('2026-10-08T20:00:00.000Z');
    });

    test('after the close but the CSV not yet updated → stale (latest completed session is today)', async () => {
        at('2026-10-09T21:00:00Z');
        mode = { sheets: 'down', cboeEnd: '2026-10-08' };
        const b = await get();
        expect(b._meta.fields.vixCurrent.stale).toBe(true);
        expect(b._meta.fields.vixCurrent.source).toMatch(/latest completed session is 2026-10-09/);
    });

    test('mixing days is visible: intraday sheet VIX live + CBOE 3M from yesterday flagged stale', async () => {
        at('2026-10-09T17:00:00Z');
        mode = { vixCell: '14.84,#N/A', cboeEnd: '2026-10-08' };
        const b = await get();
        expect(b._meta.fields.vixCurrent.stale).toBe(false);
        expect(b._meta.staleFields).toContain('vixThreeMonth');
        expect(b._meta.staleFields).not.toContain('vixCurrent');
    });
});

describe('marketClock: completed sessions', () => {
    const { latestCompletedSessionDate, dailyCloseStatus } = require('../marketClock');
    test('latest completed session flips at the close (16:00 ET; 13:00 on an early close)', () => {
        expect(latestCompletedSessionDate(Date.parse('2026-10-09T19:59:00Z'))).toBe('2026-10-08');
        expect(latestCompletedSessionDate(Date.parse('2026-10-09T20:00:00Z'))).toBe('2026-10-09');
        expect(latestCompletedSessionDate(Date.parse('2026-10-12T14:00:00Z'))).toBe('2026-10-09'); // Monday 10:00 ET → Friday
        expect(latestCompletedSessionDate(Date.parse('2026-11-27T18:30:00Z'))).toBe('2026-11-27'); // early close 13:00 ET
    });
    test('dailyCloseStatus: never current during regular hours', () => {
        expect(dailyCloseStatus('2026-10-09', Date.parse('2026-10-09T17:00:00Z')).current).toBe(false);
        expect(dailyCloseStatus('2026-10-08', Date.parse('2026-10-09T17:00:00Z'))).toMatchObject({ current: false, inSession: true });
        expect(dailyCloseStatus('2026-10-09', Date.parse('2026-10-10T15:00:00Z'))).toMatchObject({ current: true, closeMs: Date.parse('2026-10-09T20:00:00Z') });
        expect(dailyCloseStatus(null, Date.now()).current).toBe(false);
    });
});

describe('/api/sheets time budget', () => {
    test('declares maxDuration and a budget that leaves room for the last-good fill', () => {
        const src = require('fs').readFileSync(require('path').join(__dirname, '../../app/api/sheets/route.js'), 'utf8');
        const md = Number(src.match(/export const maxDuration = (\d+);/)[1]);
        const budget = Number(src.match(/const SHEETS_BUDGET_MS = (\d+);/)[1]);
        expect(budget + 3000 + 5000).toBeLessThanOrEqual(md * 1000);
    });

    test('makeBudget cuts a hung tier at the budget; a rejection degrades the same way', async () => {
        jest.useRealTimers();
        const { makeBudget } = require('../budget');
        const b = makeBudget(30, { minMs: 0 });
        const t0 = Date.now();
        expect(await b.race(new Promise(() => {}), 'late')).toBe('late');
        expect(Date.now() - t0).toBeLessThan(1000);
        expect(await b.race(Promise.reject(new Error('x')), 'late')).toBe('late');
        expect(await makeBudget(5000).race(Promise.resolve(7))).toBe(7);
        jest.useFakeTimers({ now: NOW, doNotFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'setImmediate', 'nextTick', 'queueMicrotask'] });
    });
});

