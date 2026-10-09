/**
 * @jest-environment node
 *
 * /api/aaii (the contract other repos read) and /api/sheets (AAIIDiff now from AAII,
 * never from the retired AAII Google Sheet).
 */
import fs from 'fs';
import path from 'path';

const TABLE = fs.readFileSync(path.join(__dirname, 'fixtures', 'aaii-sent-results-2026-09-27.html'), 'utf8');
let mode = 'ok';          // 'ok' | 'down'
const seen = [];

// the baked Mac-job file is real data; these tests pin the live tiers, so it is empty here
jest.mock('../data/aaiiNewest.json', () => ({}));
jest.mock('../fetcher', () => ({
    fetchText: jest.fn(async (url) => {
        seen.push(url);
        if (url.includes('1zQQ2am1yhzTwY7nx8xPak4Q0WoNMwxWj7Ekr-fDEIF4')) throw new Error('the retired AAII sheet must not be read');
        if (url.includes('aaii.com')) {
            if (mode === 'down') throw new Error(`Fetch failed for ${url}: 403 Forbidden`);
            if (url.includes('sent_results')) return TABLE;
            throw new Error('substack not needed');
        }
        if (url.includes('docs.google.com')) return 'h1,h2,h3\nON,x,y\nON,val,z\n';
        if (url.includes('cboe.com')) throw new Error('offline');
        throw new Error(`unexpected ${url}`);
    }),
    fetchJson: jest.fn(async () => { throw new Error('offline'); }),
}));

// Isolate the /tmp cache between tests.
jest.mock('../store', () => {
    const m = new Map();
    return {
        __m: m,
        loadLastGood: (k) => m.get(k) || null,
        saveLastGood: (k, data) => m.set(k, { data, savedAt: new Date().toISOString() }),
    };
});

const store = require('../store');
const req = (q = '') => new Request(`https://x.test/api/aaii${q}`, { headers: { 'user-agent': 'jest' } });

// Pin "today" so the Sep 23 survey stays this year's and fresh (timers stay real).
beforeAll(() => jest.useFakeTimers({ now: new Date('2026-09-27T18:00:00Z'), doNotFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'setImmediate', 'nextTick', 'queueMicrotask'] }));
afterAll(() => jest.useRealTimers());
beforeEach(() => { mode = 'ok'; seen.length = 0; store.__m.clear(); });

describe('/api/aaii', () => {
    const { GET } = require('../../app/api/aaii/route');

    test('200 with the contract fields; Sep 23 survey = 15.40%', async () => {
        const res = await GET(req());
        expect(res.status).toBe(200);
        const b = await res.json();
        expect(b).toMatchObject({ bull: 32.7, neutral: 19.2, bear: 48.1, diff: '15.40%', as_of: '2026-09-23', source: 'aaii.com' });
        expect(b.stale).toBe(false);
    });

    test('503 with an error and NO numbers when every source and the last good are gone', async () => {
        mode = 'down';
        const res = await GET(req());
        expect(res.status).toBe(503);
        const b = await res.json();
        expect(b.error).toMatch(/AAII unavailable/);
        expect(b.bull).toBeUndefined();
        expect(b.diff).toBeUndefined();
    });
});

describe('/api/sheets AAII overlay', () => {
    const { GET } = require('../../app/api/sheets/route');

    test('AAIIDiff comes from AAII (not the retired sheet) with as-of + source', async () => {
        const res = await GET(new Request('https://x.test/api/sheets', { headers: { 'user-agent': 'jest' } }));
        const b = await res.json();
        expect(b.AAIIDiff).toBe('15.40%');
        expect(b.AAII).toMatchObject({ as_of: '2026-09-23', source: 'aaii.com', bull: 32.7, bear: 48.1 });
        expect(seen.some((u) => u.includes('1zQQ2am1yhzTwY7nx8xPak4Q0WoNMwxWj7Ekr-fDEIF4'))).toBe(false);
        expect(b.NotSoBoring).toBe('val');    // other sheets untouched
    });

    test('AAII down → AAIIDiff N/A + hasErrors (never an old sheet value)', async () => {
        mode = 'down';
        const res = await GET(new Request('https://x.test/api/sheets', { headers: { 'user-agent': 'jest' } }));
        const b = await res.json();
        expect(b.AAIIDiff).toBe('N/A');
        expect(b.AAII).toBeNull();
        expect(b._meta.hasErrors).toBe(true);
        expect(b._meta.messages.join(' ')).toMatch(/AAII unavailable/);
    });
});
