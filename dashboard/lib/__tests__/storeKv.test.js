/**
 * @jest-environment node
 *
 * serve()'s durable KV last-known-good tier (lib/store.js + lib/kv.js):
 * live → /tmp → KV `ftb:lg:<key>` → lastResort → fallback.
 */
import fs from 'fs';
import { serve, saveLastGood, saveLastGoodKV, loadLastGoodKV, kvKeyFor, isPartialPayload, KV_REWRITE_MS, KV_MIN_GAP_MS, KV_MAX_BYTES } from '../store';
import { runInBackground } from '../background';
import { kvCall, defaultKv, parseEnvelope } from '../kv';

const fakeKv = (seed = {}) => {
    const m = new Map(Object.entries(seed));
    return {
        m,
        get: jest.fn(async (k) => (m.has(k) ? JSON.stringify(m.get(k)) : null)),
        set: jest.fn(async (k, v) => { m.set(k, v); return true; }),
    };
};
let n = 0;
const uniq = (p) => `${p}-${process.pid}-${Date.now()}-${n++}`;
const rmTmp = (key) => { for (const k of [key, `kvmark-${key}`]) { try { fs.unlinkSync(`/tmp/lg-${k.replace(/[^a-z0-9_-]/gi, '_')}.json`); } catch { /* none */ } } };
const faults = (...xs) => new Set(xs);
const good = { v: 42, _meta: { source: 'Live API', hasErrors: false, messages: ['ok'] } };
const boom = async () => { throw new Error('total outage'); };

describe('serve() KV last-good tier', () => {
    test('healthy answer is written to KV under ftb:lg:<key>', async () => {
        const key = uniq('kvw'); const kv = fakeKv();
        const res = await serve(key, async () => good, { kv });
        expect((await res.json()).v).toBe(42);
        expect(kv.set).toHaveBeenCalledTimes(1);
        expect(kv.set.mock.calls[0][0]).toBe(kvKeyFor(key));
        expect(kv.m.get(`ftb:lg:${key}`).data.v).toBe(42);
        rmTmp(key);
    });

    test('/tmp miss → serves the KV copy, relabelled + stale (never as live)', async () => {
        const key = uniq('kvr');
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: '2026-10-08T20:00:00.000Z' } });
        const res = await serve(key, boom, { kv, fallback: { error: 'x' } });
        const b = await res.json();
        expect(res.status).toBe(200);
        expect(b.v).toBe(42);
        expect(b._meta.source).toBe('KV last-good (2026-10-08T20:00:00.000Z) ← Live API');
        expect(b._meta.stale).toBe(true);
        expect(b._meta.hasErrors).toBe(true);
        expect(b._meta.lastGoodAt).toBe('2026-10-08T20:00:00.000Z');
        expect(b._meta.messages).toContain('cached: ok');
        expect(res.headers.get('cache-control')).toBe('no-store');
    });

    test('empty live payload also falls to KV', async () => {
        const key = uniq('kve');
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: new Date().toISOString() } });
        const b = await (await serve(key, async () => null, { kv })).json();
        expect(b._meta.source).toMatch(/^KV last-good/);
        expect(b._meta.messages.join(' ')).toMatch(/live produced empty/);
    });

    test('/tmp copy wins over KV (order: /tmp → KV)', async () => {
        const key = uniq('kvo');
        saveLastGood(key, { v: 1, _meta: { source: 'tmp' } });
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: new Date().toISOString() } });
        const b = await (await serve(key, boom, { kv })).json();
        expect(b.v).toBe(1);
        expect(kv.get).not.toHaveBeenCalled();
        rmTmp(key);
    });

    test('KV beats lastResort; lastResort only when KV misses', async () => {
        const key = uniq('kvl');
        const lr = jest.fn(async () => ({ v: 7, _meta: { source: 'Sheet' } }));
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: new Date().toISOString() } });
        expect((await (await serve(key, boom, { kv, lastResort: lr })).json()).v).toBe(42);
        expect(lr).not.toHaveBeenCalled();
        const b = await (await serve(uniq('kvl2'), boom, { kv: fakeKv(), lastResort: lr })).json();
        expect(b.v).toBe(7);
    });

    test('KV copy older than maxStaleMs is ignored', async () => {
        const key = uniq('kvold');
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: '2020-01-01T00:00:00.000Z' } });
        const b = await (await serve(key, boom, { kv, fallback: { error: 'none' } })).json();
        expect(b.error).toBe('none');
    });

    test('?_fail=kvlg disables only the KV read (lastResort still reachable)', async () => {
        const key = uniq('kvf');
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: new Date().toISOString() } });
        const lr = async () => ({ v: 7, _meta: { source: 'Sheet' } });
        const b = await (await serve(key, boom, { kv, lastResort: lr, faults: faults('kvlg') })).json();
        expect(b.v).toBe(7);
        expect(kv.get).not.toHaveBeenCalled();
    });

    test('?_fail=tmplg skips a warm /tmp copy so the KV tier answers', async () => {
        const key = uniq('kvtmp');
        saveLastGood(key, { v: 1, _meta: { source: 'tmp' } });
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: new Date().toISOString() } });
        const b = await (await serve(key, boom, { kv, faults: faults('tmplg') })).json();
        expect(b.v).toBe(42);
        expect(b._meta.source).toMatch(/^KV last-good/);
        rmTmp(key);
    });

    test('?_fail=lastgood disables the KV read too', async () => {
        const key = uniq('kvlg');
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: new Date().toISOString() } });
        const b = await (await serve(key, boom, { kv, fallback: { error: 'def' }, faults: faults('lastgood') })).json();
        expect(b.error).toBe('def');
        expect(kv.get).not.toHaveBeenCalled();
    });

    test('fault test mode never WRITES KV, even on a healthy answer', async () => {
        const key = uniq('kvnw'); const kv = fakeKv();
        await serve(key, async () => good, { kv, faults: faults('polygon') });
        expect(kv.set).not.toHaveBeenCalled();
    });

    test('a throwing KV client cannot break the route', async () => {
        const key = uniq('kvt');
        const kv = { get: async () => { throw new Error('kv down'); }, set: async () => { throw new Error('kv down'); } };
        const ok = await serve(key, async () => good, { kv });
        expect((await ok.json()).v).toBe(42);
        rmTmp(key);
        const b = await (await serve(uniq('kvt2'), boom, { kv, fallback: { error: 'def' } })).json();
        expect(b.error).toBe('def');
    });
});

describe('serve(): KV completeness gate + background write', () => {
    test('a full KV copy survives a partial run (served, /tmp kept, KV untouched)', async () => {
        const key = uniq('surv'); const kv = fakeKv();
        await serve(key, async () => good, { kv });
        await new Promise((r) => setImmediate(r));
        expect(kv.set).toHaveBeenCalledTimes(1);
        rmTmp(key); // new instance → no throttle marker either
        const partial = { v: 1, _meta: { source: 'Live API', hasErrors: true, messages: ['unavailable: 3 metrics'] } };
        const b = await (await serve(key, async () => partial, { kv })).json();
        await new Promise((r) => setImmediate(r));
        expect(b.v).toBe(1);
        expect(kv.set).toHaveBeenCalledTimes(1);
        expect(kv.m.get(kvKeyFor(key)).data.v).toBe(42);
        rmTmp(key);
    });

    test('the response does not wait for a slow KV SET', async () => {
        const key = uniq('slow');
        const kv = { get: async () => null, set: jest.fn(() => new Promise(() => {})) }; // never settles
        const res = await serve(key, async () => good, { kv });
        expect((await res.json()).v).toBe(42);
        expect(kv.set).toHaveBeenCalledTimes(1);
        rmTmp(key);
    });

    test('runInBackground hands the task to Vercel\'s waitUntil when the request context exists', async () => {
        const sym = Symbol.for('@vercel/request-context');
        const waitUntil = jest.fn();
        globalThis[sym] = { get: () => ({ waitUntil }) };
        try {
            await runInBackground(async () => { throw new Error('boom'); }); // never rejects
            expect(waitUntil).toHaveBeenCalledTimes(1);
        } finally { delete globalThis[sym]; }
        await expect(runInBackground(() => { throw new Error('sync boom'); })).resolves.toBeUndefined();
    });
});

describe('serve(): preferNewer (a flagged build newer than the cache)', () => {
    const staleBuild = { v: 9, _meta: { source: 'Polygon (fallback, last close 2026-10-08)', stale: true, hasErrors: true, asOf: '2026-10-08', messages: [] } };
    const preferNewer = (p, savedAt) => p._meta.asOf >= savedAt.slice(0, 10);
    const notGood = (x) => x && !x._meta?.stale;

    test('older KV copy loses to the newer flagged build (still flagged stale)', async () => {
        const key = uniq('pn');
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: '2026-10-03T20:00:00.000Z' } });
        const b = await (await serve(key, async () => staleBuild, { kv, isGood: notGood, preferNewer })).json();
        expect(b.v).toBe(9);
        expect(b._meta.stale).toBe(true);
        expect(b._meta.messages.join(' ')).toMatch(/KV last-known-good \(2026-10-03.*older/);
    });

    test('newer /tmp copy still wins over an older flagged build', async () => {
        const key = uniq('pn2');
        saveLastGood(key, good); // savedAt = now (newer than 2026-10-08)
        const b = await (await serve(key, async () => staleBuild, { kv: fakeKv(), isGood: notGood, preferNewer })).json();
        expect(b.v).toBe(42);
        expect(b._meta.stale).toBe(true);
        rmTmp(key);
    });

    test('without preferNewer the copy is served (old behaviour); a throwing hook is ignored', async () => {
        const key = uniq('pn3');
        const kv = fakeKv({ [`ftb:lg:${key}`]: { data: good, savedAt: '2026-10-03T20:00:00.000Z' } });
        expect((await (await serve(key, async () => staleBuild, { kv, isGood: notGood })).json()).v).toBe(42);
        const bad = () => { throw new Error('x'); };
        expect((await (await serve(key, async () => staleBuild, { kv, isGood: notGood, preferNewer: bad })).json()).v).toBe(42);
    });
});

describe('saveLastGoodKV throttle', () => {
    test('unchanged payload is not rewritten within the rewrite window (1 h), is after', async () => {
        const key = uniq('thr'); const kv = fakeKv(); const t0 = Date.now();
        expect(await saveLastGoodKV(key, good, { kv, now: t0 })).toBe(true);
        expect(await saveLastGoodKV(key, good, { kv, now: t0 + 5 * 60e3 })).toBe(false);
        expect(await saveLastGoodKV(key, good, { kv, now: t0 + KV_REWRITE_MS + 1 })).toBe(true);
        expect(kv.set).toHaveBeenCalledTimes(2);
        rmTmp(key);
    });

    test('changed payload is rewritten only after the gap (1 h)', async () => {
        const key = uniq('chg'); const kv = fakeKv(); const t0 = Date.now();
        await saveLastGoodKV(key, good, { kv, now: t0 });
        expect(await saveLastGoodKV(key, { ...good, v: 43 }, { kv, now: t0 + 1000 })).toBe(false);
        expect(await saveLastGoodKV(key, { ...good, v: 43 }, { kv, now: t0 + KV_MIN_GAP_MS + 1 })).toBe(true);
        expect(kv.m.get(`ftb:lg:${key}`).data.v).toBe(43);
        rmTmp(key);
    });

    test('skips payloads over 900 KB and payloads already flagged stale', async () => {
        const kv = fakeKv();
        expect(await saveLastGoodKV(uniq('big'), { blob: 'x'.repeat(KV_MAX_BYTES + 1) }, { kv })).toBe(false);
        expect(await saveLastGoodKV(uniq('stl'), { v: 1, _meta: { stale: true } }, { kv })).toBe(false);
        expect(kv.set).not.toHaveBeenCalled();
    });

    test('a last-good copy needs no 10-min freshness: both windows are >= 60 min', () => {
        expect(KV_MIN_GAP_MS).toBeGreaterThanOrEqual(60 * 60e3);
        expect(KV_REWRITE_MS).toBeGreaterThanOrEqual(60 * 60e3);
    });

    test('partial payloads (hasErrors / staleFields / staleMetrics) are never written', async () => {
        const kv = fakeKv();
        for (const meta of [{ hasErrors: true }, { staleFields: ['vixCurrent'] }, { staleMetrics: ['tnx'] }, { stale: true }]) {
            expect(isPartialPayload({ v: 1, _meta: meta })).toBe(true);
            expect(await saveLastGoodKV(uniq('part'), { v: 1, _meta: meta }, { kv })).toBe(false);
        }
        expect(isPartialPayload(good)).toBe(false);
        expect(isPartialPayload({ v: 1, _meta: { staleFields: [], staleMetrics: [] } })).toBe(false);
        expect(isPartialPayload({ v: 1 })).toBe(false);
        expect(kv.set).not.toHaveBeenCalled();
    });

    test('failed KV SET leaves no marker, so the next call retries', async () => {
        const key = uniq('fail');
        const kv = { get: async () => null, set: jest.fn(async () => false) };
        expect(await saveLastGoodKV(key, good, { kv })).toBe(false);
        expect(await saveLastGoodKV(key, good, { kv })).toBe(false);
        expect(kv.set).toHaveBeenCalledTimes(2);
    });

    test('loadLastGoodKV tolerates garbage', async () => {
        expect(await loadLastGoodKV('x', 0, { kv: { get: async () => '{not json' } })).toBeNull();
        expect(await loadLastGoodKV('x', 0, { kv: { get: async () => JSON.stringify({ data: 1 }) } })).toBeNull(); // no savedAt
    });
});

describe('lib/kv.js', () => {
    const env = { ...process.env };
    afterEach(() => { process.env = { ...env }; delete global.fetch; });

    test('no env → silently skipped (no fetch, null/false)', async () => {
        delete process.env.KV_REST_API_URL; delete process.env.KV_REST_API_TOKEN;
        global.fetch = jest.fn();
        expect(await kvCall('/get/x')).toBeNull();
        expect(await defaultKv.get('x')).toBeNull();
        expect(await defaultKv.set('x', 1)).toBe(false);
        expect(global.fetch).not.toHaveBeenCalled();
    });

    test('calls Upstash REST with no-store + bearer token; never throws on network error', async () => {
        process.env.KV_REST_API_URL = 'https://kv.test'; process.env.KV_REST_API_TOKEN = 'tok';
        global.fetch = jest.fn(async () => ({ ok: true, json: async () => ({ result: '{"a":1}' }) }));
        expect(await defaultKv.get('ftb:lg:spy')).toBe('{"a":1}');
        const [url, init] = global.fetch.mock.calls[0];
        expect(url).toBe('https://kv.test/get/ftb%3Alg%3Aspy');
        expect(init.cache).toBe('no-store');
        expect(init.headers.Authorization).toBe('Bearer tok');
        global.fetch = jest.fn(async () => { throw new Error('ECONNRESET'); });
        expect(await defaultKv.get('x')).toBeNull();
        expect(await defaultKv.set('x', 1)).toBe(false);
    });

    test('parseEnvelope never throws', () => {
        expect(parseEnvelope('{bad')).toBeNull();
        expect(parseEnvelope(null)).toBeNull();
        expect(parseEnvelope('{"data":1,"savedAt":"t"}')).toEqual({ data: 1, savedAt: 't' });
    });
});
