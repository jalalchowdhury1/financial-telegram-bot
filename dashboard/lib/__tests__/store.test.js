/**
 * @jest-environment node
 */
import { saveLastGood, loadLastGood, serve } from '../store';

describe('last-known-good store (/tmp)', () => {
    const key = `test-${Math.floor(Date.now() / 1000)}`;

    test('round-trips data with a savedAt timestamp', () => {
        saveLastGood(key, { current: 123, _meta: { source: 'x' } });
        const lg = loadLastGood(key);
        expect(lg).not.toBeNull();
        expect(lg.data.current).toBe(123);
        expect(typeof lg.savedAt).toBe('string');
    });

    test('respects maxAgeMs', () => {
        saveLastGood(key, { v: 1 });
        // generous tolerance -> returned
        expect(loadLastGood(key, 60_000)).not.toBeNull();
        // negative tolerance -> always rejected (deterministic)
        expect(loadLastGood(key, -1)).toBeNull();
    });

    test('missing key returns null, never throws', () => {
        expect(loadLastGood('definitely-missing-key-xyz')).toBeNull();
    });

    test('saveLastGood never throws on bad input', () => {
        const circular = {}; circular.self = circular;
        expect(() => saveLastGood(key, circular)).not.toThrow();
    });
});

describe('serve() lastResort (Google-Sheet fallback)', () => {
    const isGood = (p) => p && p._meta && p._meta.loadedCount > 0;
    const fallback = { error: 'FRED temporarily unavailable' };

    test('live throws + cold cache + lastResort returns payload -> serves it (stale, not error)', async () => {
        const coldKey = `cold-${Math.floor(Date.now() / 1000)}-a`;
        const lr = jest.fn(async () => ({
            indicators: { m2: { value: 4.72 } },
            _meta: { source: 'Google Sheet', loadedCount: 0 },
        }));
        const res = await serve(coldKey, async () => { throw new Error('total FRED outage'); },
            { isGood, fallback, lastResort: lr });
        const body = await res.json();
        expect(res.status).toBe(200);
        expect(body.indicators.m2.value).toBe(4.72); // served sheet data, not the error
        expect(body.error).toBeUndefined();
        expect(body._meta.stale).toBe(true);
        expect(lr).toHaveBeenCalledTimes(1);
    });

    test('live throws + cold cache + lastResort returns null -> fallback error', async () => {
        const coldKey = `cold-${Math.floor(Date.now() / 1000)}-b`;
        const res = await serve(coldKey, async () => { throw new Error('outage'); },
            { isGood, fallback, lastResort: async () => null });
        const body = await res.json();
        expect(body.error).toBe('FRED temporarily unavailable');
    });

    test('warm /tmp last-good takes precedence: lastResort is NOT called', async () => {
        const warmKey = `warm-${Math.floor(Date.now() / 1000)}-c`;
        saveLastGood(warmKey, { indicators: { m2: { value: 9.9 } }, _meta: { source: 'St. Louis Fed', loadedCount: 17 } });
        const lr = jest.fn(async () => ({ indicators: {}, _meta: { loadedCount: 0 } }));
        const res = await serve(warmKey, async () => { throw new Error('outage'); },
            { isGood, fallback, lastResort: lr });
        const body = await res.json();
        expect(body.indicators.m2.value).toBe(9.9); // from /tmp, not the sheet
        expect(lr).not.toHaveBeenCalled();
    });

    test('a throwing lastResort never breaks serve() -> falls back to error', async () => {
        const coldKey = `cold-${Math.floor(Date.now() / 1000)}-d`;
        const res = await serve(coldKey, async () => { throw new Error('outage'); },
            { isGood, fallback, lastResort: async () => { throw new Error('sheet down too'); } });
        const body = await res.json();
        expect(body.error).toBe('FRED temporarily unavailable');
    });
});

// A payload can be servable without being worth STORING — /api/fred builds one
// when live FRED is dead but the Four Horsemen resolved from Treasury/BLS. It is
// mostly cached content, so storing it would refresh `savedAt` on stale data and
// let it live past the 7-day window indefinitely.
describe('serve — shouldStore separate from isGood', () => {
    test('a servable-but-not-storable payload is returned and NOT cached', async () => {
        const key = `test-nostore-${Date.now()}`;
        const payload = { v: 1, _meta: { loadedCount: 0, horsemenLive: 3 } };
        const res = await serve(key, async () => payload, {
            isGood: (p) => p._meta.loadedCount > 0 || p._meta.horsemenLive > 0,
            shouldStore: (p) => p._meta.loadedCount > 0,
        });
        expect((await res.json()).v).toBe(1);
        expect(loadLastGood(key)).toBeNull();   // never written
    });

    test('a fully live payload IS cached', async () => {
        const key = `test-store-${Date.now()}`;
        const payload = { v: 2, _meta: { loadedCount: 17, horsemenLive: 0 } };
        await serve(key, async () => payload, {
            isGood: (p) => p._meta.loadedCount > 0 || p._meta.horsemenLive > 0,
            shouldStore: (p) => p._meta.loadedCount > 0,
        });
        expect(loadLastGood(key).data.v).toBe(2);
    });

    test('shouldStore defaults to isGood when not supplied (back-compat)', async () => {
        const key = `test-default-${Date.now()}`;
        await serve(key, async () => ({ v: 3 }), { isGood: () => true });
        expect(loadLastGood(key).data.v).toBe(3);
    });
});
