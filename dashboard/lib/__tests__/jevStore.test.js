/**
 * Tests for jevStore — /tmp + KV last-known-good for the Jev pill inputs.
 */
import { makePillStore, KV_PREFIX, REWRITE_MS } from '../jevStore';

function deps({ local = null, remote = null } = {}) {
    const tmpStore = new Map(local ? [['k', local]] : []);
    const tmp = {
        load: jest.fn((key) => tmpStore.get(key) ?? null),
        save: jest.fn((key, data) => tmpStore.set(key, { data, savedAt: new Date().toISOString() })),
    };
    const kv = {
        get: jest.fn(async () => remote),
        set: jest.fn(async () => true),
    };
    return { tmp, kv };
}

describe('makePillStore.load', () => {
    test('serves the /tmp copy without touching KV', async () => {
        const d = deps({ local: { data: { value: 1, asOf: '2026-01-01' }, savedAt: '2026-01-01T00:00:00Z' } });
        const s = makePillStore(d);
        const got = await s.load('k');
        expect(got.data.value).toBe(1);
        expect(d.kv.get).not.toHaveBeenCalled();
    });

    test('falls back to KV (JSON string) and re-warms /tmp', async () => {
        const savedAt = new Date().toISOString();
        const d = deps({ remote: JSON.stringify({ data: { value: 2, asOf: '2026-01-02' }, savedAt }) });
        const s = makePillStore(d);
        const got = await s.load('k', 7 * 864e5);
        expect(d.kv.get).toHaveBeenCalledWith(`${KV_PREFIX}k`);
        expect(got.data.value).toBe(2);
        expect(d.tmp.save).toHaveBeenCalledWith('k', { value: 2, asOf: '2026-01-02' });
    });

    test('KV copy older than maxAgeMs is ignored', async () => {
        const d = deps({ remote: { data: { value: 2 }, savedAt: '2020-01-01T00:00:00Z' } });
        const s = makePillStore(d);
        expect(await s.load('k', 7 * 864e5)).toBeNull();
    });

    test('never throws when KV explodes', async () => {
        const d = deps();
        d.kv.get.mockRejectedValue(new Error('boom'));
        expect(await makePillStore(d).load('k')).toBeNull();
    });
});

describe('makePillStore.save', () => {
    test('writes /tmp and KV on a new value', async () => {
        const d = deps();
        const s = makePillStore(d);
        const ok = await s.save('k', { value: 3, asOf: '2026-01-03', source: 'fred' });
        expect(ok).toBe(true);
        expect(d.tmp.save).toHaveBeenCalled();
        expect(d.kv.set).toHaveBeenCalledWith(`${KV_PREFIX}k`, expect.objectContaining({ data: { value: 3, asOf: '2026-01-03', source: 'fred' } }));
    });

    test('skips the KV write when value/asOf are unchanged and the local copy is recent', async () => {
        const d = deps({ local: { data: { value: 3, asOf: '2026-01-03' }, savedAt: new Date().toISOString() } });
        const s = makePillStore(d);
        const ok = await s.save('k', { value: 3, asOf: '2026-01-03', source: 'fred' });
        expect(ok).toBe(false);
        expect(d.kv.set).not.toHaveBeenCalled();
    });

    test('rewrites KV when the local copy is older than REWRITE_MS even if unchanged', async () => {
        const old = new Date(Date.now() - REWRITE_MS - 1000).toISOString();
        const d = deps({ local: { data: { value: 3, asOf: '2026-01-03' }, savedAt: old } });
        await makePillStore(d).save('k', { value: 3, asOf: '2026-01-03', source: 'fred' });
        expect(d.kv.set).toHaveBeenCalled();
    });

    test('never throws when KV explodes', async () => {
        const d = deps();
        d.kv.set.mockRejectedValue(new Error('boom'));
        expect(await makePillStore(d).save('k', { value: 1, asOf: 'x' })).toBe(false);
    });
});
