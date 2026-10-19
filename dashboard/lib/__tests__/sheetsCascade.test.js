
describe('FrontRunner artifact digit', () => {
    const { frontRunnerText } = require('../sheetsCascade');
    test('strips the n8n digit after ")" but keeps real text', () => {
        expect(frontRunnerText('BIL (T-Bill ETF)1\n\n\nThis message was sent automatically with n8n')).toBe('BIL (T-Bill ETF)');
        expect(frontRunnerText('TQQQ (3x Nasdaq)')).toBe('TQQQ (3x Nasdaq)');
        expect(frontRunnerText('UVXY 2')).toBe('UVXY 2');
    });
});

describe('KV outage never wipes last-good (audit 19 Oct 2026)', () => {
    const { persistLiveFields, fillFromLastGood, readKvRecord, KV_KEY } = require('../sheetsCascade');
    const NOW = Date.parse('2026-10-19T15:00:00Z');
    const OLD = { NotSoBoring: { value: 'ON', savedAt: '2026-10-18T20:00:00.000Z', source: 'Computed' },
        FrontRunner: { value: 'TQQQ', savedAt: '2026-10-18T20:00:00.000Z', source: 'Computed' } };
    const memStore = () => { const m = new Map(); return { load: (k) => m.get(k) || null, save: (k, d) => m.set(k, { data: d }) }; };
    const liveVix = () => ({ vixCurrent: { value: '18.20', live: true, source: 'Google Sheets' } });
    // A client like defaultKv: read() reports outages; set() records what it would write.
    const kvClient = ({ readOk = true, value = JSON.stringify({ data: OLD, savedAt: 'x' }) } = {}) => ({
        read: jest.fn(async () => (readOk ? { ok: true, value } : { ok: false, value: null })),
        get: jest.fn(async () => (readOk ? value : null)),
        set: jest.fn(async () => true),
    });

    test('readKvRecord: outage = ok:false, empty key = ok:true + null, throw = ok:false', async () => {
        expect(await readKvRecord(kvClient({ readOk: false }))).toEqual({ ok: false, data: null });
        expect(await readKvRecord(kvClient({ value: null }))).toEqual({ ok: true, data: null });
        expect(await readKvRecord({ read: async () => { throw new Error('boom'); } })).toEqual({ ok: false, data: null });
        expect((await readKvRecord(kvClient())).data.FrontRunner.value).toBe('TQQQ');
    });

    test('persist: KV read fails -> no SET at all (old fields survive)', async () => {
        const kv = kvClient({ readOk: false });
        expect(await persistLiveFields(liveVix(), { store: memStore(), kv, now: NOW })).toBe(false);
        expect(kv.set).not.toHaveBeenCalled();
    });

    test('persist: KV read ok -> merge keeps the other fields', async () => {
        const kv = kvClient();
        expect(await persistLiveFields(liveVix(), { store: memStore(), kv, now: NOW })).toBe(true);
        const [key, env] = kv.set.mock.calls[0];
        expect(key).toBe(KV_KEY);
        expect(env.data.NotSoBoring.value).toBe('ON');
        expect(env.data.FrontRunner.value).toBe('TQQQ');
        expect(env.data.vixCurrent.value).toBe('18.20');
    });

    test('persist: empty KV (first write ever) still writes', async () => {
        const kv = kvClient({ value: null });
        expect(await persistLiveFields(liveVix(), { store: memStore(), kv, now: NOW })).toBe(true);
        expect(Object.keys(kv.set.mock.calls[0][1].data)).toEqual(['vixCurrent']);
    });

    test('fill -> persist chain: a failed read in fill hands persist "unknown", and persist skips', async () => {
        const kv = kvClient({ readOk: false });
        const store = memStore();
        const fields = liveVix();
        const filled = await fillFromLastGood(fields, { store, kv, now: NOW });
        expect(filled.kvRecord).toBeUndefined();
        expect(await persistLiveFields(fields, { store, kv, now: NOW, kvRecord: filled.kvRecord })).toBe(false);
        expect(kv.set).not.toHaveBeenCalled();
    });
});
