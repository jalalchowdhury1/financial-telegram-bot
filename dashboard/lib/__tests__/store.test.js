import { saveLastGood, loadLastGood } from '../store';

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
