/**
 * The page's route reader (lib/loadJson.js): never throws, times out, retries once.
 */
import { getJson, withBust } from '../loadJson';

const ok = (body, status = 200) => ({ status, json: async () => body });
const noSleep = () => Promise.resolve();

describe('withBust', () => {
    it('adds _t only when busting', () => {
        expect(withBust('/api/spy', false, 5)).toBe('/api/spy');
        expect(withBust('/api/spy', true, 5)).toBe('/api/spy?_t=5');
        expect(withBust('/api/x?a=1', true, 5)).toBe('/api/x?a=1&_t=5');
    });
});

describe('getJson', () => {
    it('returns the parsed body; automatic loads carry no cache-buster', async () => {
        const fetchImpl = jest.fn().mockResolvedValue(ok({ a: 1 }));
        await expect(getJson('/api/spy', { fetchImpl })).resolves.toEqual({ a: 1 });
        expect(fetchImpl.mock.calls[0][0]).toBe('/api/spy');
    });
    it('a manual refresh busts the edge cache', async () => {
        const fetchImpl = jest.fn().mockResolvedValue(ok({}));
        await getJson('/api/spy', { fetchImpl, bust: true });
        expect(fetchImpl.mock.calls[0][0]).toMatch(/^\/api\/spy\?_t=\d+$/);
    });
    it('retries once after a network error', async () => {
        const fetchImpl = jest.fn().mockRejectedValueOnce(new TypeError('Failed to fetch')).mockResolvedValue(ok({ a: 2 }));
        await expect(getJson('/api/x', { fetchImpl, sleep: noSleep })).resolves.toEqual({ a: 2 });
        expect(fetchImpl).toHaveBeenCalledTimes(2);
    });
    it('retries once after a 5xx, and parses a final 5xx body rather than dropping it', async () => {
        const fetchImpl = jest.fn().mockResolvedValue(ok({ _meta: { source: 'Failed' } }, 500));
        await expect(getJson('/api/fear-greed', { fetchImpl, sleep: noSleep })).resolves.toEqual({ _meta: { source: 'Failed' } });
        expect(fetchImpl).toHaveBeenCalledTimes(2);
    });
    it('null after two failures — never throws', async () => {
        const fetchImpl = jest.fn().mockRejectedValue(new Error('offline'));
        await expect(getJson('/api/x', { fetchImpl, sleep: noSleep })).resolves.toBeNull();
        expect(fetchImpl).toHaveBeenCalledTimes(2);
    });
    it('an unreadable body counts as a failure', async () => {
        const fetchImpl = jest.fn().mockResolvedValue({ status: 200, json: async () => { throw new SyntaxError('bad json'); } });
        await expect(getJson('/api/x', { fetchImpl, sleep: noSleep })).resolves.toBeNull();
        expect(fetchImpl).toHaveBeenCalledTimes(2);
    });
    it('gives up at the timeout and does NOT retry a timeout', async () => {
        jest.useFakeTimers();
        const fetchImpl = jest.fn((url, { signal }) => new Promise((_, reject) => {
            signal.addEventListener('abort', () => reject(new Error('aborted')));
        }));
        const p = getJson('/api/slow', { fetchImpl, timeoutMs: 1000, sleep: noSleep });
        jest.advanceTimersByTime(1000);
        await expect(p).resolves.toBeNull();
        expect(fetchImpl).toHaveBeenCalledTimes(1);
        jest.useRealTimers();
    });
});
