/**
 * proxyFetch error messages reach public `_meta.messages`, so a key in the URL
 * (e.g. FRED api_key in /api/fear-greed) must never appear in them.
 */
import { maskUrl, proxyFetch } from '../fetcher';

describe('maskUrl', () => {
    test('masks api_key / apikey / token / key params, keeps the rest', () => {
        expect(maskUrl('https://x/obs?series_id=VIXCLS&api_key=SECRET1&file_type=json'))
            .toBe('https://x/obs?series_id=VIXCLS&api_key=***&file_type=json');
        expect(maskUrl('https://x/q?apiKey=SECRET2')).toBe('https://x/q?apiKey=***');
        expect(maskUrl('https://x/q?symbol=SPY&token=SECRET3')).toBe('https://x/q?symbol=SPY&token=***');
        expect(maskUrl('https://x/q?key=SECRET4&a=1')).toBe('https://x/q?key=***&a=1');
        expect(maskUrl('https://x/chart/SPY?range=1mo')).toBe('https://x/chart/SPY?range=1mo');
    });
});

describe('proxyFetch error messages', () => {
    const realFetch = global.fetch;
    afterEach(() => { global.fetch = realFetch; });

    test('HTTP error message never contains the key', async () => {
        global.fetch = jest.fn().mockResolvedValue({ ok: false, status: 429, statusText: 'Too Many Requests' });
        const err = await proxyFetch('https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key=SECRETKEY')
            .catch((e) => e);
        expect(err.message).toContain('429');
        expect(err.message).toContain('api_key=***');
        expect(err.message).not.toContain('SECRETKEY');
    });

    test('timeout message never contains the key', async () => {
        global.fetch = jest.fn().mockRejectedValue(Object.assign(new Error('aborted'), { name: 'AbortError' }));
        const err = await proxyFetch('https://x/q?api_key=SECRETKEY', { timeout: 5 }).catch((e) => e);
        expect(err.message).toContain('timed out');
        expect(err.message).not.toContain('SECRETKEY');
    });
});
