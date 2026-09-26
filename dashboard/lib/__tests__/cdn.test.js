/**
 * @jest-environment node
 *
 * Edge cache policy (lib/cdn.js) and how serve() applies it (lib/store.js).
 */
import fs from 'fs';
import { cacheHeaders, isDegraded, CDN_POLICY } from '../cdn';
import { serve } from '../store';

// serve('spy', …) writes a real /tmp last-good copy on this machine — remove it.
afterAll(() => { try { fs.unlinkSync('/tmp/lg-spy.json'); } catch { /* not written */ } });

describe('cacheHeaders', () => {
    it('a healthy answer gets the route policy at the edge; the browser never caches', () => {
        expect(cacheHeaders('market-extra', { payload: { _meta: { hasErrors: false } } })).toEqual({
            'cache-control': 'no-store',
            'vercel-cdn-cache-control': 'max-age=120, stale-while-revalidate=480',
        });
    });
    it.each([
        ['stale', { _meta: { stale: true } }],
        ['hasErrors', { _meta: { hasErrors: true } }],
        ['fallback tier', { _meta: { fallback: true } }],
    ])('a degraded (%s) answer is never cached', (_, payload) => {
        expect(cacheHeaders('spy', { payload })).toEqual({ 'cache-control': 'no-store' });
    });
    it.each([
        ['_meta.source', { _meta: { source: 'Yahoo Finance (fallback)', hasErrors: false } }],
        ['top-level source', { value: '+0.2%', source: 'Finnhub (fallback)' }],
    ])('a Lambda route\'s direct "(fallback)" answer (%s) is never cached', (_, payload) => {
        expect(cacheHeaders('spy', { payload })).toEqual({ 'cache-control': 'no-store' });
    });
    it('a route can veto caching for a reason its payload does not flag', () => {
        expect(cacheHeaders('sheets', { payload: {}, degraded: true })).toEqual({ 'cache-control': 'no-store' });
    });
    it('a fault-injection request is never cached', () => {
        expect(cacheHeaders('spy', { payload: {}, testMode: true })).toEqual({ 'cache-control': 'no-store' });
    });
    it('a route without a policy is never cached', () => {
        expect(cacheHeaders('last-run', { payload: {} })).toEqual({ 'cache-control': 'no-store' });
    });
    it('every policy keeps the worst-case age at an hour or less, prices at 10 minutes or less', () => {
        for (const [k, [maxAge, swr]] of Object.entries(CDN_POLICY)) {
            expect(maxAge + swr).toBeLessThanOrEqual(3600);
            if (['spy', 'spy-daily-move', 'market-extra', 'fear-greed', 'sheets', 'vol'].includes(k)) {
                expect(maxAge + swr).toBeLessThanOrEqual(600);
            }
        }
    });
    it('isDegraded tolerates anything', () => {
        expect(isDegraded(null)).toBe(false);
        expect(isDegraded([1, 2])).toBe(false);
        expect(isDegraded({ _meta: null })).toBe(false);
    });
});

describe('serve() + the edge policy', () => {
    const header = (res) => res.headers.get('vercel-cdn-cache-control');

    it('a healthy live answer carries the edge policy', async () => {
        const res = await serve('spy', async () => ({ price: 1, _meta: { hasErrors: false } }));
        expect(header(res)).toBe('max-age=60, stale-while-revalidate=240');
        expect(res.headers.get('cache-control')).toBe('no-store');
    });
    it('a degraded live answer does not', async () => {
        const res = await serve('spy', async () => ({ price: 1, _meta: { stale: true } }));
        expect(header(res)).toBeNull();
    });
    it('fault injection does not', async () => {
        const res = await serve('spy', async () => ({ price: 1 }), { faults: new Set(['lambda']) });
        expect(header(res)).toBeNull();
    });
    it('the fallback path (live threw, nothing cached) does not', async () => {
        const res = await serve('cdn-test-none', async () => { throw new Error('down'); }, { fallback: { price: null } });
        expect(header(res)).toBeNull();
        expect((await res.json())._meta.source).toBe('Unavailable');
    });
});
