/**
 * @jest-environment node
 *
 * Edge cache policy (lib/cdn.js) and how serve() applies it (lib/store.js).
 */
import fs from 'fs';
import { cacheHeaders, isDegraded, CDN_POLICY, DEGRADED_MAX_AGE, OFF_HOURS_MAX_AGE } from '../cdn';
import { serve } from '../store';

// serve('spy', …) writes a real /tmp last-good copy on this machine — remove it.
afterAll(() => { try { fs.unlinkSync('/tmp/lg-spy.json'); } catch { /* not written */ } });

// The policy is market-aware (off-hours stretch), so pin "now" to a session: Wed 11:00 ET.
const OPEN = Date.parse('2026-10-21T15:00:00Z');
beforeEach(() => { jest.spyOn(Date, 'now').mockReturnValue(OPEN); });
afterEach(() => { jest.restoreAllMocks(); });

describe('cacheHeaders', () => {
    it('a healthy answer gets the route policy at the edge; the browser never caches', () => {
        expect(cacheHeaders('market-extra', { payload: { _meta: { hasErrors: false } } })).toEqual({
            'cache-control': 'no-store',
            'vercel-cdn-cache-control': 'max-age=330, stale-while-revalidate=270',
        });
    });
    it.each([
        ['stale', { _meta: { stale: true } }],
        ['hasErrors', { _meta: { hasErrors: true } }],
        ['fallback tier', { _meta: { fallback: true } }],
    ])('a degraded (%s) answer gets only a 2-min edge copy, no swr', (_, payload) => {
        expect(cacheHeaders('fred', { payload })).toEqual({ 'cache-control': 'no-store', 'vercel-cdn-cache-control': `max-age=${DEGRADED_MAX_AGE}` });
    });
    it.each([
        ['_meta.source', { _meta: { source: 'Yahoo Finance (fallback)', hasErrors: false } }],
        ['top-level source', { value: '+0.2%', source: 'Finnhub (fallback)' }],
    ])('a Lambda route\'s direct "(fallback)" answer (%s) gets only the short copy', (_, payload) => {
        expect(cacheHeaders('spy', { payload })['vercel-cdn-cache-control']).toBe(`max-age=${DEGRADED_MAX_AGE}`);
    });
    it('a route can mark degraded for a reason its payload does not flag', () => {
        expect(cacheHeaders('sheets', { payload: {}, degraded: true })['vercel-cdn-cache-control']).toBe(`max-age=${DEGRADED_MAX_AGE}`);
    });
    it('a fault-injection request is never cached', () => {
        expect(cacheHeaders('spy', { payload: {}, testMode: true })).toEqual({ 'cache-control': 'no-store' });
    });
    it('a route without a policy is never cached', () => {
        expect(cacheHeaders('last-run', { payload: {} })).toEqual({ 'cache-control': 'no-store' });
    });
    it('every policy keeps the worst-case age at 3 hours or less, prices at 10 minutes or less', () => {
        for (const [k, [maxAge, swr]] of Object.entries(CDN_POLICY)) {
            expect(maxAge + swr).toBeLessThanOrEqual(3 * 3600);
            if (['spy', 'spy-daily-move', 'market-extra', 'fear-greed', 'sheets', 'vol', 'jev-pills'].includes(k)) {
                expect(maxAge + swr).toBeLessThanOrEqual(600);
                expect(maxAge).toBeGreaterThan(300); // the next 5-min refresh is an edge HIT
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
        expect(header(res)).toBe('max-age=330, stale-while-revalidate=270');
        expect(res.headers.get('cache-control')).toBe('no-store');
    });
    it('a degraded live answer gets only the short copy', async () => {
        const res = await serve('spy', async () => ({ price: 1, _meta: { stale: true } }));
        expect(header(res)).toBe(`max-age=${DEGRADED_MAX_AGE}`);
    });
    it('fault injection does not', async () => {
        const res = await serve('spy', async () => ({ price: 1 }), { faults: new Set(['lambda']) });
        expect(header(res)).toBeNull();
    });
    it('the fallback path (live threw, nothing cached) gets no full policy', async () => {
        const res = await serve('cdn-test-none', async () => { throw new Error('down'); }, { fallback: { price: null } });
        expect(header(res)).toBeNull(); // no CDN_POLICY row for this key
        expect((await res.json())._meta.source).toBe('Unavailable');
    });
});
