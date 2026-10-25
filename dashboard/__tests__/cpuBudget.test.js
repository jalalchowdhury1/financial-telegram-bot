/**
 * @jest-environment node
 *
 * Tripwire: Vercel Active CPU budget (free team, 4 h / rolling 30 days, shared by every
 * site; over it, ALL his sites pause). 24 Oct 2026: the 24/7 30-min /api/freshness
 * probe re-ran every market route each time (10-min edge copies), and degraded answers
 * were never cached, so one page load ran a failing route twice. If a test here fails,
 * the CPU leak is back — read AGENTS.md §3 "CPU budget" before changing the numbers.
 */
import fs from 'fs';
import path from 'path';
import { cacheHeaders, CDN_POLICY, DEGRADED_MAX_AGE } from '../lib/cdn';
import { refreshMs } from '../lib/marketClock';
import { freshnessHeaders } from '../app/api/freshness/route';

const API = path.join(__dirname, '..', 'app', 'api');
const NO_EDGE = ['assessment', 'last-run']; // POST-only / nothing polls it
const maxAge = (h) => Number(/max-age=(\d+)/.exec(h['vercel-cdn-cache-control'] || '')?.[1] || 0);
const at = (iso) => Date.parse(iso);

test('every GET data route sends an edge policy (a route with none runs on every request)', () => {
    for (const r of fs.readdirSync(API)) {
        if (NO_EDGE.includes(r)) continue;
        const src = fs.readFileSync(path.join(API, r, 'route.js'), 'utf8');
        expect([r, /\bserve\(\s*'|cacheHeaders\(/.test(src)]).toEqual([r, true]);
        const key = (/serve\(\s*'([^']+)'/.exec(src) || /cacheHeaders\(\s*'([^']+)'/.exec(src))[1];
        expect([r, !!CDN_POLICY[key]]).toEqual([r, true]);
    }
});

test('a degraded answer is edge-cached briefly, never re-run on every same-minute read', () => {
    for (const key of Object.keys(CDN_POLICY)) {
        const a = maxAge(cacheHeaders(key, { payload: { _meta: { stale: true } } }));
        expect([key, a > 0, a <= DEGRADED_MAX_AGE]).toEqual([key, true, true]);
    }
    expect(DEGRADED_MAX_AGE).toBeLessThan(5 * 60); // the next 5-min refresh still retries live
});

test('price rows sit ≥ 1 h at the edge off-hours, but never past the next open', () => {
    const sat = at('2026-10-24T16:00:00Z'); // Sat noon ET
    const night = at('2026-10-22T02:00:00Z'); // Wed 22:00 ET
    const preOpen = at('2026-10-22T13:00:00Z'); // Thu 9:00 ET, 30 min to the open
    for (const key of ['spy', 'spy-daily-move', 'market-extra', 'sheets', 'fear-greed', 'vol', 'jev-pills']) {
        expect(maxAge(cacheHeaders(key, { payload: {}, now: sat }))).toBeGreaterThanOrEqual(3600);
        expect(maxAge(cacheHeaders(key, { payload: {}, now: night }))).toBeGreaterThanOrEqual(3600);
        expect(maxAge(cacheHeaders(key, { payload: {}, now: preOpen }))).toBe(1800 - CDN_POLICY[key][1]); // + swr ends at the open
        expect(maxAge(cacheHeaders(key, { payload: {}, now: at('2026-10-21T20:02:00Z') }))).toBe(CDN_POLICY[key][0]); // Wed 16:02 ET: close still settling
        expect(maxAge(cacheHeaders(key, { payload: {}, now: at('2026-10-21T20:06:00Z') }))).toBeGreaterThanOrEqual(3600);
        expect(maxAge(cacheHeaders(key, { payload: {}, now: at('2026-10-21T15:00:00Z') }))).toBe(CDN_POLICY[key][0]);
    }
});

test('the page never polls faster than 5 min open / 30 min otherwise', () => {
    expect(refreshMs(at('2026-10-21T15:00:00Z'))).toBeGreaterThanOrEqual(5 * 60e3);
    expect(refreshMs(at('2026-10-24T16:00:00Z'))).toBeGreaterThanOrEqual(30 * 60e3);
    const page = fs.readFileSync(path.join(__dirname, '..', 'app', 'page.js'), 'utf8');
    expect(page).toMatch(/!document\.hidden && .*refreshMs\(\)/); // hidden tabs don't poll
});

test('/api/freshness: a green reply is edge-cached ≥ 30 min (the probe period); a red or near-red one is not', () => {
    const green = { items: [{ name: 'served:spy', inputAgeH: 2, servedAgeH: 2, graceH: 6 }, { name: 'history-sheet', inputAgeH: null, servedAgeH: 5, graceH: 30, maxAgeH: 30 }] };
    expect(maxAge(freshnessHeaders(green))).toBeGreaterThan(1800);
    const nearRed = { items: [{ name: 'history-sheet', inputAgeH: null, servedAgeH: 29.5, graceH: 30, maxAgeH: 30 }] };
    expect(maxAge(freshnessHeaders(nearRed))).toBeLessThanOrEqual(DEGRADED_MAX_AGE);
    const red = { items: [{ name: 'served:fred', inputAgeH: 10, servedAgeH: null, graceH: 6 }] };
    expect(maxAge(freshnessHeaders(red))).toBeLessThanOrEqual(DEGRADED_MAX_AGE);
});
