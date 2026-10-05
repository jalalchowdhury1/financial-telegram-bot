import { readSnap, writeSnap, savedLabel, purgeOldSnaps, isLiveAnswer, SNAP_PREFIX, SNAP_MAX_AGE_MS } from '../snapshot';

const memStore = () => {
    const m = new Map();
    return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), m };
};
const NOW = new Date(2026, 8, 26, 12, 0).getTime();

test('write then read returns the payload and its save time', () => {
    const s = memStore();
    expect(writeSnap('spy', { current: 771.35 }, NOW, s)).toBe(true);
    expect(readSnap('spy', NOW + 1000, s)).toEqual({ savedAt: NOW, data: { current: 771.35 } });
    expect([...s.m.keys()]).toEqual([`${SNAP_PREFIX}spy`]);
});

test('error payloads and non-objects are never saved', () => {
    const s = memStore();
    expect(writeSnap('spy', { error: 'down' }, NOW, s)).toBe(false);
    expect(writeSnap('spy', null, NOW, s)).toBe(false);
    expect(writeSnap('spy', 'x', NOW, s)).toBe(false);
    expect(s.m.size).toBe(0);
});

test('too old, future-stamped, corrupt or missing copies read as null', () => {
    const s = memStore();
    writeSnap('a', { v: 1 }, NOW, s);
    expect(readSnap('a', NOW + SNAP_MAX_AGE_MS + 1, s)).toBeNull();
    expect(readSnap('a', NOW - 5 * 60e3, s)).toBeNull(); // saved "in the future"
    s.setItem(`${SNAP_PREFIX}b`, '{not json');
    expect(readSnap('b', NOW, s)).toBeNull();
    s.setItem(`${SNAP_PREFIX}c`, JSON.stringify({ savedAt: 'x', data: {} }));
    expect(readSnap('c', NOW, s)).toBeNull();
    expect(readSnap('nope', NOW, s)).toBeNull();
});

test('blocked or full storage never throws', () => {
    const boom = { getItem: () => { throw new Error('SecurityError'); }, setItem: () => { throw new Error('QuotaExceeded'); } };
    expect(readSnap('spy', NOW, boom)).toBeNull();
    expect(writeSnap('spy', { v: 1 }, NOW, boom)).toBe(false);
    expect(readSnap('spy', NOW, null)).toBeNull();
    expect(writeSnap('spy', { v: 1 }, NOW, null)).toBe(false);
});

test('savedLabel: time today, weekday + time on another day', () => {
    expect(savedLabel(new Date(2026, 8, 26, 10, 42).getTime(), NOW)).toBe('10:42');
    expect(savedLabel(new Date(2026, 8, 24, 9, 5).getTime(), NOW)).toBe('Thu 09:05');
    expect(savedLabel(new Date(2026, 8, 17, 9, 5).getTime(), NOW)).toBe('Thu Sep 17 09:05'); // > 6 days: dated
    expect(savedLabel(NaN, NOW)).toBe('');
});

test('copies from other deploys are purged; this deploy and other keys stay', () => {
    const ls = window.localStorage;
    ls.setItem('fd:snap:v1:spy', '{}'); // before copies were keyed to the deploy
    ls.setItem('fd:snap:v1:abc1234:spy', '{}'); // an older deploy
    ls.setItem(`${SNAP_PREFIX}spy`, '{}');
    ls.setItem('fd:factor-win', '1y');
    expect(purgeOldSnaps(ls)).toBe(2);
    expect(Object.keys(ls).sort()).toEqual(['fd:factor-win', `${SNAP_PREFIX}spy`].sort());
    expect(purgeOldSnaps(null)).toBe(0);
});

test('isLiveAnswer: each route\'s 200-with-fallback body is not a live answer', () => {
    // the fallback bodies the routes really serve (app/api/*/route.js)
    expect(isLiveAnswer('extra', { fx: {}, commodities: {}, rates: {}, realEstate: {} })).toBe(false);
    expect(isLiveAnswer('vol', { updated_at: null, tickers: [], _meta: { source: 'Unavailable' } })).toBe(false);
    expect(isLiveAnswer('spyDailyMove', { value: null, source: 'Unavailable' })).toBe(false);
    expect(isLiveAnswer('history', { today: null, metrics: {} })).toBe(false);
    expect(isLiveAnswer('fg', { _meta: { source: 'Failed' } })).toBe(false);
    expect(isLiveAnswer('jev', { enabled: true, mode: 'rules', pills: null })).toBe(false);
    expect(isLiveAnswer('sheets', { NotSoBoring: 'N/A', FrontRunner: 'N/A', AAIIDiff: 'N/A', VIX: { current: 'N/A' } })).toBe(false);
    expect(isLiveAnswer('spy', { error: 'SPY temporarily unavailable' })).toBe(false);
    expect(isLiveAnswer('fred', null)).toBe(false);
    // real answers
    expect(isLiveAnswer('extra', { rates: { tnx: { current: 4.1 } } })).toBe(true);
    expect(isLiveAnswer('vol', { tickers: [{ ticker: 'SPY' }] })).toBe(true);
    expect(isLiveAnswer('spyDailyMove', { value: '0.54%' })).toBe(true);
    expect(isLiveAnswer('fg', { score: 37, _meta: { source: 'Stale Cache' } })).toBe(true); // real CNN numbers, labelled by its card
    expect(isLiveAnswer('jev', { enabled: false })).toBe(true); // kill switch: hide the row
    expect(isLiveAnswer('sheets', { NotSoBoring: 'OFF', FrontRunner: 'N/A' })).toBe(true);
    expect(isLiveAnswer('spy', { current: 771.35 })).toBe(true);
    expect(isLiveAnswer('fred', { yieldCurve: { current: 0.36 } })).toBe(true);
});

// The real /api/jev-pills answer, saved 2026-10-04: `pills` is an OBJECT keyed by pill
// (regime/recession/breadth/hedging/conflict), not an array. An Array.isArray test never
// passed, so the Jev card was never saved and the page jumped ~525px on every warm open.
const jevLive = require('./fixtures/jev-pills-2026-10-04.json');

test('isLiveAnswer(jev): the real payload is live, so it is saved and painted on the next open', () => {
    expect(Array.isArray(jevLive.pills)).toBe(false);
    expect(Object.keys(jevLive.pills)).toEqual(['regime', 'recession', 'breadth', 'hedging', 'conflict']);
    expect(isLiveAnswer('jev', jevLive)).toBe(true);
    const s = memStore();
    expect(writeSnap('jev', jevLive, NOW, s)).toBe(true);
    const back = readSnap('jev', NOW + 1000, s);
    expect(back.savedAt).toBe(NOW);
    expect(isLiveAnswer('jev', back.data)).toBe(true); // the saved copy reads back as the same live shape
});

test('isLiveAnswer(jev): no pills, empty pills or a junk value are not live; the kill switch is', () => {
    expect(isLiveAnswer('jev', { ...jevLive, pills: null })).toBe(false);
    expect(isLiveAnswer('jev', { ...jevLive, pills: {} })).toBe(false);
    expect(isLiveAnswer('jev', { ...jevLive, pills: [] })).toBe(false);
    expect(isLiveAnswer('jev', { ...jevLive, pills: 'x' })).toBe(false);
    expect(isLiveAnswer('jev', { enabled: true })).toBe(false);
    expect(isLiveAnswer('jev', {})).toBe(false);
    expect(isLiveAnswer('jev', { ...jevLive, error: 'down' })).toBe(false);
    expect(isLiveAnswer('jev', { enabled: false })).toBe(true); // JEV_PILLS=off: hide the row
});
