import { pickSeen, mergeSeen, readSeen, writeSeen, sinceLastVisit, SEEN_KEY, SINCE_MIN_GAP_MS } from '../lastVisit';

const T0 = new Date(2026, 8, 24, 9, 5).getTime(); // Thu 09:05 local
const HOUR = 3600e3;
const live = (o = {}) => ({
    spy: { current: 780 }, vol: { vixDay: { value: 13.8 } },
    fg: { score: 43, _meta: { source: 'CNN' } }, extra: { rates: { tnx: { current: 5.13 } } }, ...o,
});
const base = () => mergeSeen(null, pickSeen(
    { spy: { current: 770 }, vol: { vixDay: { value: 15 } }, fg: { score: 37, _meta: { source: 'CNN' } }, extra: { rates: { tnx: { current: 5.18 } } } },
    { claims: 202.25, sahmRule: 0.03, peRatio: 26.1 },
), T0);

test('pickSeen keeps live numbers only; F&G only from CNN/RapidAPI; prints only "print" metrics', () => {
    const p = pickSeen(live({ fg: { score: 40, _meta: { source: 'Yahoo ^VIX Proxy' } } }), { claims: 200, peRatio: 26, junk: 'x' });
    expect(p.v).toEqual({ spy: 780, vix: 13.8, fg: null, tnx: 5.13 });
    expect(p.p).toEqual({ claims: 200 }); // peRatio is a "move" metric
    expect(pickSeen({ spy: { error: 'x' } }).v.spy).toBeNull();
});

test('mergeSeen: a field that is not live now keeps its old value and time', () => {
    const m = mergeSeen(base(), { v: { spy: 781, vix: null }, p: {} }, T0 + 2 * HOUR);
    expect(m.v.spy).toEqual({ x: 781, at: T0 + 2 * HOUR });
    expect(m.v.vix).toEqual({ x: 15, at: T0 });
});

test('mergeSeen stamps each field with the time ITS feed landed', () => {
    const m = mergeSeen(null, { v: { spy: 780, vix: 14, tnx: 5.1 }, p: { claims: 205, rentIndex: 1800, aaiiDiff: 3 } }, T0 + 9 * HOUR,
        { spy: T0, vol: T0 + HOUR, fred: T0 + 2 * HOUR, sheets: T0 + 3 * HOUR });
    expect([m.v.spy.at, m.v.vix.at, m.v.tnx.at]).toEqual([T0, T0 + HOUR, T0 + 9 * HOUR]); // extra not landed → now
    expect([m.p.claims.at, m.p.rentIndex.at, m.p.aaiiDiff.at]).toEqual([T0 + 2 * HOUR, T0 + 9 * HOUR, T0 + 3 * HOUR]);
});

test('since last visit: SPY %, VIX %, F&G points, 10Y bp and new prints', () => {
    const d = sinceLastVisit(base(), pickSeen(live(), { claims: 205, sahmRule: 0.03 }), T0 + 50 * HOUR);
    expect(d.since).toBe('Thu 09:05');
    expect(d.items.map((i) => `${i.label} ${i.text}`)).toEqual(['SPY +1.3%', 'VIX −8.0%', 'F&G +6', '10Y −5bp']);
    expect(d.prints).toEqual(['Initial Claims (4wk)']);
    expect(d.more).toBe(0);
});

test('nothing under an hour after that visit (a reload is not a visit); no base → nothing', () => {
    expect(sinceLastVisit(base(), pickSeen(live()), T0 + SINCE_MIN_GAP_MS - 1)).toBeNull();
    expect(sinceLastVisit(null, pickSeen(live()), T0 + 50 * HOUR)).toBeNull();
});

test('a field last seen far from that visit is left out, never compared across the wrong gap', () => {
    const b = base();
    b.v.tnx.at = T0 - 3 * 24 * HOUR;
    b.p.claims.at = T0 - 3 * 24 * HOUR;
    const d = sinceLastVisit(b, pickSeen(live(), { claims: 205 }), T0 + 50 * HOUR);
    expect(d.items.map((i) => i.key)).toEqual(['spy', 'vix', 'fg']);
    expect(d.prints).toEqual([]);
});

test('unchanged numbers read "flat"', () => {
    const same = pickSeen({ spy: { current: 770 }, vol: { vixDay: { value: 15 } } });
    expect(sinceLastVisit(base(), same, T0 + 5 * HOUR).items.map((i) => i.text)).toEqual(['flat', 'flat']);
});

test('storage: round-trips, drops junk fields, never throws', () => {
    writeSeen(base(), window.localStorage);
    expect(readSeen(window.localStorage)).toEqual(base());
    window.localStorage.setItem(SEEN_KEY, JSON.stringify({ v: { spy: { x: 'NaN', at: 1 }, vix: { x: 15, at: T0 } } }));
    expect(readSeen(window.localStorage)).toEqual({ v: { vix: { x: 15, at: T0 } }, p: {} });
    window.localStorage.setItem(SEEN_KEY, '{not json');
    expect(readSeen(window.localStorage)).toBeNull();
    const boom = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('full'); } };
    expect(readSeen(boom)).toBeNull();
    expect(writeSeen(base(), boom)).toBe(false);
});
