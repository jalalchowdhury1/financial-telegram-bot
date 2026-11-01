/**
 * Jev pills never read a proxy or stale input as current (audit B3, 1 Nov 2026): the
 * F&G VIX proxy is not the CNN index, a stale FRED card is not this week's print, and the
 * "data through" date is the feed's own as-of, never today.
 */
import { assemblePills } from '../jevPills';
import { toData } from '../jevBrief';

const spy = { current: 700, ma200: { pct: -4 } };

test('a VIX-proxy F&G does not set dataAsOf.fg to today and makes no conflict pill', () => {
    const p = assemblePills({ raw: { fg: { score: 81, rating: 'EXTREME GREED', asOf: '2026-10-30', _meta: { proxy: true, source: 'CBOE VIX Proxy' } }, spy }, jevAnswers: null, yesterday: null, mode: 'rules' });
    expect(p.conflictPairs).toEqual([]);
    expect(p._meta.dataAsOf.fg).toBeNull();
});

test('a real CNN score carries CNN\'s own as-of date (New York), not the request day', () => {
    const p = assemblePills({ raw: { fg: { score: 26, rating: 'FEAR', asOf: '2026-10-30T23:59:48+00:00', _meta: { source: 'CNN' } }, spy }, jevAnswers: null, yesterday: null, mode: 'rules' });
    expect(p._meta.dataAsOf.fg).toBe('2026-10-30');
    expect(p.conflictPairs.length).toBe(0);   // fear + weak trend agree
});

test('a FRED card flagged stale is not read as a current input', () => {
    const d = toData({ fred: { indicators: { claims: { value: 230, stale: true }, sahmRule: { value: 0.2, stale: false } }, checklist: { nfci: { value: -0.5, stale: true } } } });
    expect(d.fred.claims).toBeNull();
    expect(d.fred.nfci).toBeNull();
    expect(d.fred.sahmRule).toBe(0.2);
});
