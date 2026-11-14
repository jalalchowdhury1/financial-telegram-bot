import { validateSplit } from '../split';
import { cacheHeaders } from '../cdn';

describe('validateSplit', () => {
    const good = { asOf: '2026-11-13', main: 71.1, target: 70, band: 10, anchored: '2026-11-13' };
    it('accepts the Mac snapshot', () => expect(validateSplit(good)).toBe(true));
    it.each([null, {}, { ...good, main: null }, { ...good, main: 140 }, { ...good, asOf: null }, { ...good, band: 'x' }])(
        'rejects %j', (s) => expect(validateSplit(s)).toBe(false));
});

test('the split is never held at the edge past 11 min, market open or shut (the 04:00 run lands off-hours)', () => {
    const ok = { asOf: '2026-11-13', main: 71.12, _meta: { hasErrors: false } };
    for (const now of ['2026-11-14T21:00:00Z', '2026-11-17T09:00:00Z', '2026-11-17T15:00:00Z']) {
        expect(cacheHeaders('split', { payload: ok, now: Date.parse(now) })['vercel-cdn-cache-control']).toBe('max-age=300, stale-while-revalidate=360');
    }
});
