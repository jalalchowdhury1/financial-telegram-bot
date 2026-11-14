import { validateSplit } from '../split';

describe('validateSplit', () => {
    const good = { asOf: '2026-11-13', main: 71.1, target: 70, band: 10, anchored: '2026-11-13' };
    it('accepts the Mac snapshot', () => expect(validateSplit(good)).toBe(true));
    it.each([null, {}, { ...good, main: null }, { ...good, main: 140 }, { ...good, asOf: null }, { ...good, band: 'x' }])(
        'rejects %j', (s) => expect(validateSplit(s)).toBe(false));
});
