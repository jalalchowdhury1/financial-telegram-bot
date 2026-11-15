/**
 * @jest-environment node
 *
 * Centred: the public feed's validator, its 30 h stale rule / freshness item, and /api/centred
 * (good, bad, gist down → last good, nothing at all → a visible "no feed"). Synthetic feeds only.
 */
import fs from 'fs';
import { validateCentred } from '../centred';
import { centredItem, centredStale } from '../servedFreshness';
import { saveLastGood } from '../store';
import { compareLines } from '../../components/CentredDetails';
import { GET } from '../../app/api/centred/route';

const NOW = Date.parse('2026-11-15T12:00:00Z');
const good = { v: 1, generated_at: '2026-11-15T04:10:00-05:00', prices_to: '2026-11-13', state: 'off', note: '', checked: 5, off: 2,
    since: '2026-11-08', verdict: 'no harm', delta: { cagr: -0.4, maxdd: 0.9, sharpe: 0.01 }, history: [{ date: '2026-11-14', off: 2, checked: 5 }] };

describe('validateCentred', () => {
    it('accepts the public feed shape', () => expect(validateCentred(good)).toBe(true));
    it('accepts a first-night feed (unknown, nothing judged yet)', () => expect(validateCentred({ v: 1, generated_at: '2026-11-14T21:40:00-05:00', prices_to: null,
        state: 'unknown', note: 'first check runs tonight', checked: 0, off: 0, since: null, verdict: 'not judged', delta: null, history: [] })).toBe(true));
    it.each([
        null, {}, { ...good, v: 2 }, { ...good, state: 'odd' }, { ...good, off: 6 }, { ...good, checked: -1 }, { ...good, off: 1.5 },
        { ...good, generated_at: 'soon' }, { ...good, generated_at: null }, { ...good, verdict: 'great' }, { ...good, delta: { cagr: 1 } },
        { ...good, history: [{ date: 'x', off: 1, checked: 5 }] }, { ...good, history: 'no' },
    ])('rejects %j', (s) => expect(validateCentred(s)).toBe(false));
});

describe('stale rule + freshness item', () => {
    it('fresh under 30 h, red past it', () => {
        expect(centredItem(good, NOW)).toEqual({ name: 'centred', inputAgeH: null, servedAgeH: 2.8, graceH: 30, maxAgeH: 30 });
        expect(centredStale(good, NOW)).toBe(false);
        expect(centredStale(good, NOW + 28 * 3600e3)).toBe(true);
    });
    it('nothing served: no age but the cap stays, which the fleet grades red ("nothing served")', () => {
        expect(centredItem(null, NOW)).toMatchObject({ servedAgeH: null, maxAgeH: 30 });
    });
});

test('compare lines: plain words, short enough for a phone', () => {
    expect(compareLines({ cagr: -0.4, maxdd: 0.9, sharpe: 0.01 })).toEqual(['Moving them: gain −0.4 %/yr', 'Worst fall 0.9 pts smaller']);
    expect(compareLines({ cagr: 1.25, maxdd: -2, sharpe: 0 })).toEqual(['Moving them: gain +1.3 %/yr', 'Worst fall 2.0 pts bigger']);
    expect(compareLines(null)).toEqual(['Moving them: not judged yet', null]);
});

describe('/api/centred', () => {
    const LG = '/tmp/lg-centred.json';
    let saved = null;
    beforeAll(() => { if (fs.existsSync(LG)) saved = fs.readFileSync(LG); });
    afterAll(() => { if (saved) fs.writeFileSync(LG, saved); else if (fs.existsSync(LG)) fs.unlinkSync(LG); });
    const answer = (status, body) => { global.fetch = jest.fn(async () => ({ ok: status === 200, status, json: async () => body })); };
    const call = async (q) => (await GET(new Request(`https://x.test/api/centred${q}`, { headers: { 'user-agent': 'jest' } }))).json();
    const fresh = { ...good, generated_at: new Date().toISOString() };

    it('relays a good feed from the gist', async () => {
        answer(200, fresh);
        const b = await call('?_fail=none,tmplg');
        expect(global.fetch.mock.calls[0][0]).toMatch(/^https:\/\/gist\.githubusercontent\.com\/.+\/centred-public\.json\?t=\d+$/);
        expect(b).toMatchObject({ state: 'off', off: 2, checked: 5, _meta: { source: 'gist', stale: false } });
    });
    it('a bad feed is never relayed: the "no feed" fallback instead', async () => {
        answer(200, { ...fresh, off: 9 });
        const b = await call('?_fail=lastgood');
        expect(b).toMatchObject({ state: 'unknown', note: 'no feed', generated_at: null, _meta: { source: 'Unavailable', hasErrors: true } });
    });
    it('gist down → the last good copy, flagged stale', async () => {
        saveLastGood('centred', fresh);
        answer(500, {});
        const b = await call('?_fail=gist');
        expect(b).toMatchObject({ state: 'off', off: 2, _meta: { stale: true, hasErrors: true } });
    });
    it('gist down and no copy → "no feed", never a throw', async () => {
        answer(500, {});
        expect(await call('?_fail=gist,lastgood')).toMatchObject({ state: 'unknown', note: 'no feed' });
    });
});
