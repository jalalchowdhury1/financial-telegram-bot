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
import { levels, marbles } from '../../components/CentredDetails';
import { GET } from '../../app/api/centred/route';

const NOW = Date.parse('2026-11-15T12:00:00Z');
const good = { v: 1, generated_at: '2026-11-15T04:10:00-05:00', prices_to: '2026-11-13', state: 'off', note: '', checked: 5, off: 2,
    since: '2026-11-08', verdict: 'no harm', delta: { cagr: -0.4, maxdd: 0.9, sharpe: 0.01 }, history: [{ date: '2026-11-14', off: 2, checked: 5 }] };

describe('validateCentred', () => {
    it('accepts the public feed shape', () => expect(validateCentred(good)).toBe(true));
    it('accepts a first-night feed (unknown, nothing judged yet)', () => expect(validateCentred({ v: 1, generated_at: '2026-11-14T21:40:00-05:00', prices_to: null,
        state: 'unknown', note: 'first check runs tonight', checked: 0, off: 0, since: null, verdict: 'not judged', delta: null, history: [] })).toBe(true));
    it('accepts an edge count (at an edge: not off, not in the middle)', () => expect(validateCentred({ ...good, edge: 3 })).toBe(true));
    it.each([
        { ...good, edge: 4 }, { ...good, edge: -1 }, { ...good, worlds: 'many' },
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

test('spirit levels: bubble dead centre when nothing changes, right = better, left = worse, never past the vial', () => {
    expect(levels({ cagr: -0.4, maxdd: 0.9, sharpe: 0.01 })).toEqual([
        { label: 'Gain', text: '−0.4 %/yr', x: -0.08, tone: 'is-bad' }, { label: 'Worst fall', text: '0.9 pts smaller', x: 0.18, tone: 'is-good' },
        { label: 'Sharpe', text: '+0.01', x: 0.04, tone: 'is-good' }]);
    expect(levels({ cagr: 40, maxdd: -0.04, sharpe: 0 }).map((l) => [l.text, l.x, l.tone])).toEqual([['+40.0 %/yr', 1, 'is-good'], ['same', 0, ''], ['same', 0, '']]);
    expect(levels(null)).toBe(null);
});

test('the table: every setting is one marble; edges sit at both rims, off ones hang over the lips', () => {
    const m = marbles({ checked: 31, off: 2, edge: 11 });
    expect([m.mid, m.edge, m.off, m.list.length]).toEqual([18, 11, 2, 31]);
    const on = m.list.filter((b) => b.kind !== 'off');
    expect(on.map((b) => b.kind).join('')).toBe('edge'.repeat(6) + 'mid'.repeat(18) + 'edge'.repeat(5));
    expect(on.every((b, i) => b.x > 22 && b.x < 298 && (i === 0 || b.x - on[i - 1].x >= 2 * b.r))).toBe(true);   // on the top, never overlapping
    expect(m.list.filter((b) => b.kind === 'off').map((b) => b.side)).toEqual([-1, 1]);
    expect(marbles({ checked: 5, off: 0 })).toMatchObject({ mid: 5, edge: 0, off: 0 });           // an older feed has no edge count
    expect(marbles({ checked: 200, off: 3, edge: 500 }).list).toHaveLength(200);                    // odd counts never draw extra marbles
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
