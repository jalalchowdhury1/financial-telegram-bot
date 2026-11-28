/**
 * @jest-environment node
 *
 * Own watch: the allow-list that rebuilds the public feed, its 30 h unread rule / freshness item, /api/own (good,
 * junk stripped, bad, gist down → last good, nothing at all → unread), and the Today card's verdict. Synthetic feeds only.
 */
import fs from 'fs';
import { publicOwn, validateOwn } from '../own';
import { ownItem, ownStale } from '../servedFreshness';
import { saveLastGood } from '../store';
import { todayVerdict, ownRead, nearText, holdsText } from '../../components/TodayCard';
import { GET } from '../../app/api/own/route';
import { CDN_POLICY } from '../cdn';
import fixture from './own.fixture.json';

const NOW = Date.parse('2026-11-28T12:00:00Z');

describe('publicOwn: the allow-list', () => {
    it('keeps the contract fixture as is', () => expect(publicOwn(fixture, NOW)).toEqual(fixture));
    it('drops every field it does not know, at every level', () => {
        const junk = { ...fixture, sid: 'ABCDEF123', dollars: 123456, conditions: ['AAA RSI(10) < 30'],
            alarms: fixture.alarms.map((a) => ({ ...a, threshold: 30, ticker: 'AAA' })),
            race: { ...fixture.race, symphony: 'xyz', rows: fixture.race.rows.map((r) => ({ ...r, sid: 'abc', usd: 99 })) } };
        const out = publicOwn(junk, NOW);
        expect(out).toEqual(fixture);
        expect(JSON.stringify(out)).not.toMatch(/"sid"|dollars|conditions|threshold|ticker|symphony|usd|AAA|ABCDEF/);
    });
    it.each([
        ['v != 1', { ...fixture, v: 2 }],
        ['no made', { ...fixture, made: undefined }],
        ['made as free text', { ...fixture, made: 'yesterday' }],
        ['made far in the future', { ...fixture, made: '2026-12-30T06:25:00-05:00' }],
        ['no alarms', { ...fixture, alarms: [] }],
        ['an unknown alarm state', { ...fixture, alarms: [{ ...fixture.alarms[0], state: 'fine' }] }],
        ['a ticker in an alarm name', { ...fixture, alarms: [{ ...fixture.alarms[0], name: 'SOXL slide' }] }],
        ['a name too long', { ...fixture, alarms: [{ ...fixture.alarms[0], name: 'x'.repeat(29) }] }],
        ['a full-width name', { ...fixture, alarms: [{ ...fixture.alarms[0], name: 'Ｂａｌｌａｓｔ slide' }] }],
        ['two alarms with one id', { ...fixture, alarms: [fixture.alarms[0], { ...fixture.alarms[1], id: fixture.alarms[0].id }] }],
    ])('refuses a feed with %s', (_, s) => {
        expect(publicOwn(s, NOW)).toBeNull();
        expect(validateOwn(s, NOW)).toBe(false);
    });
    it('a note that looks like a condition or dollars is blanked, the alarm kept', () => {
        for (const note of ['AAA under 30', 'RSI(10) < 30', '$12,000 down', 'x'.repeat(29), '€12,345', '12,345 usd', 'ｄｏｗｎ 3%', 'a_b']) {
            expect(publicOwn({ ...fixture, alarms: [{ ...fixture.alarms[0], note }] }, NOW).alarms[0].note).toBe('');
        }
    });
    it('every real name and note passes as written', () => {
        const names = ['Ballast slide', 'Ballast in falls', 'Book v old ballast', 'Dip engine'];
        const notes = ['6 months +11%', '6 months −23%', '0 of 2 strikes', '2 of 2 strikes', 'last 40 dip days +165', 'last 40 dip days −0',
            'price pack behind', 'not read', 'Main rules changed', 'old ballast not served', 'numbers old'];
        for (const name of names) for (const note of notes) {
            expect(publicOwn({ ...fixture, alarms: [{ id: 'a', name, state: 'ok', note }] }, NOW).alarms[0]).toEqual({ id: 'a', name, state: 'ok', note });
        }
        const rows = ['Faster dip', 'Guard off', 'From scratch', 'Simple'].map((name, i) => ({ id: `r${i}`, name, vs: 0, fall: 0 }));
        expect(publicOwn({ ...fixture, race: { ...fixture.race, rows } }, NOW).race.rows.map((r) => r.name)).toEqual(['Faster dip', 'Guard off', 'From scratch', 'Simple']);
    });
    it('near: a gap above 0 never rounds down to 0 ("on"); 0 stays 0', () => {
        expect(publicOwn({ ...fixture, near: { gap: 0.04, side: 'dip' } }, NOW).near).toEqual({ gap: 0.1, side: 'dip' });
        expect(publicOwn({ ...fixture, near: { gap: 0.001, side: 'hot' } }, NOW).near).toEqual({ gap: 0.1, side: 'hot' });
        expect(publicOwn({ ...fixture, near: { gap: 0, side: 'dip' } }, NOW).near).toEqual({ gap: 0, side: 'dip' });
    });
    it('near: gap 0..100 rounded to 1 decimal, side dip or hot; anything else (or missing) is null, the feed still reads', () => {
        expect(publicOwn({ ...fixture, near: { gap: 11.14, side: 'hot', ticker: 'AAA', level: 30 } }, NOW).near).toEqual({ gap: 11.1, side: 'hot' });
        const { near: _, ...without } = fixture;
        for (const n of [undefined, null, { gap: -1, side: 'dip' }, { gap: 101, side: 'dip' }, { gap: '4', side: 'dip' }, { gap: NaN, side: 'dip' }, { gap: 4, side: 'DIP' }, { gap: 4 }]) {
            const out = publicOwn(n === undefined ? without : { ...fixture, near: n }, NOW);
            expect(out.near).toBeNull();
            expect(out.alarms).toHaveLength(4);
        }
    });
    it('holds summing to 95..105 pass (rounding), outside that they are dropped alone', () => {
        expect(publicOwn({ ...fixture, holds: [{ kind: 'bonds', pct: 50 }, { kind: 'cash', pct: 45 }] }, NOW).holds).toHaveLength(2);
        expect(publicOwn({ ...fixture, holds: [{ kind: 'bonds', pct: 55 }, { kind: 'cash', pct: 50 }] }, NOW).holds).toHaveLength(2);
    });
    it('holds: 1 to 6 known kinds with whole-number shares; anything else (or missing) is null, the feed still reads', () => {
        expect(publicOwn({ ...fixture, holds: [{ kind: 'stocks', pct: 100, ticker: 'AAA' }] }, NOW).holds).toEqual([{ kind: 'stocks', pct: 100 }]);
        const { holds: _, ...without } = fixture;
        for (const h of [undefined, null, [], [{ kind: 'gold', pct: 50 }], [{ kind: 'bonds', pct: 50.5 }], [{ kind: 'bonds', pct: 101 }], [{ kind: 'bonds', pct: -1 }],
            [{ kind: 'bonds' }], 'bonds', Array(7).fill({ kind: 'cash', pct: 10 }), [{ kind: 'bonds', pct: 50 }, { kind: 'cash', pct: 44 }], [{ kind: 'bonds', pct: 60 }, { kind: 'cash', pct: 46 }]]) {
            const out = publicOwn(h === undefined ? without : { ...fixture, holds: h }, NOW);
            expect(out.holds).toBeNull();
            expect(out.alarms).toHaveLength(4);
        }
    });
    it('a broken race is dropped alone; the alarms still read', () => {
        for (const race of [null, { ...fixture.race, rows: [] }, { ...fixture.race, rows: [{ ...fixture.race.rows[0], vs: 'up' }] }, { ...fixture.race, day: -1 }]) {
            const out = publicOwn({ ...fixture, race }, NOW);
            expect(out.race).toBeNull();
            expect(out.alarms).toHaveLength(4);
        }
    });
});

describe('30 h unread rule + freshness item', () => {
    it('fresh under 30 h, red past it', () => {
        expect(ownItem(fixture, NOW)).toEqual({ name: 'own', inputAgeH: null, servedAgeH: 0.6, graceH: 30, maxAgeH: 30 });
        expect(ownStale(fixture, NOW)).toBe(false);
        expect(ownStale(fixture, NOW + 30 * 3600e3)).toBe(true);
    });
    it('nothing served (the route\'s fallback): no age, the cap stays, and it reads as unread', () => {
        const none = { v: 1, made: null, asof: null, holds: null, near: null, alarms: [], race: null };
        expect(ownItem(none, NOW)).toMatchObject({ servedAgeH: null, maxAgeH: 30 });
        expect(ownStale(none, NOW)).toBe(true);
        expect(ownRead(none, NOW)).toBeNull();
        expect(ownRead(undefined, NOW)).toBeUndefined();
        expect(ownRead(fixture, NOW)).toBe(fixture);
        expect(ownRead(fixture, NOW + 31 * 3600e3)).toBeNull();
    });
});

describe('the Today verdict', () => {
    const calm = { split: { main: 71, out: false, totalsOld: false }, centred: { off: 0 } };
    const tripped = { ...fixture, alarms: fixture.alarms.map((a) => (a.id === 'ballast2' ? { ...a, state: 'tripped' } : a)) };
    const texts = (v) => v.parts.map((p) => `${p.tone}:${p.text}`);
    it('everything read and quiet → Nothing to do', () => expect(todayVerdict(calm, fixture)).toEqual({ quiet: true, parts: [] }));
    it('feed missing → never quiet: Alarms unread, in grey', () => expect(texts(todayVerdict(calm, null))).toEqual(['unread:Alarms unread']));
    it('split unread → Split unread', () => expect(texts(todayVerdict({ ...calm, split: null }, fixture))).toEqual(['unread:Split unread']));
    it('one alarm unread inside a read feed → never quiet; two say alarms', () => {
        const one = { ...fixture, alarms: fixture.alarms.map((a, i) => (i === 0 ? { ...a, state: 'unread' } : a)) };
        expect(texts(todayVerdict(calm, one))).toEqual(['unread:1 alarm unread']);
        const two = { ...fixture, alarms: fixture.alarms.map((a, i) => (i < 2 ? { ...a, state: 'unread' } : a)) };
        expect(texts(todayVerdict(calm, two))).toEqual(['unread:2 alarms unread']);
    });
    it('Split totals old is said, in amber', () => expect(texts(todayVerdict({ ...calm, split: { ...calm.split, totalsOld: true } }, fixture))).toEqual(['warn:Split totals old']));
    it.each([['holds', 'Holds unread', 'holds'], ['near', 'Nearest unread', 'near'], ['race', 'Race unread', 'race']])('a read feed with %s null → never quiet: %s, in grey', (k, text, cell) => {
        expect(todayVerdict(calm, { ...fixture, [k]: null })).toEqual({ quiet: false, parts: [{ text, cell, tone: 'unread' }] });
    });
    it('the worst day: every part on one list, amber first', () => {
        const v = todayVerdict({ ...calm, split: { ...calm.split, totalsOld: true } }, { ...tripped, holds: null, race: null, alarms: tripped.alarms.map((a) => (a.id === 'dip' ? { ...a, state: 'unread' } : a)) });
        expect(texts(v)).toEqual(['warn:Ballast in falls', 'warn:Split totals old', 'unread:1 alarm unread', 'unread:Holds unread', 'unread:Race unread']);
    });
    it('a tripped alarm speaks by its name', () => expect(texts(todayVerdict(calm, tripped))).toEqual(['warn:Ballast in falls']));
    it('most important first: split, alarms, dip day, off centre, then the unread ones', () => {
        const v = todayVerdict({ split: { main: 82, out: true }, centred: { off: 1 } }, { ...tripped, near: { gap: 0, side: 'dip' } });
        expect(texts(v)).toEqual(['warn:Split 82 / 18', 'warn:Ballast in falls', 'warn:Dip day', 'warn:1 off centre']);
        expect(texts(todayVerdict({ ...calm, split: { main: 82, out: true } }, null))).toEqual(['warn:Split 82 / 18', 'unread:Alarms unread']);
    });
    it('a crashed Signals card (null) reads as unread everywhere, never quiet', () => {
        expect(texts(todayVerdict(null, fixture))).toEqual(['unread:Split unread', 'unread:Centred unread']);
    });
    it('still loading with nothing to say → null (the card shows …)', () => {
        expect(todayVerdict(undefined, fixture)).toBeNull();
        expect(todayVerdict(calm, undefined)).toBeNull();
    });
    it('the Nearest cell: points and side, "on" at 0, null when not read', () => {
        expect(nearText({ gap: 11.1, side: 'hot' })).toBe('11.1 pts · hot side');
        expect(nearText({ gap: 3, side: 'dip' })).toBe('3.0 pts · dip side');
        expect(nearText({ gap: 0, side: 'dip' })).toBe('on · dip side');
        expect(nearText(null)).toBeNull();
    });
    it('near off its line adds no verdict item', () => {
        for (const near of [{ gap: 0.1, side: 'hot' }, { gap: 0.1, side: 'dip' }]) expect(todayVerdict(calm, { ...fixture, near })).toEqual({ quiet: true, parts: [] });
    });
    it('Hot day: gap 0 on the hot side, amber like Dip day, tapping opens Nearest (the cell is amber too)', () => {
        expect(todayVerdict(calm, { ...fixture, near: { gap: 0, side: 'hot' } }).parts).toEqual([{ text: 'Hot day', cell: 'near', tone: 'warn' }]);
    });
    it('Dip day comes only from Main\'s own rule: gap 0 on the dip side, amber, tapping opens Nearest', () => {
        expect(todayVerdict(calm, { ...fixture, near: { gap: 0, side: 'dip' } }).parts).toEqual([{ text: 'Dip day', cell: 'near', tone: 'warn' }]);
    });
    it('the Holds cell: one kind = its label, more = top 3 with shares then +N, null = not read', () => {
        expect(holdsText([{ kind: 'stocks', pct: 100 }])).toBe('Stocks 3x');
        expect(holdsText(fixture.holds)).toBe('Bonds 50 · Cash 50');
        expect(holdsText([{ kind: 'stocks', pct: 50 }, { kind: 'fear', pct: 25 }, { kind: 'bonds', pct: 25 }])).toBe('Stocks 3x 50 · Fear fund 25 · Bonds 25');
        expect(holdsText([{ kind: 'stocks', pct: 40 }, { kind: 'bonds', pct: 30 }, { kind: 'cash', pct: 20 }, { kind: 'short', pct: 10 }])).toBe('Stocks 3x 40 · Bonds 30 · Cash 20 +1');
        expect(holdsText(null)).toBeNull();
    });
});

describe('/api/own', () => {
    it('the edge keeps a copy at most 10 min (an alarm ping links straight to the card)', () => expect(CDN_POLICY.own).toEqual([300, 300]));
    const LG = '/tmp/lg-own.json';
    let saved = null;
    beforeAll(() => { if (fs.existsSync(LG)) saved = fs.readFileSync(LG); });
    afterAll(() => { if (saved) fs.writeFileSync(LG, saved); else if (fs.existsSync(LG)) fs.unlinkSync(LG); });
    const answer = (status, body) => { global.fetch = jest.fn(async () => ({ ok: status === 200, status, json: async () => body })); };
    const call = async (q) => (await GET(new Request(`https://x.test/api/own${q}`, { headers: { 'user-agent': 'jest' } }))).json();
    const fresh = { ...fixture, made: new Date().toISOString() };

    it('relays a good feed from the split\'s gist', async () => {
        answer(200, fresh);
        const b = await call('?_fail=none,tmplg');
        expect(global.fetch.mock.calls[0][0]).toMatch(/^https:\/\/gist\.githubusercontent\.com\/.+\/own\.json\?t=\d+$/);
        expect(b).toMatchObject({ v: 1, asof: '2026-11-27', race: { day: 1 }, _meta: { source: 'gist', stale: false } });
        expect(b.alarms).toHaveLength(4);
    });
    it('junk fields in the gist never reach the site', async () => {
        answer(200, { ...fresh, sid: 'SECRET1', alarms: fresh.alarms.map((a) => ({ ...a, line: 'AAA RSI(10) < 30' })) });
        const b = await call('?_fail=none,tmplg');
        expect(JSON.stringify(b)).not.toMatch(/SECRET1|AAA|"line"|"sid"/);
    });
    it('an old feed is relayed flagged stale (the card reads it as unread)', async () => {
        answer(200, { ...fixture, made: '2026-01-02T06:25:00-05:00' });
        const b = await call('?_fail=none,tmplg');
        expect(b._meta).toMatchObject({ stale: true, hasErrors: true });
        expect(ownRead(publicOwn(b))).toBeNull();
    });
    it.each([['v != 1', { ...fresh, v: 2 }], ['missing (404)', null]])('a feed %s is never relayed: the unread fallback', async (_, body) => {
        answer(body ? 200 : 404, body);
        const b = await call('?_fail=lastgood');
        expect(b).toMatchObject({ made: null, holds: null, near: null, alarms: [], race: null, _meta: { source: 'Unavailable', hasErrors: true } });
        expect(ownRead(publicOwn(b))).toBeNull();
    });
    it('gist down → the last good copy, flagged stale', async () => {
        saveLastGood('own', publicOwn(fresh));
        answer(500, {});
        expect(await call('?_fail=gist')).toMatchObject({ asof: '2026-11-27', _meta: { stale: true, hasErrors: true } });
    });
    it('gist down and no copy → unread, never a throw', async () => {
        answer(500, {});
        expect(await call('?_fail=gist,lastgood')).toMatchObject({ made: null, alarms: [] });
    });
});
