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
import { todayVerdict, ownRead, nearest } from '../../components/TodayCard';
import { GET } from '../../app/api/own/route';
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
        expect(JSON.stringify(out)).not.toMatch(/sid|dollars|conditions|threshold|ticker|symphony|usd|AAA|ABCDEF/);
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
    ])('refuses a feed with %s', (_, s) => {
        expect(publicOwn(s, NOW)).toBeNull();
        expect(validateOwn(s, NOW)).toBe(false);
    });
    it('a note that looks like a condition or dollars is blanked, the alarm kept', () => {
        for (const note of ['AAA under 30', 'RSI(10) < 30', '$12,000 down', 'x'.repeat(29)]) {
            expect(publicOwn({ ...fixture, alarms: [{ ...fixture.alarms[0], note }] }, NOW).alarms[0].note).toBe('');
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
        const none = { v: 1, made: null, asof: null, alarms: [], race: null };
        expect(ownItem(none, NOW)).toMatchObject({ servedAgeH: null, maxAgeH: 30 });
        expect(ownStale(none, NOW)).toBe(true);
        expect(ownRead(none, NOW)).toBeNull();
        expect(ownRead(undefined, NOW)).toBeUndefined();
        expect(ownRead(fixture, NOW)).toBe(fixture);
        expect(ownRead(fixture, NOW + 31 * 3600e3)).toBeNull();
    });
});

describe('the Today verdict', () => {
    const calm = { fr: 'BIL (T-Bill ETF)', frMode: 'Cash', split: { main: 71, out: false, totalsOld: false }, centred: { off: 0 } };
    const tripped = { ...fixture, alarms: fixture.alarms.map((a) => (a.id === 'ballast2' ? { ...a, state: 'tripped' } : a)) };
    const texts = (v) => v.parts.map((p) => `${p.tone}:${p.text}`);
    it('everything read and quiet → Nothing to do', () => expect(todayVerdict(calm, fixture)).toEqual({ quiet: true, parts: [] }));
    it('feed missing → never quiet: Alarms unread, in grey', () => expect(texts(todayVerdict(calm, null))).toEqual(['unread:Alarms unread']));
    it('split unread → Split unread', () => expect(texts(todayVerdict({ ...calm, split: null }, fixture))).toEqual(['unread:Split unread']));
    it('one alarm unread inside a read feed → never quiet', () => {
        const one = { ...fixture, alarms: fixture.alarms.map((a, i) => (i === 0 ? { ...a, state: 'unread' } : a)) };
        expect(texts(todayVerdict(calm, one))).toEqual(['unread:1 alarm unread']);
    });
    it('a tripped alarm speaks by its name', () => expect(texts(todayVerdict(calm, tripped))).toEqual(['warn:Ballast in falls']));
    it('most important first: split, alarms, dip day, off centre, then the unread ones', () => {
        const v = todayVerdict({ fr: 'Select bottom 2', frMode: 'Dip buy', split: { main: 82, out: true }, centred: { off: 1 } }, tripped);
        expect(texts(v)).toEqual(['warn:Split 82 / 18', 'warn:Ballast in falls', 'warn:Dip day', 'warn:1 off centre']);
        expect(texts(todayVerdict({ ...calm, split: { main: 82, out: true }, fr: null }, null)))
            .toEqual(['warn:Split 82 / 18', 'unread:Alarms unread', 'unread:FrontRunner unread']);
    });
    it('a crashed Signals card (null) reads as unread everywhere, never quiet', () => {
        expect(texts(todayVerdict(null, fixture))).toEqual(['unread:Split unread', 'unread:FrontRunner unread', 'unread:Centred unread']);
    });
    it('still loading with nothing to say → null (the card shows …)', () => {
        expect(todayVerdict(undefined, fixture)).toBeNull();
        expect(todayVerdict(calm, undefined)).toBeNull();
    });
    it('nearest = the top line\'s closest: read, not firing, highest share', () => {
        expect(nearest([null, { fire: 'X', share: 9, gap: 'a' }, { fire: null, share: 0.5, gap: 'b' }, { fire: false, share: 0.9, gap: 'c' }]).gap).toBe('c');
        expect(nearest(null)).toBeNull();
    });
});

describe('/api/own', () => {
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
        expect(b).toMatchObject({ made: null, alarms: [], race: null, _meta: { source: 'Unavailable', hasErrors: true } });
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
