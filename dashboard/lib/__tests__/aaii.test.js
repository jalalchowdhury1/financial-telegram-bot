/**
 * @jest-environment node
 */
import fs from 'fs';
import path from 'path';
import {
    parseAaiiHtml, parseSubstackBody, parseSurveyDate, formatDiff, toPayload, ageDays,
    fetchAaiiLive, resolveAaii, AAII_URL, SUBSTACK_BASE,
} from '../aaii';

// The real aaii.com results table, saved 2026-09-27 (newest row Sep 23: 32.7/19.2/48.1).
const TABLE = fs.readFileSync(path.join(__dirname, 'fixtures', 'aaii-sent-results-2026-09-27.html'), 'utf8');
// Real prose of the Sep 23 survey post on insights.aaii.com (published Sat 2026-09-26).
const SUBSTACK_BODY = '<p>Pessimism among individual investors about the short-term outlook for stocks decreased in the latest AAII Sentiment Survey. Meanwhile, optimism and neutral sentiment increased.</p><p>Bullish sentiment, expectations that stock prices will rise over the next six months, increased 3.9 percentage points to 32.7%. Bullish sentiment is below its historical average of 37.5% for the eighth time in 10 weeks.</p><p>Neutral sentiment, expectations that stock prices will stay essentially unchanged over the next six months, increased 1.3 percentage points to 19.2%.</p><p>Bearish sentiment, expectations that stock prices will fall over the next six months, decreased 5.2 percentage points to 48.1%.</p>';
const NOW = new Date('2026-09-27T18:00:00Z');

describe('parseAaiiHtml', () => {
    test('reads the newest row of the real table', () => {
        expect(parseAaiiHtml(TABLE, NOW)).toEqual({ bull: 32.7, neutral: 19.2, bear: 48.1, as_of: '2026-09-23' });
    });
    test('loose pass survives an attribute reshuffle', () => {
        const html = '<table><tr><td class="x">Sep 23</td><td>32.7%</td><td>19.2%</td><td>48.1%</td></tr></table>';
        expect(parseAaiiHtml(html, NOW)).toEqual({ bull: 32.7, neutral: 19.2, bear: 48.1, as_of: '2026-09-23' });
    });
    test('rejects rows that do not sum to 100 and junk input', () => {
        expect(parseAaiiHtml('<td class="tableTxt">Sep 23</td><td class="tableTxt">50%</td><td class="tableTxt">50%</td><td class="tableTxt">50%</td>', NOW)).toBeNull();
        expect(parseAaiiHtml('', NOW)).toBeNull();
        expect(parseAaiiHtml(null, NOW)).toBeNull();
        expect(parseAaiiHtml('<html>Access denied</html>', NOW)).toBeNull();
    });
});

describe('parseSubstackBody', () => {
    test('real prose → same numbers as aaii.com, survey date = the Wednesday before the post', () => {
        expect(parseSubstackBody(SUBSTACK_BODY, '2026-09-26T15:30:46.947Z', NOW))
            .toEqual({ bull: 32.7, neutral: 19.2, bear: 48.1, as_of: '2026-09-23' });
    });
    test('a Thursday post maps to the day before', () => {
        expect(parseSubstackBody(SUBSTACK_BODY, '2026-09-24T15:00:00Z', NOW).as_of).toBe('2026-09-23');
    });
    test('"week ending" wins when present', () => {
        expect(parseSubstackBody(`For the week ending September 16, ${SUBSTACK_BODY}`, '2026-09-26T15:30:00Z', NOW).as_of).toBe('2026-09-16');
    });
    test('missing numbers → null', () => {
        expect(parseSubstackBody('<p>Bullish sentiment increased.</p>', '2026-09-26T15:30:00Z', NOW)).toBeNull();
    });
});

describe('format + staleness', () => {
    test('diff is bear − bull in the sheet E2 format', () => {
        expect(formatDiff(32.7, 48.1)).toBe('15.40%');
        expect(formatDiff(45, 30)).toBe('-15.00%');
    });
    test('stale only past 9 days', () => {
        const row = { bull: 32.7, neutral: 19.2, bear: 48.1, as_of: '2026-09-23' };
        expect(toPayload(row, 'aaii.com', NOW)).toEqual({ bull: 32.7, neutral: 19.2, bear: 48.1, diff: '15.40%', as_of: '2026-09-23', source: 'aaii.com', stale: false });
        expect(toPayload(row, 'aaii.com', new Date('2026-10-02T12:00:00Z')).stale).toBe(false); // 9 days
        expect(toPayload(row, 'aaii.com', new Date('2026-10-03T12:00:00Z')).stale).toBe(true);  // 10 days
        expect(ageDays('garbage', NOW)).toBe(Infinity);
    });
    test('year rollover: a Dec date read in January is last year', () => {
        expect(parseSurveyDate('Dec 31', new Date('2027-01-02T00:00:00Z')).toISOString().slice(0, 10)).toBe('2026-12-31');
    });
});

// A fake fetchText routed by URL; `fail` lists URL prefixes that throw.
function fakeFetch({ fail = [] } = {}) {
    const calls = [];
    const fn = async (url) => {
        calls.push(url);
        if (fail.some((p) => url.startsWith(p))) throw new Error(`Fetch failed for ${url}: 403 Forbidden`);
        if (url === AAII_URL) return TABLE;
        if (url.startsWith(`${SUBSTACK_BASE}/api/v1/archive`)) return JSON.stringify([
            { title: 'Weekly market notes', slug: 'weekly-notes', post_date: '2026-09-27T10:00:00Z' },
            { title: 'AAII Sentiment Survey: Pessimism Pulls Back', slug: 'aaii-sentiment-survey-pessimism-pulls-cd1', post_date: '2026-09-26T15:30:46.947Z' },
        ]);
        if (url.startsWith(`${SUBSTACK_BASE}/api/v1/posts/`)) return JSON.stringify({ body_html: SUBSTACK_BODY });
        if (url === `${SUBSTACK_BASE}/feed`) return `<rss><channel><item><title><![CDATA[AAII Sentiment Survey: Pessimism Pulls Back]]></title><pubDate>Sat, 26 Sep 2026 15:30:46 GMT</pubDate><content:encoded><![CDATA[${SUBSTACK_BODY}]]></content:encoded></item></channel></rss>`;
        throw new Error(`unexpected ${url}`);
    };
    fn.calls = calls;
    return fn;
}

function memStore() {
    const m = new Map();
    return {
        m,
        load: (k, maxAge) => {
            const v = m.get(k);
            if (!v) return null;
            if (maxAge && Date.now() - new Date(v.savedAt).getTime() > maxAge) return null;
            return v;
        },
        save: (k, data) => m.set(k, { data, savedAt: new Date().toISOString() }),
    };
}

describe('fetchAaiiLive tiers', () => {
    test('aaii.com first', async () => {
        const r = await fetchAaiiLive({ fetchText: fakeFetch(), now: NOW });
        expect(r.payload).toMatchObject({ source: 'aaii.com', diff: '15.40%', as_of: '2026-09-23' });
    });
    test('aaii.com blocked → Substack API, same week and numbers', async () => {
        const r = await fetchAaiiLive({ fetchText: fakeFetch({ fail: [AAII_URL] }), now: NOW });
        expect(r.payload).toEqual({ bull: 32.7, neutral: 19.2, bear: 48.1, diff: '15.40%', as_of: '2026-09-23', source: 'substack', stale: false });
    });
    test('API blocked too → Substack RSS', async () => {
        const r = await fetchAaiiLive({ fetchText: fakeFetch({ fail: [AAII_URL, `${SUBSTACK_BASE}/api`] }), now: NOW });
        expect(r.payload).toMatchObject({ source: 'substack', as_of: '2026-09-23', diff: '15.40%' });
        expect(r.messages.join(' ')).toMatch(/substack rss: ok/);
    });
    test('everything blocked → null payload, every tier named', async () => {
        const r = await fetchAaiiLive({ fetchText: fakeFetch({ fail: ['https://'] }), now: NOW });
        expect(r.payload).toBeNull();
        expect(r.messages).toHaveLength(3);
    });
    test('injected fault skips a tier without fetching it', async () => {
        const f = fakeFetch();
        const trip = (n) => { if (n === 'aaii_http') throw new Error('[injected fault: aaii_http]'); };
        const r = await fetchAaiiLive({ fetchText: f, trip, now: NOW });
        expect(r.payload.source).toBe('substack');
        expect(f.calls).not.toContain(AAII_URL);
    });
});

describe('resolveAaii caching', () => {
    test('second call inside 3 h is served from the cache (no fetch)', async () => {
        const store = memStore();
        const f = fakeFetch();
        await resolveAaii({ fetchText: f, store, now: NOW });
        const n = f.calls.length;
        const r = await resolveAaii({ fetchText: f, store, now: NOW });
        expect(f.calls.length).toBe(n);
        expect(r.payload.diff).toBe('15.40%');
        expect(r.cachedAt).toBeTruthy();
    });
    test('a cached copy is re-judged for staleness at read time', async () => {
        const store = memStore();
        await resolveAaii({ fetchText: fakeFetch(), store, now: NOW });
        const r = await resolveAaii({ fetchText: fakeFetch(), store, now: new Date('2026-10-05T00:00:00Z') });
        expect(r.payload.stale).toBe(true);
    });
    test('all tiers down → last good, flagged; with aaii_lastgood → null (route answers 503)', async () => {
        const store = memStore();
        store.save('aaii-live', { bull: 32.7, neutral: 19.2, bear: 48.1, diff: '15.40%', as_of: '2026-09-23', source: 'aaii.com', stale: false });
        store.m.get('aaii-live').savedAt = new Date(Date.now() - 5 * 3600e3).toISOString(); // older than the 3 h cache
        const down = fakeFetch({ fail: ['https://'] });
        const lg = await resolveAaii({ fetchText: down, store, now: NOW });
        expect(lg.lastGood).toBe(true);
        expect(lg.payload.diff).toBe('15.40%');
        const none = await resolveAaii({ fetchText: down, store, faults: new Set(['aaii_lastgood']), now: NOW });
        expect(none.payload).toBeNull();
    });
    test('never goes backwards: a live tier with an OLDER survey loses to the newest one in KV', async () => {
        const store = memStore();
        const kvMap = new Map();
        const kv = { get: async (k) => kvMap.get(k) ?? null, set: async (k, v) => { kvMap.set(k, v); return true; } };
        const later = new Date('2026-10-09T18:00:00Z');
        kvMap.set('ftb:aaii:newest', { data: { bull: 40, neutral: 21.3, bear: 38.7, diff: '-1.30%', as_of: '2026-10-07', source: 'aaii.com', stale: false }, savedAt: '2026-10-08T20:00:00Z' });
        // live answers with the 23 Sep survey (the fixtures): older than KV's 7 Oct
        const r = await resolveAaii({ fetchText: fakeFetch(), store, kv, now: later });
        expect(r.payload.as_of).toBe('2026-10-07');
        expect(r.payload.diff).toBe('-1.30%');
        expect(r.messages.join(' ')).toMatch(/older survey \(2026-09-23\)/);
        expect(kvMap.get('ftb:aaii:newest').data.as_of).toBe('2026-10-07'); // not overwritten
    });
    test('a NEWER live survey replaces the KV copy; KV also backs a cold instance when every tier is down', async () => {
        const kvMap = new Map();
        const kv = { get: async (k) => kvMap.get(k) ?? null, set: async (k, v) => { kvMap.set(k, v); return true; } };
        await resolveAaii({ fetchText: fakeFetch(), store: memStore(), kv, now: NOW });
        expect(kvMap.get('ftb:aaii:newest').data.as_of).toBe('2026-09-23');
        const cold = await resolveAaii({ fetchText: fakeFetch({ fail: ['https://'] }), store: memStore(), kv, now: NOW });
        expect(cold.lastGood).toBe(true);
        expect(cold.payload.diff).toBe('15.40%');
    });
    test('a newer KV survey beats this instance\'s 3 h cache', async () => {
        const store = memStore();
        await resolveAaii({ fetchText: fakeFetch(), store, now: NOW }); // caches 23 Sep here
        const kv = { get: async () => ({ data: { bull: 40.3, neutral: 20.8, bear: 39, diff: '-1.30%', as_of: '2026-09-25', source: 'macromicro', stale: false }, savedAt: NOW.toISOString() }), set: async () => true };
        const r = await resolveAaii({ fetchText: fakeFetch(), store, kv, now: NOW });
        expect(r.payload.diff).toBe('-1.30%');
        expect(r.payload.source).toBe('macromicro');
    });
    test('fault-test calls never write the cache', async () => {
        const store = memStore();
        await resolveAaii({ fetchText: fakeFetch(), store, faults: new Set(['aaii_http']), now: NOW });
        expect(store.m.size).toBe(0);
    });
});
