/**
 * @jest-environment node
 *
 * /api/polymarket: Lambda board first; a list the Lambda could not load is filled
 * straight from Polymarket; a Lambda still on the old `{bets}` code (deploy race) or a
 * dead Lambda falls through to the direct build, which mirrors bot/fetchers.py.
 * Every call carries `?_fail=` (test mode: nothing is written to /tmp or KV) and `tmplg`
 * (never read a /tmp copy a local dev server may have left behind).
 */

let mode;
const calls = [];
const ev = (title, slug, markets, tags = []) => ({ title, slug, markets, tags, volume: 2e6, volume24hr: 2e5, endDate: '2026-11-03T00:00:00Z' });
const mkt = (label, yes, extra = {}) => ({
    groupItemTitle: label, outcomes: '["Yes","No"]', outcomePrices: JSON.stringify([String(yes), String(1 - yes)]),
    clobTokenIds: JSON.stringify([`tok-${label || 'yes'}-${yes}`, 'n']), oneDayPriceChange: 0.01, ...extra,
});
const FEATURED = { events: [
    ev('Balance of Power: 2026 Midterms', 'balance-of-power-2026-midterms', [mkt('Republicans Sweep', 0.08), mkt('Democrats Sweep', 0.62)]),
    ev("Ballon d'Or Winner 2026", 'ballon-dor-winner-2026', [mkt('Harry Kane', 0.66)]),
    ev('Bitcoin Up or Down - 5 min', 'btc-updown-5m', [mkt('Up', 0.5)]),
    ev('US recession by end of 2026?', 'us-recession-by-end-of-2026', [mkt('', 0.065)]),
    ev('Brazil Presidential Election', 'brazil-presidential-election', [mkt('Flávio Bolsonaro', 0.88)], [{ label: 'Macro Election 2', slug: 'macro-election-2' }]),
] };
const hist = (from, to, n = 60) => Array.from({ length: n }, (_, i) => ({ t: i, p: from + ((to - from) * i) / (n - 1) }));
const MOVERS = { markets: [
    { id: '1', question: 'Will Putin meet Lukashenko in Turkmenistan?', currentPrice: 0.015, livePriceChange: -54, events: [{ slug: 'putin-turkmenistan', volume: 79000 }], history: hist(0.555, 0.015) },
    { id: 's1', question: 'Sports thing', currentPrice: 0.5, livePriceChange: 40, events: [{ slug: 'sports', volume: 9e5 }], history: [] },
    { id: '2', question: 'Will Querétaro FC win?', currentPrice: 0.6, livePriceChange: 30, events: [{ slug: 'q', volume: 9e5 }], history: [] },
    { id: '3', question: 'Thin market?', currentPrice: 0.6, livePriceChange: 50, events: [{ slug: 'thin', volume: 2e4 }], history: [] },
] };
const GAMMA = {
    false: [{ question: 'Will SpaceXAI rename itself by October 15?', events: [{ slug: 'spacexai-rename' }], outcomePrices: '["0.865","0.135"]', oneDayPriceChange: 0.52, volumeNum: 62000, clobTokenIds: '["tok-sx","n"]', endDate: '2099-01-01T00:00:00Z' },
        { question: 'Saudi action against Yemen on October 6?', events: [{ slug: 'old' }], outcomePrices: '["0.7","0.3"]', oneDayPriceChange: 0.3, volumeNum: 9e5, endDate: '2020-10-06T00:00:00Z' }],
    true: [{ question: 'Will KMT win the most local elections?', events: [{ slug: 'kmt' }], outcomePrices: '["0.53","0.47"]', oneDayPriceChange: -0.345, volumeNum: 91000, clobTokenIds: '["tok-kmt","n"]' }],
};
const MACRO = {
    'macro-graph': [ev('US recession by end of 2026?', 'us-recession-by-end-of-2026', [mkt('', 0.065, { clobTokenIds: '["tok-rec","n"]', oneMonthPriceChange: -0.03 })])],
    'macro-single': [ev('Another Fed rate hike in 2026?', 'fed-hike-2026', [mkt('', 0.775, { clobTokenIds: '["tok-fed","n"]' })])],
};
const CLOB = { 'tok-rec': [0.085, 0.07, 0.065], 'tok-fed': [0.79, 0.775] };

jest.mock('../fetcher', () => ({
    fetchJson: jest.fn(async (url, opts) => {
        calls.push({ url, opts });
        const u = new URL(url);
        const q = Object.fromEntries(u.searchParams);
        if (u.pathname === '/events/keyset') return FEATURED;
        if (u.host === 'polymarket.com' && u.pathname === '/api/biggest-movers') {
            if (mode.movers === 'down') throw new Error('Fetch failed: 403');
            return q.category === 'sports' ? { markets: [{ id: 's1' }] } : MOVERS;
        }
        if (u.pathname === '/markets') return GAMMA[q.ascending];
        if (u.pathname === '/events') {
            if (mode.macro === 'down') throw new Error('Fetch failed: 500');
            return MACRO[q.tag_slug] || [];
        }
        if (u.pathname === '/prices-history') {
            return { history: (CLOB[q.market] || [0.3, 0.35, 0.4]).map((p, t) => ({ t, p })) };
        }
        throw new Error(`offline ${url}`);
    }),
}));

const { GET } = require('../../app/api/polymarket/route');
const call = async (q = '?_fail=none,tmplg') => {
    const res = await GET(new Request(`https://x.test/api/polymarket${q}`, { headers: { 'user-agent': 'jest' } }));
    return { res, b: await res.json() };
};
const LAMBDA_BOARD = {
    trending: [{ title: 'Balance of Power: 2026 Midterms', slug: 'balance-of-power-2026-midterms', outcomes: [{ label: 'Democrats Sweep', odds: 0.62, change: 0.01 }], nOutcomes: 2 },
        { title: 'US recession by end of 2026?', slug: 'us-recession-by-end-of-2026', outcomes: [{ label: 'Yes', odds: 0.065, change: null }], nOutcomes: 1 }],
    breaking: [{ question: 'From the Lambda', slug: 'lam', odds: 0.4, change: -0.2, volume: 1e5, spark: [0.6, 0.4] }],
    macro: [{ title: 'US recession by end of 2026?', slug: 'us-recession-by-end-of-2026', label: 'Yes', odds: 0.065, change: -0.02, volume: 1e6, spark: [0.085, 0.065] }],
    sources: { trending: 'featured', breaking: 'biggest-movers', macro: 'macro-tags' },
    source: 'Polymarket API', timestamp: '2026-10-11T04:00:00Z', error: null,
};
const lambdaAnswers = (body, status = 200) => {
    global.fetch = jest.fn(async () => ({ ok: status === 200, status, json: async () => body }));
};

beforeAll(() => { process.env.LAMBDA_URL = 'https://lambda.test'; });
afterAll(() => { delete process.env.LAMBDA_URL; });
beforeEach(() => { mode = {}; calls.length = 0; lambdaAnswers(LAMBDA_BOARD); });

describe('/api/polymarket', () => {
    test('a healthy Lambda board is served as-is, nothing fetched directly', async () => {
        const { res, b } = await call();
        expect(res.status).toBe(200);
        expect(global.fetch).toHaveBeenCalledWith('https://lambda.test/api/polymarket', expect.objectContaining({ cache: 'no-store' }));
        expect(calls).toHaveLength(0);
        expect(b.source).toBe('Polymarket API');
        expect(b.sources).toEqual(LAMBDA_BOARD.sources);
        expect(b._meta.hasErrors).toBe(false);
        expect(b.breaking[0].question).toBe('From the Lambda');
        // a market the Macro strip shows is not repeated under Trending
        expect(b.trending.map((t) => t.title)).toEqual(['Balance of Power: 2026 Midterms']);
    });

    test('a list the Lambda could not load is filled straight from Polymarket', async () => {
        lambdaAnswers({ ...LAMBDA_BOARD, breaking: [], sources: { ...LAMBDA_BOARD.sources, breaking: null } });
        const { b } = await call();
        expect(b.source).toBe('Polymarket API');
        expect(b.sources.breaking).toBe('biggest-movers (direct)');
        expect(b.breaking.map((m) => m.question)).toEqual(['Will Putin meet Lukashenko in Turkmenistan?']);
        expect(b._meta.hasErrors).toBe(false);
        expect(calls.some((c) => c.url.includes('/events/keyset'))).toBe(false);   // only what was missing
    });

    test('an empty list WITH a source is a quiet day: left alone, not an error', async () => {
        lambdaAnswers({ ...LAMBDA_BOARD, breaking: [] });
        const { b } = await call();
        expect(b.breaking).toEqual([]);
        expect(calls).toHaveLength(0);
        expect(b._meta.hasErrors).toBe(false);
    });

    test('a Lambda still on the old {bets} code falls through to the direct build', async () => {
        lambdaAnswers({ bets: [{ name: 'old', odds: 0.5, volume: 1 }], source: 'Polymarket API', error: null });
        const { b } = await call();
        expect(b.source).toBe('Polymarket Gamma API (fallback)');
        expect(b._meta.messages).toContain('Lambda returned no usable board');
        expect(b.trending.length).toBeGreaterThan(0);
        expect(b.bets).toBeUndefined();
    });

    test('direct build mirrors the Lambda curation (sports, coin flips, macro duplicates out)', async () => {
        const { b } = await call('?_fail=lambda,tmplg');
        expect(global.fetch).not.toHaveBeenCalled();
        expect(b.source).toBe('Polymarket Gamma API (fallback)');
        expect(b.sources).toEqual({ trending: 'featured', breaking: 'biggest-movers', macro: 'macro-tags' });
        expect(b.trending.map((t) => [t.title, t.topic])).toEqual([
            ['Balance of Power: 2026 Midterms', 'Politics'], ['Brazil Presidential Election', 'Politics']]);
        expect(b.trending[0].outcomes.map((o) => o.label)).toEqual(['Democrats Sweep', 'Republicans Sweep']);
        const [putin] = b.breaking;
        expect(b.breaking).toHaveLength(1);   // sports bucket, "FC" and < $50k are gone
        expect(putin).toMatchObject({ slug: 'putin-turkmenistan', odds: 0.015, change: -0.54, topic: 'Geopolitics' });
        expect(putin.spark.length).toBeLessThanOrEqual(48);
        expect(putin.spark[putin.spark.length - 1]).toBeCloseTo(0.015, 3);
        expect(b.macro.map((t) => [t.title, t.odds, t.change])).toEqual([
            ['US recession by end of 2026?', 0.065, -0.02], ['Another Fed rate hike in 2026?', 0.775, -0.015]]);
        expect(b._meta.hasErrors).toBe(false);
        // the ~2.5 MB featured feed skips Next's data cache; every timeout fits the deadline
        expect(calls.find((c) => c.url.includes('/events/keyset')).opts.revalidate).toBe(0);
        for (const c of calls) expect(c.opts.timeout).toBeGreaterThanOrEqual(1000);
        for (const c of calls) expect(c.opts.timeout).toBeLessThanOrEqual(15000);
    });

    test('movers feed down: Gamma backup, with its own sparklines and no ended markets', async () => {
        mode.movers = 'down';
        const { b } = await call('?_fail=lambda,tmplg');
        expect(b.sources.breaking).toBe('gamma');
        expect(b.breaking.map((m) => [m.question, m.change])).toEqual([
            ['Will SpaceXAI rename itself by October 15?', 0.52], ['Will KMT win the most local elections?', -0.345]]);
        expect(b.breaking[0].spark).toEqual([0.3, 0.35, 0.4]);
        expect(b._meta.messages.join(' ')).toMatch(/biggest-movers unavailable/);
    });

    test('a list with no source at all is flagged (never cached, never saved as last-good)', async () => {
        mode.macro = 'down';
        const { b } = await call('?_fail=lambda,tmplg');
        expect(b.macro).toEqual([]);
        expect(b.sources.macro).toBeNull();
        expect(b._meta.hasErrors).toBe(true);
        expect(b._meta.messages).toContain('no source answered for: macro');
        expect(require('../cdn').isDegraded(b)).toBe(true);
    });

    test('every tier off: an empty board labelled Unavailable', async () => {
        const { res, b } = await call('?_fail=lambda,gamma,lastgood');
        expect(res.status).toBe(200);
        expect(b).toMatchObject({ trending: [], breaking: [], macro: [] });
        expect(b._meta.source).toBe('Unavailable');
    });

    test('?debug=compare shows both paths side by side', async () => {
        const { b } = await call('?debug=compare');
        expect(b.lambda.trending.count).toBe(2);
        expect(b.fallback.trending.top[0]).toBe('Balance of Power: 2026 Midterms');
        expect(b.fallback.sources).toEqual({ trending: 'featured', breaking: 'biggest-movers', macro: 'macro-tags' });
    });
});
