/**
 * @jest-environment node
 *
 * lib/spyTiers.js — the direct tiers under /api/spy and /api/spy-daily-move, and the
 * two routes' wiring. Every upstream is mocked; `now` is pinned per test.
 */
jest.mock('../sources', () => ({
    ...jest.requireActual('../sources'),
    polygonDaily: jest.fn(),
    finnhubQuote: jest.fn(),
    cnbcQuotes: jest.fn(),
    nasdaqHistory: jest.fn(),
    nasdaqQuote: jest.fn(),
    yahooChart: jest.fn(),
}));

const src = require('../sources');
const { latestSessionDate, buildSpy, fallbackSpy, fallbackMove, return3yFrom, threeYearsBefore } = require('../spyTiers');

// Fri 9 Oct 2026, 14:00 ET (market open) → latest session = 2026-10-09.
const OPEN = Date.parse('2026-10-09T18:00:00Z');
// Fri 9 Oct 2026, 08:00 ET (pre-open) → latest session = Thu 2026-10-08.
const PRE = Date.parse('2026-10-09T12:00:00Z');

/** n weekday bars ending on `last`, price = base + i*step (oldest->newest). */
function bars(n, last, { base = 400, step = 0.5 } = {}) {
    const out = [];
    let t = Date.parse(`${last}T12:00:00Z`);
    while (out.length < n) {
        const d = new Date(t);
        if (d.getUTCDay() !== 0 && d.getUTCDay() !== 6) out.push(d.toISOString().slice(0, 10));
        t -= 864e5;
    }
    return out.reverse().map((date, i) => ({ date, price: base + i * step }));
}
const poly = (history) => ({ history, current: history[history.length - 1].price, prevClose: history[history.length - 2].price });
const cnbc = (price, change, asOf) => ({ SPY: { price, change, changePct: (change / (price - change)) * 100, asOf, lastTime: `${asOf}T16:00:00.000-0400` } });
const ENV = { POLYGON_KEY: 'p', FINNHUB_KEY: 'f' };
const F = (...names) => new Set(names);

beforeEach(() => {
    jest.resetAllMocks();
    for (const k of ['polygonDaily', 'finnhubQuote', 'cnbcQuotes', 'nasdaqHistory', 'nasdaqQuote', 'yahooChart']) {
        src[k].mockRejectedValue(new Error(`${k} offline`));
    }
});

describe('latestSessionDate', () => {
    test.each([
        ['2026-10-09T18:00:00Z', '2026-10-09'], // Fri, open
        ['2026-10-09T13:31:00Z', '2026-10-09'], // Fri 9:31 ET
        ['2026-10-09T13:29:00Z', '2026-10-08'], // Fri 9:29 ET → Thu
        ['2026-10-10T18:00:00Z', '2026-10-09'], // Sat → Fri
        ['2026-10-12T03:00:00Z', '2026-10-09'], // Sun 23:00 ET → Fri
        ['2026-11-26T18:00:00Z', '2026-11-25'], // Thanksgiving → Wed
        ['2026-10-09T23:30:00Z', '2026-10-09'], // Fri evening
    ])('%s → %s', (iso, want) => expect(latestSessionDate(Date.parse(iso))).toBe(want));
});

describe('buildSpy / return3yFrom', () => {
    test('3Y = last close on/before the same date 3 years earlier (owner pick 2026-10-09)', () => {
        // Real SPY: 2026-10-09 → 2023-10-09 (432.29) → 80.10%. The Sheet's 1095-days rule
        // landed on 2023-10-10 (79.17%) because 2024 was a leap year.
        const h = [
            { date: '2023-10-05', price: 424.5 }, { date: '2023-10-06', price: 429.54 },
            { date: '2023-10-09', price: 432.29 }, { date: '2023-10-10', price: 434.54 },
            { date: '2026-10-09', price: 778.57 },
        ];
        expect(return3yFrom(h, 778.57)).toBeCloseTo(80.10, 2);
        // anniversary on a weekend (2023-10-08 was a Sunday) → the Friday before
        expect(return3yFrom(h, 778.57, '2026-10-08')).toBeCloseTo(((778.57 - 429.54) / 429.54) * 100, 6);
    });
    test('threeYearsBefore: same date, Feb 29 → Feb 28', () => {
        expect(threeYearsBefore('2026-10-09')).toBe('2023-10-09');
        expect(threeYearsBefore('2028-02-29')).toBe('2025-02-28');
    });
    test('3Y anchors on the live spot\'s date, not the last bar (Polygon ends yesterday)', () => {
        const h = [
            { date: '2023-10-06', price: 429.54 }, { date: '2023-10-09', price: 432.29 },
            { date: '2023-10-10', price: 434.54 }, { date: '2026-10-08', price: 774 },
        ];
        expect(return3yFrom(h, 778.57)).toBeCloseTo(((778.57 - 429.54) / 429.54) * 100, 6); // anchored on 10-08
        expect(return3yFrom(h, 778.57, '2026-10-09')).toBeCloseTo(80.10, 2);              // anchored on spot's day
    });
    test('3Y is null when the bars do not reach 3 years back (never a 2Y return labelled 3Y)', () => {
        expect(return3yFrom(bars(500, '2026-10-08'), 120)).toBeNull();
        expect(return3yFrom([], 120)).toBeNull();
        expect(return3yFrom(bars(800, '2026-10-08'), 120)).not.toBeNull();
    });
    test('extra.return3y fills only a missing 3Y', () => {
        const h = bars(300, '2026-10-08');
        expect(buildSpy(h, 600, 590, 'X', { return3y: 42 }).return3y).toBe(42);
        expect(buildSpy(bars(800, '2026-10-08'), 600, 590, 'X', { return3y: 42 }).return3y).not.toBe(42);
    });
    test('throws under 220 bars', () => expect(() => buildSpy(bars(219, '2026-10-08'), 1, 1, 'X')).toThrow(/insufficient/));
});

describe('fallbackSpy (/api/spy direct tiers)', () => {
    test('Polygon + Finnhub spot; 3Y filled from Nasdaq bars that line up', async () => {
        const p = bars(500, '2026-10-08');                  // ~2y, like the free tier
        const nq = bars(1000, '2026-10-09', { base: 400 - 499 * 0.5 }); // same price on the same dates
        src.polygonDaily.mockResolvedValue(poly(p));
        src.finnhubQuote.mockResolvedValue({ current: 700, prevClose: p[p.length - 1].price });
        src.nasdaqHistory.mockResolvedValue(nq);
        const out = await fallbackSpy([], F(), { env: ENV, now: OPEN });
        expect(out._meta.source).toBe('Polygon + Finnhub (fallback)');
        expect(out.current).toBe(700);
        const upto = nq.filter((b) => b.date <= '2026-10-08');
        const base = upto.filter((b) => b.date <= '2023-10-09').pop().price; // spot's day 2026-10-09 → 2023-10-09
        expect(out.return3y).toBeCloseTo(((700 - base) / base) * 100, 6);
        expect(out._meta.stale).toBeUndefined();
    });

    test('Nasdaq bars that do NOT line up leave 3Y null (honest N/A)', async () => {
        const p = bars(500, '2026-10-08');
        src.polygonDaily.mockResolvedValue(poly(p));
        src.finnhubQuote.mockResolvedValue({ current: 700, prevClose: 649.5 });
        src.nasdaqHistory.mockResolvedValue(bars(1000, '2026-10-09', { base: 10 }));
        const msgs = [];
        const out = await fallbackSpy(msgs, F(), { env: ENV, now: OPEN });
        expect(out.return3y).toBeNull();
        expect(msgs.join(' ')).toMatch(/3Y from Nasdaq unavailable: bars do not line up/);
    });

    test('_fail=finnhub → CNBC spot overlays Polygon', async () => {
        const p = bars(800, '2026-10-08');
        src.polygonDaily.mockResolvedValue(poly(p));
        src.cnbcQuotes.mockResolvedValue(cnbc(800, 0.5, '2026-10-09'));
        const out = await fallbackSpy([], F('finnhub'), { env: ENV, now: OPEN });
        expect(src.finnhubQuote).not.toHaveBeenCalled();
        expect(out._meta.source).toBe('Polygon + CNBC (fallback)');
        expect(out.current).toBe(800);
        expect(out.dailyChange.value).toBeCloseTo(0.5);
        expect(out.return3y).not.toBeNull(); // 800 bars → computed from Polygon itself
        expect(src.nasdaqHistory).not.toHaveBeenCalled();
    });

    test('_fail=polygon → Nasdaq tier with CNBC spot, full 3Y', async () => {
        src.nasdaqHistory.mockResolvedValue(bars(1000, '2026-10-08'));
        src.cnbcQuotes.mockResolvedValue(cnbc(900, 1, '2026-10-09'));
        const out = await fallbackSpy([], F('polygon', 'finnhub'), { env: ENV, now: OPEN });
        expect(src.polygonDaily).not.toHaveBeenCalled();
        expect(out._meta.source).toBe('Nasdaq + CNBC (fallback)');
        expect(out.current).toBe(900);
        expect(out.return3y).not.toBeNull();
        expect(out.chartHistory).toHaveLength(302);
    });

    test('a CNBC quote from an older day is rejected; Nasdaq bars dated today stand alone', async () => {
        const nq = bars(1000, '2026-10-09');
        src.nasdaqHistory.mockResolvedValue(nq);
        src.cnbcQuotes.mockResolvedValue(cnbc(900, 1, '2026-10-08'));
        const msgs = [];
        const out = await fallbackSpy(msgs, F('polygon', 'finnhub'), { env: ENV, now: OPEN });
        expect(out._meta.source).toBe('Nasdaq (fallback)');
        expect(out.current).toBe(nq[999].price);
        expect(msgs.join(' ')).toMatch(/CNBC quote dated 2026-10-08, latest session 2026-10-09/);
    });

    test('an implausible spot (>15% off the bars) is ignored', async () => {
        const nq = bars(1000, '2026-10-09');
        src.nasdaqHistory.mockResolvedValue(nq);
        src.cnbcQuotes.mockResolvedValue(cnbc(nq[999].price * 2, 1, '2026-10-09'));
        const out = await fallbackSpy([], F('polygon', 'finnhub'), { env: ENV, now: OPEN });
        expect(out._meta.source).toBe('Nasdaq (fallback)');
    });

    test('no tier has the latest session + no spot → flagged-stale build, relabelled', async () => {
        src.polygonDaily.mockResolvedValue(poly(bars(800, '2026-10-08')));
        const out = await fallbackSpy([], F('finnhub', 'cnbc', 'nasdaq', 'yahoo'), { env: ENV, now: OPEN });
        expect(out._meta).toMatchObject({ stale: true, hasErrors: true, asOf: '2026-10-08', source: 'Polygon (fallback, last close 2026-10-08)' });
    });

    test('a stale Polygon is passed over for a fresh Nasdaq', async () => {
        src.polygonDaily.mockResolvedValue(poly(bars(800, '2026-10-08')));
        src.nasdaqHistory.mockImplementation(async () => bars(1000, '2026-10-09'));
        const out = await fallbackSpy([], F('finnhub', 'cnbc'), { env: ENV, now: OPEN });
        expect(out._meta.source).toBe('Nasdaq (fallback)');
        expect(out._meta.stale).toBeUndefined();
    });

    test('pre-open: yesterday-dated bars ARE the latest session → live label', async () => {
        src.polygonDaily.mockResolvedValue(poly(bars(800, '2026-10-08')));
        const out = await fallbackSpy([], F('finnhub', 'cnbc'), { env: ENV, now: PRE });
        expect(out._meta.source).toBe('Polygon (fallback)');
        expect(out._meta.stale).toBeUndefined();
    });

    test('Yahoo: prevClose from the bars, never chartPreviousClose (= close before the RANGE)', async () => {
        const h = bars(1000, '2026-10-09');
        src.yahooChart.mockResolvedValue({ history: h, current: h[999].price, prevClose: 100, meta: { regularMarketTime: OPEN / 1000, chartPreviousClose: 100 } });
        const out = await fallbackSpy([], F('polygon', 'nasdaq', 'finnhub', 'cnbc'), { env: ENV, now: OPEN });
        expect(out._meta.source).toBe('Yahoo Finance (fallback)');
        expect(out.dailyChange.value).toBeCloseTo(0.5);
    });

    test('everything down → throws (serve() then uses last-known-good)', async () => {
        await expect(fallbackSpy([], F(), { env: ENV, now: OPEN })).rejects.toThrow(/all SPY tiers failed/);
    });

    test('no POLYGON_KEY → straight to Nasdaq', async () => {
        src.nasdaqHistory.mockResolvedValue(bars(1000, '2026-10-09'));
        const msgs = [];
        const out = await fallbackSpy(msgs, F(), { env: {}, now: OPEN });
        expect(out._meta.source).toBe('Nasdaq (fallback)');
        expect(msgs).toContain('POLYGON_KEY not configured');
    });
});

describe('fallbackMove (/api/spy-daily-move direct tiers)', () => {
    test('Finnhub first', async () => {
        src.finnhubQuote.mockResolvedValue({ current: 778.57, prevClose: 773.93 });
        expect(await fallbackMove([], F(), { env: ENV, now: OPEN })).toEqual({ value: '+0.60%', source: 'Finnhub (fallback)' });
    });

    test('_fail=finnhub → CNBC change_pct for the latest session', async () => {
        src.cnbcQuotes.mockResolvedValue({ SPY: { price: 778.57, change: 4.64, changePct: 0.5995, asOf: '2026-10-09' } });
        expect(await fallbackMove([], F('finnhub'), { env: ENV, now: OPEN })).toEqual({ value: '+0.60%', source: 'CNBC (fallback)', asOf: '2026-10-09' });
    });

    test('_fail=finnhub,cnbc → Nasdaq quote for the latest session; a stale one is skipped', async () => {
        src.nasdaqQuote.mockResolvedValue({ current: 778.57, changePct: 0.6, asOf: '2026-10-09' });
        expect(await fallbackMove([], F('finnhub', 'cnbc'), { env: ENV, now: OPEN })).toEqual({ value: '+0.60%', source: 'Nasdaq (fallback)', asOf: '2026-10-09' });
        src.nasdaqQuote.mockResolvedValue({ current: 773.93, changePct: -0.2, asOf: '2026-10-08' });
        const msgs = [];
        await expect(fallbackMove(msgs, F('finnhub', 'cnbc', 'polygon', 'yahoo'), { env: ENV, now: OPEN })).rejects.toThrow();
        expect(msgs.join(' ')).toMatch(/Nasdaq failed: quote dated 2026-10-08 is not the latest session/);
    });

    test('Polygon with only YESTERDAY\'s bar is skipped, never shown as today\'s move', async () => {
        const p = bars(250, '2026-10-08');
        src.polygonDaily.mockResolvedValue(poly(p));
        const msgs = [];
        await expect(fallbackMove(msgs, F('finnhub', 'cnbc', 'yahoo'), { env: ENV, now: OPEN })).rejects.toThrow(/all SPY move tiers failed/);
        expect(msgs.join(' ')).toMatch(/Polygon failed: newest bar 2026-10-08 is not the latest session 2026-10-09; skipped/);
    });

    test('Polygon dated the latest session is used (pre-open → yesterday is latest)', async () => {
        src.polygonDaily.mockResolvedValue(poly(bars(250, '2026-10-08', { base: 100, step: 1 })));
        const out = await fallbackMove([], F('finnhub', 'cnbc'), { env: ENV, now: PRE });
        expect(out).toMatchObject({ source: 'Polygon (fallback)', asOf: '2026-10-08', value: '+0.29%' });
    });

    test('Yahoo: stale quote skipped; fresh one uses the bars\' previous close', async () => {
        const h = bars(5, '2026-10-09', { base: 100, step: 1 }); // 100..104
        src.yahooChart.mockResolvedValue({ history: h, current: 104, prevClose: 50, meta: { regularMarketTime: OPEN / 1000, chartPreviousClose: 50 } });
        expect(await fallbackMove([], F('finnhub', 'cnbc', 'polygon'), { env: ENV, now: OPEN })).toMatchObject({ value: '+0.97%', source: 'Yahoo Finance (fallback)' });
        src.yahooChart.mockResolvedValue({ history: h.slice(0, 4), current: 103, prevClose: 50, meta: { regularMarketTime: Date.parse('2026-10-08T20:00:00Z') / 1000 } });
        await expect(fallbackMove([], F('finnhub', 'cnbc', 'polygon'), { env: ENV, now: OPEN })).rejects.toThrow(/not the latest session/);
    });

    test('a stale CNBC quote is not used', async () => {
        src.cnbcQuotes.mockResolvedValue({ SPY: { price: 778.57, change: 4.64, changePct: 0.5995, asOf: '2026-10-08' } });
        const msgs = [];
        await expect(fallbackMove(msgs, F('finnhub', 'polygon', 'yahoo'), { env: ENV, now: OPEN })).rejects.toThrow();
        expect(msgs.join(' ')).toMatch(/CNBC failed: CNBC quote dated 2026-10-08/);
    });
});

describe('routes: fault names reach the tiers; stale builds are not "good"', () => {
    const served = [];
    jest.doMock('../store', () => ({
        serve: async (key, produce, opts) => {
            try {
                const p = await produce();
                served.push({ key, good: !!opts.isGood(p), p });
                return Response.json(p);
            } catch (e) { served.push({ key, err: e.message }); return Response.json(opts.fallback); }
        },
    }));
    const req = (path) => new Request(`https://x.test${path}`, { headers: { 'user-agent': 'jest' } });
    const OLD_ENV = process.env;
    beforeEach(() => { served.length = 0; process.env = { ...OLD_ENV, POLYGON_KEY: 'p', FINNHUB_KEY: 'f' }; delete process.env.LAMBDA_URL; });
    afterAll(() => { process.env = OLD_ENV; });

    test('/api/spy with every live tier failed/stale → a stale-only build is served but not good', async () => {
        const { GET } = require('../../app/api/spy/route');
        src.nasdaqHistory.mockResolvedValue(bars(1000, '2020-01-02')); // years old → stale everywhere
        const res = await GET(req('/api/spy?_fail=lambda,polygon,finnhub,cnbc,yahoo'));
        const b = await res.json();
        expect(b._meta.stale).toBe(true);
        expect(served[0]).toMatchObject({ key: 'spy', good: false });
        expect(src.polygonDaily).not.toHaveBeenCalled();
    });

    test('/api/spy: a stale build newer than the last-good copy is preferred (spyPreferNewer)', () => {
        const { spyPreferNewer } = require('../spyTiers');
        const h = bars(300, '2025-08-01');
        const asOf = h[h.length - 1].date;
        const stale = buildSpy(h, 600, 599, `Polygon (fallback, last close ${asOf})`, { meta: { stale: true, hasErrors: true, asOf } });
        const dayBefore = new Date(Date.parse(`${asOf}T20:00:00Z`) - 3 * 864e5).toISOString();
        const sameDayIntraday = `${asOf}T15:00:00.000Z`;
        const dayAfter = new Date(Date.parse(`${asOf}T20:00:00Z`) + 864e5).toISOString();
        expect(spyPreferNewer(stale, dayBefore)).toBe(true);       // KV copy 3 days older → serve yesterday's close
        expect(spyPreferNewer(stale, sameDayIntraday)).toBe(true); // close beats that day's intraday copy
        expect(spyPreferNewer(stale, dayAfter)).toBe(false);       // a newer copy keeps winning
        expect(spyPreferNewer({ ...stale, _meta: { ...stale._meta, stale: false } }, dayBefore)).toBe(false); // only flagged builds
        expect(spyPreferNewer(stale, 'garbage')).toBe(false);
    });

    test('/api/spy-daily-move?_fail=finnhub → CNBC', async () => {
        const { GET } = require('../../app/api/spy-daily-move/route');
        src.cnbcQuotes.mockResolvedValue({ SPY: { price: 778.57, change: 4.64, changePct: 0.5995, asOf: latestSessionDate() } });
        const b = await (await GET(req('/api/spy-daily-move?_fail=lambda,finnhub'))).json();
        expect(b).toMatchObject({ value: '+0.60%', source: 'CNBC (fallback)' });
        expect(served[0].good).toBe(true);
        expect(src.finnhubQuote).not.toHaveBeenCalled();
    });
});
