import {
    rubberBandItem, historyItem, newestRowOf, latestCloseMs, isRed, buildFreshness,
    servedCopyItem, SERVED_ROUTES, fwdPeItem, polymarketItem, peItem,
} from '../servedFreshness';
import { parseRows } from '../../app/api/history/route';

// Producer shape: scripts/rubber_band.py build_snapshot() — "asOf": dates[-1] as
// strftime("%Y-%m-%d"), "generatedAt": isoformat(timespec="seconds").
const snap = (asOf, generatedAt) => ({
    asOf, generatedAt,
    spec: { version: '1.0' },
    verdict: { colour: 'green', text: 'The rubber band is working.' },
    dials: { slow: { colour: 'green' }, fast: { colour: 'green' }, age: { colour: 'green' }, rip: { colour: 'green' }, machines: { colour: 'green' } },
    counts: { dips_total: 30, rips_total: 20, bars: 6000, first_bar: '2002-01-02' },
    history: [{ d: asOf, px: 480.1, rsi: 51.2, slow: 0.6, fast: 0.6, rip: -0.1 }],
    _meta: { source: 'gist', hasErrors: false, stale: false, ageDays: 0, messages: [] },
});
const t = (iso) => Date.parse(iso);

describe('latestCloseMs', () => {
    test('Fri 2026-10-02 close is 20:00Z (EDT); weekend still points at it', () => {
        expect(latestCloseMs(t('2026-10-02T21:00:00Z'))).toBe(t('2026-10-02T20:00:00Z'));
        expect(latestCloseMs(t('2026-10-04T15:00:00Z'))).toBe(t('2026-10-02T20:00:00Z'));
    });
    test('before the close the newest is yesterday; holidays + early closes respected', () => {
        expect(latestCloseMs(t('2026-10-02T15:00:00Z'))).toBe(t('2026-10-01T20:00:00Z'));
        // Thanksgiving Thu 2026-11-26 closed; Fri 11-27 early close 13:00 EST = 18:00Z.
        expect(latestCloseMs(t('2026-11-27T12:00:00Z'))).toBe(t('2026-11-25T21:00:00Z'));
        expect(latestCloseMs(t('2026-11-27T19:00:00Z'))).toBe(t('2026-11-27T18:00:00Z'));
    });
});

describe('rubber-band item', () => {
    test('fresh: Friday snapshot (generated 18:30 ET) read on Sat night = green', () => {
        const now = t('2026-10-03T02:48:00Z');
        const it = rubberBandItem(snap('2026-10-02', '2026-10-02T22:30:19+00:00'), now);
        expect(it).toEqual({ name: 'rubber-band', inputAgeH: 6.8, servedAgeH: 6.8, graceH: 8 });
        expect(isRed(it)).toBe(false);
    });
    test('fresh: Friday snapshot still green on Monday morning (weekend is a legit gap)', () => {
        const it = rubberBandItem(snap('2026-10-02'), t('2026-10-05T13:00:00Z'));
        expect(isRed(it)).toBe(false);
    });
    test('fresh: Friday 18:00 ET, run not due yet → newest close is newer but within grace', () => {
        const it = rubberBandItem(snap('2026-10-01'), t('2026-10-02T22:00:00Z'));
        expect(it.inputAgeH).toBe(2);
        expect(isRed(it)).toBe(false);
    });
    test('stale: Monday night with only Friday\'s snapshot (missed run) = red', () => {
        const it = rubberBandItem(snap('2026-10-02'), t('2026-10-06T05:00:00Z'));
        expect(it.inputAgeH).toBe(9);
        expect(it.servedAgeH).toBe(81);
        expect(isRed(it)).toBe(true);
    });
    test('fallback payload (asOf null) after the grace = red', () => {
        const it = rubberBandItem({ asOf: null, dials: null, verdict: null }, t('2026-10-06T05:00:00Z'));
        expect(it.servedAgeH).toBeNull();
        expect(isRed(it)).toBe(true);
    });
});

// Real Sheet1 CSV shape (financial-dashboard-history scraper.py writes it; Date = runner UTC date).
const CSV = [
    'Date,Yield Curve (10Y-2Y),Profit Margin,Sahm Rule,Consumer Sentiment,Initial Claims (4wk)',
    '2026-10-01,0.41,15.7,0.03,51.7,202.25',
    '2026-10-01,0.41,15.7,0.03,51.7,202.25',
    '2026-10-02,0.46,15.7,0.03,51.7,200',
    '2026-10-02,0.46,15.7,0.03,51.7,200',
    '2026-10-03,0.45,15.7,0.03,51.7,200',
].join('\n');

describe('history-sheet item', () => {
    test('newestRowOf counts the newest UTC date\'s rows from real CSV rows', () => {
        expect(newestRowOf(parseRows(CSV))).toEqual({ date: '2026-10-03', rows: 1 });
        expect(newestRowOf([])).toBeNull();
    });
    test('fresh: the 02:00Z row read at 02:48Z = green', () => {
        const it = historyItem({ _meta: { newestRow: newestRowOf(parseRows(CSV)) } }, t('2026-10-03T02:48:00Z'));
        expect(it).toEqual({ name: 'history-sheet', inputAgeH: null, servedAgeH: 0.8, graceH: 12, maxAgeH: 30 });
        expect(isRed(it)).toBe(false);
    });
    test('fresh: one dropped run (next row a day later) still green', () => {
        const it = historyItem({ _meta: { newestRow: { date: '2026-10-02', rows: 2 } } }, t('2026-10-03T15:00:00Z'));
        expect(it.servedAgeH).toBe(25);
        expect(isRed(it)).toBe(false);
    });
    test('stale: newest row two days old = red', () => {
        const it = historyItem({ _meta: { newestRow: { date: '2026-10-01', rows: 2 } } }, t('2026-10-03T02:48:00Z'));
        expect(it.servedAgeH).toBe(36.8);
        expect(isRed(it)).toBe(true);
    });
    test('payload without newestRow → servedAgeH null', () => {
        expect(historyItem({ today: null, metrics: {} }).servedAgeH).toBeNull();
    });
});

test('buildFreshness emits only names + hour counts', () => {
    const out = buildFreshness({ rubberBand: snap('2026-10-02'), history: { _meta: { newestRow: { date: '2026-10-03', rows: 1 } } } }, t('2026-10-03T02:48:00Z'));
    expect(out.app).toBe('financial-telegram-bot');
    expect(out.v).toBe(1);
    expect(out.items.map((i) => i.name)).toEqual(['rubber-band', 'history-sheet', ...SERVED_ROUTES.map((r) => `served:${r}`), 'pe']);
    expect(JSON.stringify(out)).not.toMatch(/\d{4}-\d{2}-\d{2}/);
});

// Real /api/* shapes from the 2026-10-09 live fault matrix (lib/store.js serve() labels).
describe('served:* items (market routes)', () => {
    const NOW = t('2026-10-10T02:45:00Z');           // Fri close 2026-10-09 20:00Z → input 6.75 h
    const live = { current: 778.57, _meta: { source: 'Polygon + Finnhub Spot' } };
    const kv = (iso) => ({ current: 778.57, _meta: { source: `KV last-good (${iso}) ← Google Sheet`, stale: true, lastGoodAt: iso } });
    test('live payload → green, served = input', () => {
        const it = servedCopyItem('spy', live, NOW);
        expect(it).toEqual({ name: 'served:spy', inputAgeH: 6.8, servedAgeH: 6.8, graceH: 6 });
        expect(isRed(it)).toBe(false);
    });
    test('copy saved after the close still reflects it → green', () => {
        expect(isRed(servedCopyItem('spy', kv('2026-10-10T02:40:11.060Z'), NOW))).toBe(false);
    });
    test('copy saved BEFORE the newest close, input past grace → red', () => {
        const it = servedCopyItem('spy', kv('2026-10-08T21:00:00.000Z'), NOW);
        expect(it.servedAgeH).toBe(29.8);
        expect(isRed(it)).toBe(true);
    });
    test('time read from the source label when lastGoodAt is absent (fear-greed "Stale cache (iso) ← CNN")', () => {
        const fg = { score: 45, _meta: { source: 'Stale cache (2026-10-08T21:00:00.000Z) ← CNN', stale: true } };
        expect(isRed(servedCopyItem('fear-greed', fg, NOW))).toBe(true);
    });
    test('flagged-stale spy build (old close, spyTiers shape) → aged from that close → red', () => {
        const stale = { current: 770.1, _meta: { source: 'Yahoo Finance (fallback, last close 2026-10-08)', hasErrors: true, stale: true, asOf: '2026-10-08' } };
        const it = servedCopyItem('spy', stale, NOW);
        expect(it.servedAgeH).toBe(30.8);
        expect(isRed(it)).toBe(true);
    });
    test('Unavailable / error / no answer → servedAgeH null → red past grace', () => {
        for (const p of [{ _meta: { source: 'Unavailable' } }, { score: 'N/A', error: 'Fear & Greed unavailable', _meta: { source: 'Failed' } }, null]) {
            const it = servedCopyItem('x', p, NOW);
            expect(it.servedAgeH).toBeNull();
            expect(isRed(it)).toBe(true);
        }
    });
});

// A live payload is as old as its own newest market date (audit 25 Oct 2026): a feed that
// still answers but stopped moving must not grade green. Shapes: live /api/market-extra
// (metric lastDate) and /api/fear-greed (CNN asOf ISO; VIX proxies a plain date).
describe('served:* live payloads graded by their own dates', () => {
    const mx = (tnx, cl = tnx, t2y = tnx, fx = cl) => ({
        fx: { usdcad: { current: 1.3998, lastDate: fx }, usdbdt: { current: 124.1 } },  // ER-API spot: undated
        commodities: { cl: { current: 103.27, lastDate: cl }, gc: { current: 3874, lastDate: fx }, btc: { current: 80769, lastDate: fx } },
        rates: { tnx: { current: 5, lastDate: tnx }, t2y: { current: 4.69, lastDate: t2y }, mortgageRate: { current: 7.32, lastDate: '2026-09-01' } },
        _meta: { source: 'yfinance/Polygon/Finnhub/FRED/ER-API', hasErrors: false },
    });
    const red = (name, p, now) => isRed(servedCopyItem(name, p, t(now)));

    test('weekend (Sun 25 Oct): Friday yields + oil → green; the 10Y still on Thursday → red', () => {
        const now = '2026-10-25T21:53:00Z';
        expect(red('market-extra', mx('2026-10-23'), now)).toBe(false);
        expect(servedCopyItem('market-extra', mx('2026-10-23'), t(now)).servedAgeH).toBe(49.9);
        expect(red('market-extra', mx('2026-10-22', '2026-10-23'), now)).toBe(true);
        expect(red('market-extra', mx('2026-10-23', '2026-10-25'), now)).toBe(false);   // Sunday-evening oil bar
    });
    test('weekday night (Tue 03:00Z): Monday yields + oil dated today → green; FRED a session late → red', () => {
        const now = '2026-10-20T03:00:00Z';                                             // Mon close 20:00Z, input 7 h
        expect(red('market-extra', mx('2026-10-19', '2026-10-20'), now)).toBe(false);
        expect(red('market-extra', mx('2026-10-16', '2026-10-20'), now)).toBe(true);
        expect(red('market-extra', mx('2026-10-19', '2026-10-16'), now)).toBe(true);    // oil frozen
    });
    test('weekday daytime (before the close): yesterday\'s yields are the newest → green', () => {
        expect(red('market-extra', mx('2026-10-19', '2026-10-20'), '2026-10-20T17:00:00Z')).toBe(false);
    });
    test('bond-market holidays with NYSE open publish no yields: prior day stays green', () => {
        // Columbus Day Mon 12 Oct 2026, Veterans Day Wed 11 Nov 2026 (no Treasury CSV row).
        expect(red('market-extra', mx('2026-10-09', '2026-10-12'), '2026-10-13T03:00:00Z')).toBe(false);
        expect(red('market-extra', mx('2026-11-10', '2026-11-11'), '2026-11-12T04:00:00Z')).toBe(false);
        expect(red('market-extra', mx('2026-11-09', '2026-11-11'), '2026-11-12T04:00:00Z')).toBe(true);
    });
    test('Veterans Day on a Sunday is observed Monday: no yields that day (no 11/12/2018 row), Friday stays green', () => {
        // Sun 11 Nov 2029 → bonds shut Mon 12 Nov; 23:30 ET that night
        expect(red('market-extra', mx('2029-11-09', '2029-11-12'), '2029-11-13T04:30:00Z')).toBe(false);
        expect(red('market-extra', mx('2029-11-09', '2029-11-13'), '2029-11-14T04:30:00Z')).toBe(true);
    });
    test('FX / gold / BTC are graded too (audit A3): live 1 Nov shape green; FX frozen on Thursday red', () => {
        const now = '2026-11-01T21:22:00Z';                                             // Sunday
        const live = mx('2026-10-30');
        live.commodities.btc.lastDate = '2026-10-31';                                    // BTC trades the weekend
        expect(red('market-extra', live, now)).toBe(false);
        expect(red('market-extra', mx('2026-10-30', '2026-10-30', '2026-10-30', '2026-10-29'), now)).toBe(true);
    });
    test('no dated metric → falls back to the input age (as before)', () => {
        const it = servedCopyItem('market-extra', { fx: { usdcad: { current: 1.4 } }, _meta: { source: 'Lambda' } }, t('2026-10-20T03:00:00Z'));
        expect(it.servedAgeH).toBe(it.inputAgeH);
    });
    test('fear-greed: CNN asOf after the newest close → green; frozen 2 days → red; proxy date', () => {
        const fg = (asOf, meta = { source: 'CNN', hasErrors: false }) => ({ score: 20.98, rating: 'extreme fear', asOf, _meta: meta });
        const now = '2026-10-25T21:53:00Z';
        expect(red('fear-greed', fg('2026-10-23T23:59:44+00:00'), now)).toBe(false);
        expect(red('fear-greed', fg('2026-10-21T23:59:44+00:00'), now)).toBe(true);
        expect(red('fear-greed', fg('2026-10-20T15:00:00+00:00'), '2026-10-20T17:00:00Z')).toBe(false); // intraday print
        // A VIX proxy is NOT the CNN index (route: proxy + hasErrors): never green, however fresh.
        expect(red('fear-greed', fg('2026-10-23', { source: 'CBOE VIX Proxy', proxy: true, hasErrors: true }), now)).toBe(true);
        expect(red('fear-greed', fg(undefined), now)).toBe(false);                     // no date: input age
    });
    test('fred / sheets: a live answer flagged hasErrors (partial FRED load, stale AAII) is not green', () => {
        const now = '2026-11-01T21:22:00Z';
        const fred = (hasErrors, loadedCount) => ({ yieldCurve: { value: 0.38, asOf: '2026-10-30' }, _meta: { source: 'St. Louis Fed', hasErrors, loadedCount, messages: [`Loaded ${loadedCount}/17 series`] } });
        expect(red('fred', fred(false, 17), now)).toBe(false);
        expect(red('fred', fred(true, 15), now)).toBe(true);
        const sheets = (aaiiStale) => ({ NotSoBoring: 'ON', AAII: { bull: 37.8, as_of: '2026-10-28', stale: aaiiStale, lastGood: false },
            _meta: { source: 'Computed (daily prices) + Google Sheets (Live)', hasErrors: aaiiStale, stale: false, staleFields: [] } });
        expect(red('sheets', sheets(false), now)).toBe(false);
        expect(red('sheets', sheets(true), now)).toBe(true);
        // within the grace after a close nothing is red yet (same rule as every row)
        expect(red('fred', fred(true, 15), '2026-10-30T21:00:00Z')).toBe(false);
    });
    test('vol + breadth: graded by their oldest row date (live 1 Nov shapes green; a lagging row red)', () => {
        const now = '2026-11-01T21:22:00Z';
        const vol = (uvxy) => ({ updated_at: '2026-10-30', tickers: [{ ticker: 'SPY', iv: 19.42, asOf: '2026-10-30' }, { ticker: 'UVXY', iv: 90, asOf: uvxy }],
            _meta: { source: 'VIX:cboe · VVIX:cboe', hasErrors: false, fallback: false } });
        expect(SERVED_ROUTES).toEqual(expect.arrayContaining(['vol', 'breadth']));
        expect(red('vol', vol('2026-10-30'), now)).toBe(false);
        expect(red('vol', vol('2026-10-28'), now)).toBe(true);
        const breadth = (hyg) => ({ updated_at: '2026-10-30', pairs: { rspSpy: { ratio: 0.29, asOf: '2026-10-30' }, hygLqd: { ratio: 0.74, asOf: hyg } },
            _meta: { source: 'RSP:polygon · SPY:polygon', hasErrors: false, messages: [] } });
        expect(red('breadth', breadth('2026-10-30'), now)).toBe(false);
        expect(red('breadth', breadth('2026-10-29'), now)).toBe(true);
    });
    test('spy-daily-move carries no date: graded as before', () => {
        const it = servedCopyItem('spy-daily-move', { value: '-0.52%', source: 'Google Sheets' }, t('2026-10-25T21:53:00Z'));
        expect(it.servedAgeH).toBe(it.inputAgeH);
    });
});

// Polymarket trades 24/7: graded by the board's own fetch `timestamp` (a last-good copy keeps
// the original one), against a 6 h cap. Nothing usable served = item missing (fleet: red).
describe('polymarket item', () => {
    const board = (timestamp, meta = { hasErrors: false, messages: [] }) => ({ trending: [{ slug: 'a' }], breaking: [], macro: [], source: 'Polymarket API', timestamp, _meta: meta });
    test('live 1 Nov board = green; a 7 h old copy = red; Unavailable / no answer = missing', () => {
        const now = t('2026-11-01T21:25:00Z');
        const it = polymarketItem(board('2026-11-01T21:22:29.697932Z'), now);
        expect(it).toEqual({ name: 'served:polymarket', inputAgeH: null, servedAgeH: 0, graceH: 6, maxAgeH: 6 });
        expect(isRed(it)).toBe(false);
        const old = board('2026-11-01T14:00:00Z', { source: 'KV last-good (2026-11-01T14:00:05Z) ← Polymarket API', stale: true, lastGoodAt: '2026-11-01T14:00:05Z' });
        expect(isRed(polymarketItem(old, now))).toBe(true);
        expect(polymarketItem({ trending: [], timestamp: '2026-11-01T21:25:00Z', _meta: { source: 'Unavailable', hasErrors: true } }, now)).toBeNull();
        expect(polymarketItem(null, now)).toBeNull();
    });
});

// Writer shape: app/api/fwd-pe/route.js body (rows/latest/seen from lib/fwdPe.js resolveFwdPe).
describe('fwd-pe item (FactSet forward P/E)', () => {
    const body = (latest, seen) => ({ rows: [], latest: { date: latest, pe: 18.7, a5: 19.8, a10: 19.1 }, seen, checkedAt: '2026-10-30T19:00:00.000Z', _meta: { bakedTo: '2026-10-23', messages: [] } });
    test('the newest report served = fresh, through a skipped holiday week', () => {
        const it = fwdPeItem(body('2026-10-30', '2026-10-30'), t('2026-11-12T15:00:00Z'));
        expect(it).toMatchObject({ name: 'fwd-pe', graceH: 6, maxAgeH: 720 });
        expect(it.servedAgeH).toBe(it.inputAgeH);
        expect(isRed(it)).toBe(false);
    });
    test('a newer report it could not read = red after 6 h', () => {
        expect(isRed(fwdPeItem(body('2026-10-30', '2026-11-06'), t('2026-11-06T22:00:00Z')))).toBe(false);
        expect(isRed(fwdPeItem(body('2026-10-30', '2026-11-06'), t('2026-11-07T01:00:00Z')))).toBe(true);
    });
    test('nothing new for 30 days = red; no answer = item missing (fleet expect_items: red)', () => {
        expect(isRed(fwdPeItem(body('2026-10-30', '2026-10-30'), t('2026-12-01T00:00:00Z')))).toBe(true);
        const out = buildFreshness({ rubberBand: snap('2026-10-02'), history: null, fwdPe: null }, t('2026-10-03T02:48:00Z'));
        expect(out.items.map((i) => i.name)).not.toContain('fwd-pe');
        expect(buildFreshness({ fwdPe: body('2026-10-30', '2026-10-30') }, t('2026-11-01T00:00:00Z')).items.map((i) => i.name)).toContain('fwd-pe');
    });
});

// Writer shape: app/api/fred/route.js buildResponse (peRatio / peRatioAsOf / peIsCape from lib/peRatio.js).
describe('pe item (Market Valuation tile: trailing P/E)', () => {
    const fred = (peRatioAsOf, extra = {}) => ({ peRatio: 25.86, peRatioAsOf, peSource: 'multpl', peIsCape: false, peStale: false, _meta: { source: 'St. Louis Fed' }, ...extra });
    test('live 1 Nov: Friday\'s multpl page on Sunday = green; Yahoo\'s fetch stamp and an intraday page never read as the future', () => {
        const it = peItem(fred('2026-10-30'), t('2026-11-01T23:31:00Z'));
        expect(it).toMatchObject({ name: 'pe', graceH: 6 });
        expect(it.servedAgeH).toBe(it.inputAgeH);
        expect(isRed(it)).toBe(false);
        expect(peItem(fred('2026-11-02T15:00:00.000Z', { peSource: 'yahoo' }), t('2026-11-02T15:10:00Z')).servedAgeH).toBe(0.2);
        expect(peItem(fred('2026-11-02'), t('2026-11-02T16:00:00Z')).servedAgeH).toBe(0);
    });
    test('the prior session\'s P/E = green up to 6 h after the close, red after', () => {
        expect(isRed(peItem(fred('2026-10-30'), t('2026-11-03T02:00:00Z')))).toBe(false); // Mon 9 pm ET
        expect(isRed(peItem(fred('2026-10-30'), t('2026-11-03T04:00:00Z')))).toBe(true); // Mon 11 pm ET
    });
    test('a frozen page, CAPE standing in or no P/E = red; the item is always there', () => {
        const now = t('2026-11-01T23:31:00Z');
        expect(isRed(peItem(fred('2026-10-22', { peStale: true }), now))).toBe(true);
        expect(isRed(peItem(fred('2026-10-30', { peIsCape: true }), now))).toBe(true);
        expect(isRed(peItem(fred(null, { peRatio: null, peSource: null }), now))).toBe(true);
        expect(isRed(peItem(null, now))).toBe(true);
        expect(buildFreshness({}, now).items.map((i) => i.name)).toContain('pe');
    });
});
