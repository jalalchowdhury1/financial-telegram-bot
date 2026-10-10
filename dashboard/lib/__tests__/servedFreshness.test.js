import {
    rubberBandItem, historyItem, newestRowOf, latestCloseMs, isRed, buildFreshness,
    servedCopyItem, SERVED_ROUTES,
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
        expect(it).toEqual({ name: 'rubber-band', inputAgeH: 6.8, servedAgeH: 6.8, graceH: 6 });
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
        const it = rubberBandItem(snap('2026-10-02'), t('2026-10-06T03:00:00Z'));
        expect(it.inputAgeH).toBe(7);
        expect(it.servedAgeH).toBe(79);
        expect(isRed(it)).toBe(true);
    });
    test('fallback payload (asOf null) after the grace = red', () => {
        const it = rubberBandItem({ asOf: null, dials: null, verdict: null }, t('2026-10-06T03:00:00Z'));
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
    expect(out.items.map((i) => i.name)).toEqual(['rubber-band', 'history-sheet', ...SERVED_ROUTES.map((r) => `served:${r}`)]);
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
    test('Unavailable / error / no answer → servedAgeH null → red past grace', () => {
        for (const p of [{ _meta: { source: 'Unavailable' } }, { score: 'N/A', error: 'Fear & Greed unavailable', _meta: { source: 'Failed' } }, null]) {
            const it = servedCopyItem('x', p, NOW);
            expect(it.servedAgeH).toBeNull();
            expect(isRed(it)).toBe(true);
        }
    });
});
