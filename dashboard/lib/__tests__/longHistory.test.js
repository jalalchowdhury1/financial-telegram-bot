import fs from 'fs';
import path from 'path';
import { decodeLong, joinLong, cutFor, spanDays, loadLong, peekLong, resetLong, longInfo } from '../longHistory';
import { SHEET_METRICS } from '../marks';
import INDEX from '../data/longHistoryIndex.json';

const sheet = [
    { date: '2026-03-12', value: 10 },
    { date: '2026-03-13', value: 11 },
    { date: '2026-09-05', value: 12 },
    { date: '2026-10-09', value: 13 },
];
const long = [
    { date: '1990-01-05', value: 1 },
    { date: '2026-03-11', value: 2 },
    { date: '2026-03-12', value: 99 }, // overlaps the sheet: must never be drawn
    { date: '2026-10-08', value: 99 },
];

afterEach(() => { resetLong(); delete global.fetch; });

describe('join', () => {
    test('baked points only BEFORE the sheet\'s first point; every sheet point kept as is', () => {
        const j = joinLong(sheet, long, { from: '1990-01-05' });
        expect(j.map((p) => p.value)).toEqual([1, 2, 10, 11, 12, 13]);
        expect(j.filter((p) => p.long).map((p) => p.date)).toEqual(['1990-01-05', '2026-03-11']);
    });

    test('sheetFrom: the sheet is a different number before it, so baked points run up to it', () => {
        const info = { from: '1990-01-05', sheetFrom: '2026-09-05' };
        expect(cutFor(sheet, info)).toBe('2026-09-05');
        const j = joinLong(sheet, [...long.slice(0, 2), { date: '2026-08-01', value: 5 }], info);
        expect(j.map((p) => p.value)).toEqual([1, 2, 5, 12, 13]);
    });

    test('no baked points (none, failed, or all after the cut) → the sheet, untouched', () => {
        expect(joinLong(sheet, null, null)).toBe(sheet);
        expect(joinLong(sheet, [], null)).toBe(sheet);
        expect(joinLong(sheet, [{ date: '2026-05-01', value: 1 }], {})).toBe(sheet);
    });

    test('span counts from the older of the baked start and the sheet start', () => {
        expect(spanDays(sheet, null)).toBe(211);
        expect(spanDays(sheet, { from: '2025-10-09' })).toBe(365);
        expect(spanDays([], { from: '1990-01-01' })).toBe(0);
    });

    test('decode: day offsets → dates; malformed bodies → null; non-numbers skipped', () => {
        expect(decodeLong({ from: '2026-01-30', t: [0, 2, 3], v: [1, 'x', 3] }))
            .toEqual([{ date: '2026-01-30', value: 1 }, { date: '2026-02-02', value: 3 }]);
        expect(decodeLong({ from: 'bad', t: [0], v: [1] })).toBeNull();
        expect(decodeLong({ from: '2026-01-01', t: [0, 1], v: [1] })).toBeNull();
        expect(decodeLong(null)).toBeNull();
    });
});

describe('load', () => {
    const key = Object.keys(INDEX.keys)[0];
    const body = { from: '2000-01-03', t: [0, 7], v: [1, 2] };

    test('fetched once per page from /history/<key>.json, then served synchronously', async () => {
        global.fetch = jest.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(body) }));
        expect(peekLong(key)).toBeUndefined();
        const [a, b] = await Promise.all([loadLong(key), loadLong(key)]);
        expect(a).toEqual([{ date: '2000-01-03', value: 1 }, { date: '2000-01-10', value: 2 }]);
        expect(b).toBe(a);
        expect(global.fetch).toHaveBeenCalledTimes(1);
        expect(global.fetch).toHaveBeenCalledWith(`/history/${key}.json`);
        expect(peekLong(key)).toBe(a);
    });

    test('never rejects; a failure is forgotten so the next open tries again', async () => {
        global.fetch = jest.fn(() => Promise.resolve({ ok: false, status: 404 }));
        await expect(loadLong(key)).resolves.toBeNull();
        global.fetch = jest.fn(() => { throw new Error('offline'); });
        await expect(loadLong(key)).resolves.toBeNull();
        delete global.fetch; // jsdom without fetch at all
        await expect(loadLong(key)).resolves.toBeNull();
        global.fetch = jest.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(body) }));
        await expect(loadLong(key)).resolves.toHaveLength(2);
    });

    test('a stat without a baked file never fetches', async () => {
        global.fetch = jest.fn();
        expect(longInfo('usdbdt')).toBeNull();
        await expect(loadLong('usdbdt')).resolves.toBeNull();
        expect(global.fetch).not.toHaveBeenCalled();
    });
});

describe('the baked files (scripts/bake_long_history.py)', () => {
    const dir = path.join(__dirname, '..', '..', 'public', 'history');
    const keys = Object.keys(INDEX.keys);

    test('every stat with a sheet chart and a long public source is baked; BDT pairs are not', () => {
        expect(keys.length).toBeGreaterThanOrEqual(34);
        for (const k of ['vixCurrent', 'vix3m', 'cnnFearGreed', 'profitMargin', 'claims', 'peRatio', 'aaiiDiff', 'gold', 'dxy']) {
            expect(keys).toContain(k);
        }
        for (const k of ['usdbdt', 'inrbdt', 'cadbdt']) expect(keys).not.toContain(k);
    });

    test.each(Object.keys(INDEX.keys))('%s: a sheet stat, decodes, matches its index row, ends inside the sheet era', (key) => {
        const info = INDEX.keys[key];
        expect(SHEET_METRICS[key]).toBeDefined(); // its chart exists (lib/marks.js), so the popover can open
        const file = path.join(dir, `${key}.json`);
        expect(fs.statSync(file).size).toBeLessThan(120 * 1024); // one popover fetch stays light
        const body = JSON.parse(fs.readFileSync(file, 'utf8'));
        const pts = decodeLong(body);
        expect(pts).toHaveLength(info.n);
        expect(pts[0].date).toBe(info.from);
        expect(pts[pts.length - 1].date).toBe(info.to);
        for (let i = 1; i < pts.length; i++) expect(pts[i].date > pts[i - 1].date).toBe(true);
        // no gap before the sheet: the bake reaches past the sheet's first row (2026-03-12)
        expect(info.to >= '2026-03-12').toBe(true);
        if (info.sheetFrom) expect(info.to >= info.sheetFrom).toBe(true);
        expect(typeof info.source).toBe('string');
    });
});
