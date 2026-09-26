/**
 * 🧬 20Y / 30Y / 40Y factor windows from the Ken French Data Library (lib/factorsLong.js).
 */
import zlib from 'zlib';
import {
    unzipFirst, parseFrenchCsv, pickSeries, buildLong, validLong, computeLongWindow, attachLong,
    longFromZips, LONG_KEYS, LONG_SOURCES, LONG_WINDOWS,
} from '../factorsLong';
import bakedLong from '../data/factorsLong.json';

/** A minimal one-file .zip (local header + data + central directory + EOCD). */
function makeZip(name, text, { method = 8 } = {}) {
    const raw = Buffer.from(text, 'latin1');
    const data = method === 8 ? zlib.deflateRawSync(raw) : raw;
    const nm = Buffer.from(name);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(method, 8);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(raw.length, 22);
    local.writeUInt16LE(nm.length, 26);
    const cd = Buffer.alloc(46);
    cd.writeUInt32LE(0x02014b50, 0);
    cd.writeUInt16LE(method, 10);
    cd.writeUInt32LE(data.length, 20);
    cd.writeUInt32LE(raw.length, 24);
    cd.writeUInt16LE(nm.length, 28);
    cd.writeUInt32LE(0, 42);
    const cdOffset = local.length + nm.length + data.length;
    const eocd = Buffer.alloc(22);
    eocd.writeUInt32LE(0x06054b50, 0);
    eocd.writeUInt16LE(1, 8);
    eocd.writeUInt16LE(1, 10);
    eocd.writeUInt32LE(cd.length + nm.length, 12);
    eocd.writeUInt32LE(cdOffset, 16);
    return Buffer.concat([local, nm, data, cd, nm, eocd]);
}

const ym = (i) => { // month i after 1963-07
    const y = 1963 + Math.floor((6 + i) / 12);
    const m = ((6 + i) % 12) + 1;
    return `${y}${String(m).padStart(2, '0')}`;
};

/** A Ken-French-shaped CSV: prose, VW monthly table, EW monthly table, annual table. */
function frenchCsv(columns, n, valueAt) {
    const row = (i, f) => `${ym(i)},${columns.map((_, c) => `   ${f(i, c).toFixed(4)}`).join(',')}`;
    return [
        'This file was created using the 202608 CRSP database.',
        '',
        '  Average Value Weighted Returns -- Monthly',
        `,${columns.join(',')}`,
        ...Array.from({ length: n }, (_, i) => row(i, valueAt)),
        '',
        '  Average Equal Weighted Returns -- Monthly',
        `,${columns.join(',')}`,
        ...Array.from({ length: n }, (_, i) => row(i, () => 99)),
        '',
        '  Average Value Weighted Returns -- Annual',
        `,${columns.join(',')}`,
        `1964,${columns.map(() => '5.0').join(',')}`,
    ].join('\r\n');
}

const N = 12 * 40 + 1 + 24; // enough for 40Y plus a little

function syntheticZips({ n = N, market = 1, factor = 1.5 } = {}) {
    const zips = {};
    for (const k of LONG_KEYS) {
        const cols = k === 'market' ? ['Mkt-RF', 'SMB', 'HML', 'RF'] : ['A', ...LONG_SOURCES[k].columns, 'Z'];
        const at = k === 'market'
            ? (i, c) => (c === 0 ? market - 0.25 : c === 3 ? 0.25 : 7)   // Mkt-RF + RF = market
            : (i, c) => (c === 1 ? factor : -50);
        zips[k] = makeZip(`${LONG_SOURCES[k].file}.csv`, frenchCsv(cols, n, at));
    }
    return zips;
}

describe('unzipFirst', () => {
    it('reads deflated and stored entries', () => {
        expect(unzipFirst(makeZip('a.csv', 'hello,world'))).toBe('hello,world');
        expect(unzipFirst(makeZip('a.csv', 'stored', { method: 0 }))).toBe('stored');
    });
    it('throws on something that is not a zip', () => {
        expect(() => unzipFirst(Buffer.from('<html>rate limited</html>'))).toThrow(/not a zip/);
    });
});

describe('parseFrenchCsv', () => {
    it('returns the FIRST monthly table (value-weighted), skipping prose', () => {
        const t = parseFrenchCsv(frenchCsv(['X', 'Y'], 3, (i, c) => i + c));
        expect(t.columns).toEqual(['X', 'Y']);
        expect(t.rows.map((r) => r.month)).toEqual(['1963-07', '1963-08', '1963-09']);
        expect(t.rows[2].values).toEqual([2, 3]);
    });
    it('refuses a file whose first monthly table is equal-weighted', () => {
        const csv = ['  Average Equal Weighted Returns -- Monthly', ',X', '196307, 1.0'].join('\n');
        expect(() => parseFrenchCsv(csv)).toThrow(/equal-weighted/);
    });
    it('null when there is no monthly table at all', () => {
        expect(parseFrenchCsv('nothing here')).toBeNull();
    });
});

describe('pickSeries', () => {
    const t = { columns: ['Mkt-RF', 'RF', 'Lo 30'], rows: [{ month: '1963-07', values: [1, 0.25, -99.99] }] };
    it('sums the listed columns (market = Mkt-RF + RF)', () => {
        expect(pickSeries(t, ['Mkt-RF', 'RF'])).toEqual([['1963-07', 1.25]]);
    });
    it('throws on a missing column or a missing value (-99.99)', () => {
        expect(() => pickSeries(t, ['BIG HiBM'])).toThrow(/missing column/);
        expect(() => pickSeries(t, ['Lo 30'])).toThrow(/missing Lo 30 at 1963-07/);
    });
});

describe('buildLong + validLong', () => {
    it('aligns every proxy on the common months', () => {
        const long = longFromZips(syntheticZips());
        expect(long.start).toBe('1963-07');
        expect(long.months).toHaveLength(N);
        expect(long.through).toBe(long.months[N - 1]);
        expect(long.series.market[0]).toBeCloseTo(1, 10);
        expect(long.series.value[0]).toBe(1.5);
        expect(validLong(long)).toBe(true);
    });
    it('refuses less than 40 years of history', () => {
        expect(() => longFromZips(syntheticZips({ n: 12 * 40 }))).toThrow(/too short/);
    });
    it('validLong rejects gaps, length mismatches and impossible returns', () => {
        const long = longFromZips(syntheticZips());
        expect(validLong({ ...long, months: [...long.months.slice(0, 5), ...long.months.slice(6)] })).toBe(false);
        expect(validLong({ ...long, series: { ...long.series, value: long.series.value.slice(1) } })).toBe(false);
        expect(validLong({ ...long, series: { ...long.series, size: long.series.size.map((v, i) => (i === 3 ? -100 : v)) } })).toBe(false);
        expect(validLong({ ...long, through: '1999-01' })).toBe(false);
        expect(validLong(null)).toBe(false);
    });
});

describe('computeLongWindow', () => {
    const long = longFromZips(syntheticZips({ market: 1, factor: 1.5 }));
    it('compounds exactly 12×years monthly returns from the end of the base month', () => {
        const w = computeLongWindow(long, 'value', 20);
        const F = 1.015 ** 240;
        const B = 1.01 ** 240;
        expect(w.f).toBeCloseTo((F - 1) * 100, 1);
        expect(w.b).toBeCloseTo((B - 1) * 100, 1);
        expect(w.rel).toBeCloseTo((F / B - 1) * 100, 1);
        expect(w.basis).toBe('research');
        expect(w.spark).toHaveLength(60);
        expect(w.spark[0]).toBe(0);
        expect(w.spark[59]).toBe(w.rel);
        expect(w.to.slice(0, 7)).toBe(long.through);
        expect(w.from.slice(0, 7)).toBe(long.months[long.months.length - 1 - 240]);
    });
    it('null when history is shorter than the window', () => {
        expect(computeLongWindow(long, 'value', 50)).toBeNull();
    });
    it('matches an independent Python calculation on the real bake (value, 20Y)', () => {
        // Python, straight from the Dartmouth CSVs (2026-09-26): 240 months, f=641.05 b=772.74 rel=-15.09
        if (bakedLong.through !== '2026-08') return; // re-baked since — the numbers moved on
        const w = computeLongWindow(bakedLong, 'value', 20);
        expect([w.f, w.b, w.rel]).toEqual([641.05, 772.74, -15.09]);
    });
});

describe('attachLong', () => {
    const long = longFromZips(syntheticZips());
    const payload = {
        asOf: '2026-09-25',
        windows: ['1M', '10Y'],
        factors: [{ key: 'value', label: 'Value', windows: { '1M': { rel: 1 }, '10Y': null } }],
        _meta: { messages: [] },
    };
    it('adds 20Y/30Y/40Y to every factor and says where they came from', () => {
        const p = attachLong(payload, long, 'live');
        expect(p.windows).toEqual(['1M', '10Y', ...LONG_WINDOWS]);
        expect(Object.keys(p.factors[0].windows)).toEqual(['1M', '10Y', '20Y', '30Y', '40Y']);
        expect(p.factors[0].windows['1M']).toEqual({ rel: 1 });
        expect(p.factors[0].longProxy).toBe('large-cap value');
        expect(p.long).toMatchObject({ through: long.through, source: 'live', basis: 'total return, monthly' });
        expect(payload.windows).toEqual(['1M', '10Y']); // input untouched
    });
    it('is idempotent (a cached payload that already has long windows)', () => {
        const once = attachLong(payload, long, 'live');
        expect(attachLong(once, long, 'live').windows).toEqual(once.windows);
    });
    it('returns the payload unchanged when the long history is unusable', () => {
        expect(attachLong(payload, null, 'live')).toBe(payload);
        expect(attachLong(payload, { months: [] }, 'live')).toBe(payload);
        expect(attachLong(null, long, 'live')).toBeNull();
    });
});

describe('the committed bake', () => {
    it('is valid and covers 40 years for every factor', () => {
        expect(validLong(bakedLong)).toBe(true);
        expect(bakedLong.bakedAt).toMatch(/^\d{4}-\d{2}-\d{2}$/);
        for (const k of LONG_KEYS) expect(bakedLong.series[k].length).toBeGreaterThanOrEqual(12 * 40 + 1);
    });
});
