/**
 * 🧬 Long factor windows — 20Y / 30Y / 40Y — from the Ken French Data Library.
 *
 * Why a different source: the factor ETFs are young (VLUE/MTUM/QUAL 2013, USMV
 * 2011), so no ETF can show 20+ years. For those windows the row switches to the
 * academic research portfolios the factor ETFs are modelled on: Fama/French
 * monthly returns built from CRSP, available from July 1963 for all five factors.
 *
 * It is a DIFFERENT BASIS from the short windows, and the UI says so every time:
 *   1M … 10Y   ETF vs SPY, daily, price only (dividends excluded)
 *   20Y … 40Y  research portfolio vs the whole US market, TOTAL return
 *              (dividends reinvested), monthly, ~1–2 months behind
 *
 * Proxies (value-weighted; "BIG" = above the NYSE median market cap):
 *   market    F-F_Research_Data_Factors      Mkt-RF + RF
 *   value     6_Portfolios_2x3               BIG HiBM     (cheapest 30% by book/price)
 *   momentum  6_Portfolios_ME_Prior_12_2     BIG HiPRIOR  (top 30% by 12-2 month return)
 *   quality   6_Portfolios_ME_OP_2x3         BIG HiOP     (top 30% by operating profitability)
 *   size      Portfolios_Formed_on_ME        Lo 30        (smallest 30% by market cap)
 *   lowvol    Portfolios_Formed_on_VAR       Lo 20        (lowest 20% by past variance)
 *
 * Tiers (wired in app/api/factors/route.js): live Dartmouth zips (Next data cache,
 * 7 days) → baked lib/data/factorsLong.json (scripts/bake-factors-long.mjs) → no
 * long windows (the 20Y/30Y/40Y buttons disable; nothing else on the row changes).
 *
 * Pure except unzipFirst (node zlib). Self-contained on purpose: the bake script
 * imports this file directly with plain Node.
 */
import zlib from 'zlib';

export const LONG_WINDOWS = ['20Y', '30Y', '40Y'];
export const LONG_YEARS = { '20Y': 20, '30Y': 30, '40Y': 40 };
export const KF_BASE = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp';
export const KF_PROVIDER = 'Ken French Data Library (Fama/French, CRSP)';

export const LONG_SOURCES = {
    market: { file: 'F-F_Research_Data_Factors', columns: ['Mkt-RF', 'RF'], label: 'the whole US stock market' },
    value: { file: '6_Portfolios_2x3', columns: ['BIG HiBM'], label: 'large-cap value' },
    momentum: { file: '6_Portfolios_ME_Prior_12_2', columns: ['BIG HiPRIOR'], label: 'large-cap momentum' },
    quality: { file: '6_Portfolios_ME_OP_2x3', columns: ['BIG HiOP'], label: 'large-cap high profitability' },
    size: { file: 'Portfolios_Formed_on_ME', columns: ['Lo 30'], label: 'the smallest 30% of stocks' },
    lowvol: { file: 'Portfolios_Formed_on_VAR', columns: ['Lo 20'], label: 'the calmest 20% of stocks' },
};
export const LONG_KEYS = Object.keys(LONG_SOURCES);

/** The longest window needs its base month plus this many monthly returns. */
const MIN_MONTHS = 12 * 40 + 1;
const SPARK_SLOTS = 60;
const round2 = (x) => Math.round(x * 100) / 100;

/** Text of the first file in a .zip (the Dartmouth zips hold exactly one CSV). Throws on anything odd. */
export function unzipFirst(buf) {
    const b = Buffer.from(buf);
    let eocd = -1; // end-of-central-directory record, within the last 22 + 64 KB
    for (let i = b.length - 22; i >= Math.max(0, b.length - 22 - 0xffff); i--) {
        if (b.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error('not a zip file');
    const cd = b.readUInt32LE(eocd + 16);
    if (cd + 46 > b.length || b.readUInt32LE(cd) !== 0x02014b50) throw new Error('bad zip directory');
    const method = b.readUInt16LE(cd + 10);
    const size = b.readUInt32LE(cd + 20);
    const local = b.readUInt32LE(cd + 42);
    if (local + 30 > b.length || b.readUInt32LE(local) !== 0x04034b50) throw new Error('bad zip entry');
    const start = local + 30 + b.readUInt16LE(local + 26) + b.readUInt16LE(local + 28);
    const data = b.subarray(start, start + size);
    if (method === 0) return data.toString('latin1');
    if (method === 8) return zlib.inflateRawSync(data).toString('latin1');
    throw new Error(`unsupported zip method ${method}`);
}

/**
 * The FIRST monthly table of a Ken French CSV: `{ columns, rows: [{month:'YYYY-MM', values}] }`.
 * In every file used here that is the value-weighted monthly table (annual tables have
 * 4-digit years, so the 6-digit row pattern skips them; an equal-weighted label is refused).
 */
export function parseFrenchCsv(text) {
    const lines = String(text || '').split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
        if (!lines[i].startsWith(',')) continue;
        const label = (lines[i - 1] || '').trim();
        const columns = lines[i].split(',').slice(1).map((s) => s.trim());
        const rows = [];
        for (let j = i + 1; j < lines.length; j++) {
            const m = /^\s*(\d{4})(\d{2})\s*,(.*)$/.exec(lines[j]);
            if (!m) break;
            rows.push({ month: `${m[1]}-${m[2]}`, values: m[3].split(',').map((s) => Number(s.trim())) });
        }
        if (!rows.length) continue;
        if (/equal/i.test(label)) throw new Error(`first monthly table is equal-weighted (${label})`);
        return { columns, rows };
    }
    return null;
}

/** Monthly % returns for one proxy (columns summed, e.g. Mkt-RF + RF). Missing (≤ -99.99) → throws. */
export function pickSeries(table, columns) {
    if (!table) throw new Error('no monthly table');
    const idx = columns.map((c) => table.columns.indexOf(c));
    if (idx.some((k) => k < 0)) throw new Error(`missing column ${columns.join('+')}`);
    return table.rows.map(({ month, values }) => {
        const parts = idx.map((k) => values[k]);
        if (parts.some((v) => !Number.isFinite(v) || v <= -99.99)) throw new Error(`missing ${columns.join('+')} at ${month}`);
        return [month, parts.reduce((a, v) => a + v, 0)];
    });
}

const nextMonth = (ym) => {
    const [y, m] = ym.split('-').map(Number);
    return m === 12 ? `${y + 1}-01` : `${y}-${String(m + 1).padStart(2, '0')}`;
};

/**
 * Align the six proxies on their common months → the stored/served shape:
 * `{ start, through, months: [...], series: { market: [...], value: [...], ... } }`.
 * @param {Record<string, [string, number][]>} byKey
 */
export function buildLong(byKey) {
    const maps = {};
    for (const k of LONG_KEYS) {
        if (!Array.isArray(byKey[k]) || !byKey[k].length) throw new Error(`no data for ${k}`);
        maps[k] = new Map(byKey[k]);
    }
    const first = LONG_KEYS.map((k) => byKey[k][0][0]).sort().pop();                 // latest start
    const last = LONG_KEYS.map((k) => byKey[k][byKey[k].length - 1][0]).sort()[0];   // earliest end
    const months = [];
    for (let m = first; m <= last; m = nextMonth(m)) months.push(m);
    const series = {};
    for (const k of LONG_KEYS) {
        series[k] = months.map((m) => {
            const v = maps[k].get(m);
            if (!Number.isFinite(v)) throw new Error(`${k} has a gap at ${m}`);
            return v;
        });
    }
    const long = { start: months[0], through: months[months.length - 1], months, series };
    if (!validLong(long)) throw new Error(`long history too short or malformed (${months.length} months)`);
    return long;
}

/** Shape check for anything about to be served (live, baked, or a cached copy). */
export function validLong(long) {
    if (!long || !Array.isArray(long.months) || long.months.length < MIN_MONTHS || !long.series) return false;
    const n = long.months.length;
    if (long.through !== long.months[n - 1]) return false;
    for (let i = 1; i < n; i++) if (long.months[i] !== nextMonth(long.months[i - 1])) return false;
    return LONG_KEYS.every((k) => Array.isArray(long.series[k]) && long.series[k].length === n
        && long.series[k].every((v) => Number.isFinite(v) && v > -100));
}

const monthEnd = (ym) => {
    const [y, m] = ym.split('-').map(Number);
    return new Date(Date.UTC(y, m, 0)).toISOString().slice(0, 10);
};

/**
 * One long window, in the same shape computeWindow() gives the ETF windows:
 * `{ rel, f, b, from, to, spark, basis:'research' }`. Base = the end of the month
 * `years` before the last month, so 20Y = exactly 240 monthly returns.
 */
export function computeLongWindow(long, key, years) {
    const n = long.months.length;
    const m = years * 12;
    const fr = long.series[key];
    const br = long.series.market;
    if (!fr || n < m + 1) return null;
    const i0 = n - 1 - m;
    let F = 1;
    let B = 1;
    const rel = [0];
    for (let i = i0 + 1; i < n; i++) {
        F *= 1 + fr[i] / 100;
        B *= 1 + br[i] / 100;
        rel.push((F / B - 1) * 100);
    }
    const spark = [];
    for (let k = 0; k < SPARK_SLOTS; k++) spark.push(round2(rel[Math.round((k * (rel.length - 1)) / (SPARK_SLOTS - 1))]));
    return {
        rel: round2(rel[rel.length - 1]),
        f: round2((F - 1) * 100),
        b: round2((B - 1) * 100),
        from: monthEnd(long.months[i0]),
        to: monthEnd(long.months[n - 1]),
        spark,
        basis: 'research',
    };
}

/**
 * Add the 20Y/30Y/40Y windows to a factor payload. Never throws; returns the payload
 * unchanged if `long` is missing or malformed (the long buttons then disable).
 * @param {string} source  'live' | 'baked <date>' — shown in _meta, never guessed.
 */
export function attachLong(payload, long, source) {
    try {
        if (!payload || !Array.isArray(payload.factors) || !validLong(long)) return payload;
        const factors = payload.factors.map((f) => {
            if (!long.series[f.key] || f.key === 'market') return f;
            const windows = { ...(f.windows || {}) };
            for (const w of LONG_WINDOWS) windows[w] = computeLongWindow(long, f.key, LONG_YEARS[w]);
            return { ...f, windows, longProxy: LONG_SOURCES[f.key].label };
        });
        const shortWins = (payload.windows || []).filter((w) => !LONG_WINDOWS.includes(w));
        return {
            ...payload,
            factors,
            windows: [...shortWins, ...LONG_WINDOWS],
            long: {
                through: long.through,
                start: long.start,
                source,
                provider: KF_PROVIDER,
                basis: 'total return, monthly',
                benchmark: LONG_SOURCES.market.label,
            },
        };
    } catch {
        return payload;
    }
}

/** Parse the six downloaded zips (keyed like LONG_SOURCES) into a validated long history. */
export function longFromZips(zips) {
    const byKey = {};
    for (const k of LONG_KEYS) byKey[k] = pickSeries(parseFrenchCsv(unzipFirst(zips[k])), LONG_SOURCES[k].columns);
    return buildLong(byKey);
}
