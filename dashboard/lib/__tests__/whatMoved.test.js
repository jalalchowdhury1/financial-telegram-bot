import {
    change, sigmaOf, buildMoveDigest, collectMoves, fmtMove, roundsToZero, bizDaysBetween, marketDateOf,
    FG_SIGMA_BAKED, TOP_N, SHEET_MOVERS,
} from '../whatMoved';

const NOW = new Date('2026-09-26T16:00:00Z'); // noon in New York
/** n prices alternating a, b, a, b … dated back from 2026-09-25 */
const alt = (n, a, b, last = '2026-09-25') => Array.from({ length: n }, (_, i) => ({
    date: new Date(Date.parse(`${last}T00:00:00Z`) - (n - 1 - i) * 86400000).toISOString().slice(0, 10),
    price: i % 2 ? b : a,
}));

test('change: percent, basis points, points', () => {
    expect(change(100, 102, 'pct')).toBeCloseTo(2);
    expect(change(4.1, 4.05, 'bp')).toBeCloseTo(-5);
    expect(change(30, 37, 'pts')).toBe(7);
    expect(change(0, 1, 'pct')).toBeNull();
    expect(change(NaN, 1, 'pts')).toBeNull();
});

test('sigmaOf: σ of daily changes, needs 20, skips flat (weekend) repeats', () => {
    const v = Array.from({ length: 21 }, (_, i) => i % 2); // 0,1,0,1… → ±1
    expect(sigmaOf(v, 'pts')).toBeCloseTo(1);
    expect(sigmaOf(v.slice(0, 20), 'pts')).toBeNull(); // 19 changes
    const withRepeats = v.flatMap((x) => [x, x]); // every value twice: zeros skipped
    expect(sigmaOf(withRepeats, 'pts')).toBeCloseTo(1);
    expect(sigmaOf(Array(30).fill(5), 'pts')).toBeNull();
    expect(sigmaOf(null, 'pts')).toBeNull();
});

test('fmtMove / roundsToZero', () => {
    expect(fmtMove(0.54, 'pct')).toBe('+0.5%');
    expect(fmtMove(-0.07, 'pct')).toBe('−0.07%');
    expect(fmtMove(6.6, 'bp')).toBe('+7bp');
    expect(fmtMove(-3.2, 'pts')).toBe('−3');
    expect(roundsToZero(0.004, 'pct')).toBe(true);
    expect(roundsToZero(0.4, 'bp')).toBe(true);
    expect(roundsToZero(0.6, 'pts')).toBe(false);
});

test('buildMoveDigest: backup σ from the sheet columns, rows after today ignored', () => {
    const row = (date, vix) => { const r = Array(71).fill(''); r[0] = date; r[SHEET_MOVERS.vix.col] = String(vix); return r; };
    const rows = [['Date']];
    for (let i = 0; i < 25; i++) rows.push(row(`2026-08-${String(i + 1).padStart(2, '0')}`, i % 2 ? 22 : 20));
    const d = buildMoveDigest(rows, NOW);
    expect(d.vix.sigma).toBeCloseTo(9.545, 2); // +10 % / −9.09 % swings
    rows.push(row('2026-09-27', 99)); // tomorrow's row: never read
    expect(buildMoveDigest(rows, NOW).vix.sigma).toBeCloseTo(9.545, 2);
    expect(d.fg).toBeUndefined(); // too few rows → absent, never zero
});

const feeds = () => ({
    spy: { current: 102, dailyChange: { value: 2, pct: 2 }, chartHistory: alt(30, 100, 101) },
    fg: { score: 40, previousClose: 30, _meta: { source: 'CNN' } },
    extra: {
        rates: { tnx: { current: 4.1, dailyChange: { value: 0.1, pct: 2.5 }, history: alt(30, 4.0, 4.05), lastDate: '2026-09-24' } },
        commodities: {
            btc: { current: 84000, dailyChange: { value: 0, pct: 0 }, history: alt(30, 80000, 82000) },
            gc: { current: 4300, dailyChange: { value: 20, pct: 0.5 }, history: [{ date: '2026-09-25', price: 4300 }] },
        },
    },
    vol: { vixDay: { value: 20, prev: 16, asOf: '2026-09-25', sigma: 5 } },
    history: { moves: { vix: { sigma: 50 } } },
});

test('collectMoves ranks by × a normal day and leaves out what it cannot judge', () => {
    const m = collectMoves(feeds(), NOW);
    expect(m.map((x) => x.key)).toEqual(['vix', 'fg', 'spy', 'tnx']);
    const [vix, fg, spy, tnx] = m;
    expect(vix).toMatchObject({ text: '+25.0%', prev: 16, value: 20, big: true, jump: 'Volatility' });
    expect(vix.z).toBeCloseTo(5);
    expect(fg).toMatchObject({ text: '+10', jump: 'Fear & Greed' });
    expect(fg.z).toBeCloseTo(10 / FG_SIGMA_BAKED);
    expect(spy.text).toBe('+2.0%');
    expect(tnx).toMatchObject({ text: '+10bp', big: true, day: 'Thu' }); // FRED: a business day behind
    expect(vix.day).toBeNull();
    expect(spy.day).toBeNull();
    // btc: flat day · gc: no history for σ · cl: no data
});

test('collectMoves: F&G uses the sheet σ once it exists; stale quotes are dropped', () => {
    const f = feeds();
    f.history.moves.fg = { recent: [], sigma: 2 };
    f.extra.rates.tnx.lastDate = '2026-09-10';
    const m = collectMoves(f, NOW);
    expect(m.find((x) => x.key === 'fg').z).toBeCloseTo(5);
    expect(m.find((x) => x.key === 'tnx')).toBeUndefined();
});

test('collectMoves: VIX σ falls back to the sheet; a stale VIX is dropped; junk gives []', () => {
    const f = feeds();
    f.vol.vixDay.sigma = null;
    expect(collectMoves(f, NOW).find((x) => x.key === 'vix').z).toBeCloseTo(0.5); // 25 % / 50
    f.vol.vixDay.asOf = '2026-09-01';
    expect(collectMoves(f, NOW).find((x) => x.key === 'vix')).toBeUndefined();
    expect(collectMoves({}, NOW)).toEqual([]);
    expect(collectMoves({ spy: { error: 'x' }, fg: 'junk', extra: { rates: 5 }, vol: [], history: null }, NOW)).toEqual([]);
});

test('collectMoves never returns more than TOP_N', () => {
    const f = feeds();
    f.extra.commodities.btc.dailyChange = { value: 5000, pct: 6 };
    f.extra.commodities.cl = { current: 90, dailyChange: { value: 3, pct: 3.4 }, history: alt(30, 88, 90) };
    expect(collectMoves(f, NOW)).toHaveLength(TOP_N);
});

test('bizDaysBetween / marketDateOf', () => {
    expect(bizDaysBetween('2026-09-24', '2026-09-25')).toBe(1); // Thu → Fri
    expect(bizDaysBetween('2026-09-25', '2026-09-28')).toBe(1); // Fri → Mon
    expect(bizDaysBetween('2026-09-23', '2026-09-25')).toBe(2);
    expect(bizDaysBetween('2026-09-25', '2026-09-25')).toBe(0);
    expect(marketDateOf(feeds())).toBe('2026-09-25');
    expect(marketDateOf({ spy: { error: 'x' }, vol: null })).toBeNull();
});

test('collectMoves: a feed 2+ business days behind the market date is dropped, 1 is tagged', () => {
    const f = feeds();
    f.extra.rates.tnx.lastDate = '2026-09-23'; // Wed vs the market's Fri
    expect(collectMoves(f, NOW).find((x) => x.key === 'tnx')).toBeUndefined();
    f.extra.rates.tnx.lastDate = '2026-09-25';
    expect(collectMoves(f, NOW).find((x) => x.key === 'tnx').day).toBeNull();
});

test('collectMoves: F&G from the VIX proxy or a stale cache is never ranked', () => {
    for (const source of ['Yahoo ^VIX Proxy', 'FRED VIXCLS Proxy', 'Stale Cache', 'Failed', undefined]) {
        const f = feeds();
        f.fg._meta = source ? { source } : undefined;
        expect(collectMoves(f, NOW).find((x) => x.key === 'fg')).toBeUndefined();
    }
    const f = feeds();
    f.fg._meta.source = 'RapidAPI';
    expect(collectMoves(f, NOW).find((x) => x.key === 'fg')).toBeDefined();
});
