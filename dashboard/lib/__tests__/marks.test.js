import {
    classify, parseValue, todayET, isUnitJump,
    buildSeries, historyFor, markFor, buildDigest, SHEET_METRICS,
} from '../marks';

/** Build a [{date,value}] series from compact [date, value] pairs. */
const S = (pairs) => pairs.map(([date, value]) => ({ date, value }));

/** A flat series long enough to satisfy the 2σ minimum sample size. */
function flat(n, value, endDate = '2026-08-25') {
    const out = [];
    const d = new Date(`${endDate}T00:00:00Z`);
    for (let i = n - 1; i >= 0; i--) {
        const t = new Date(d.getTime() - i * 86400000);
        out.push({ date: t.toISOString().slice(0, 10), value });
    }
    return out;
}

describe('classify', () => {
    it('marks slow-cadence metrics as print metrics', () => {
        ['sahmRule', 'sentiment', 'claims', 'nfci', 'm2', 'retail', 'housing', 'indpro',
            'jolts', 'durable', 'savings', 'rentIndex', 'mortgagePayment', 'mortgageRate',
            'atnhpi', 'aaiiDiff', 'profitMargin'].forEach(k => {
                expect(classify(k)).toBe('print');
            });
    });

    it('marks daily-but-meaningful metrics as move metrics', () => {
        ['yieldCurve', 'creditSpread', 'realYields', 'peRatio', 'copperGold'].forEach(k => {
            expect(classify(k)).toBe('move');
        });
    });

    it('never marks anything that changes every day', () => {
        ['tnx', 't2y', 'dxy', 'cl', 'usdcad', 'usdinr', 'usdbdt', 'inrbdt', 'cadinr',
            'cadbdt', 'gold', 'btc', 'vixCurrent', 'vix3m', 'vixFearGreed', 'lei'].forEach(k => {
                expect(classify(k)).toBe('none');
            });
    });

    it('returns none for an unknown key rather than throwing', () => {
        expect(classify('somethingNew')).toBe('none');
        expect(classify(undefined)).toBe('none');
    });

    it('every sheet metric carries the measured change rate that justifies its class', () => {
        Object.entries(SHEET_METRICS).forEach(([key, m]) => {
            expect(typeof m.col).toBe('number');
            expect(typeof m.rate).toBe('number');
            if (m.kind === 'print') expect(m.rate).toBeLessThan(0.2);
            if (m.kind === 'move') expect(m.rate).toBeGreaterThanOrEqual(0.2);
        });
    });
});

describe('parseValue', () => {
    it('reads plain and formatted numbers', () => {
        expect(parseValue('1975.52')).toBe(1975.52);
        expect(parseValue('-0.56')).toBe(-0.56);
        expect(parseValue('1,239')).toBe(1239);
    });

    it('treats blanks and N/A as absent', () => {
        ['', '   ', 'N/A', 'n/a', 'GREED05', 'abc'].forEach(v => {
            expect(parseValue(v)).toBeNull();
        });
        expect(parseValue(undefined)).toBeNull();
        expect(parseValue(null)).toBeNull();
    });

    it('treats an exact zero as absent — older rows use 0 as a missing sentinel', () => {
        // 2026-05-08: seven metrics went 0 -> a real value when the scraper was fixed.
        // Suppressing a genuine 0.00 print is a false negative we accept; lighting a
        // false mark is not.
        expect(parseValue('0')).toBeNull();
        expect(parseValue('0.00')).toBeNull();
    });
});

describe('todayET', () => {
    it('resolves the New York date, not the server date', () => {
        // 2026-08-26 01:30 UTC is still 2026-08-25 in New York (21:30 EDT).
        expect(todayET(new Date('2026-08-26T01:30:00Z'))).toBe('2026-08-25');
    });

    it('rolls over at New York midnight', () => {
        expect(todayET(new Date('2026-08-26T03:59:00Z'))).toBe('2026-08-25');
        expect(todayET(new Date('2026-08-26T04:01:00Z'))).toBe('2026-08-26');
    });

    it('returns an ISO date string that sorts lexically', () => {
        expect(todayET(new Date('2026-01-05T12:00:00Z'))).toBe('2026-01-05');
    });
});

describe('isUnitJump', () => {
    it('catches the 2026-03-18 unit switch', () => {
        expect(isUnitJump(212000, 212)).toBe(true);      // claims
        expect(isUnitJump(1487000, 1487)).toBe(true);    // housing starts
        expect(isUnitJump(1487, 1487000)).toBe(true);    // and in reverse
    });

    it('leaves ordinary moves alone', () => {
        expect(isUnitJump(212, 214)).toBe(false);
        expect(isUnitJump(2.8, -8.8)).toBe(false);       // AAII sign flip
        expect(isUnitJump(8.19, 5.0)).toBe(false);
        expect(isUnitJump(0.37, 0.1)).toBe(false);
    });

    it('does not fire on merely large ratios that are not near a factor of 1000', () => {
        expect(isUnitJump(1, 150)).toBe(false);
        expect(isUnitJump(1, 10)).toBe(false);
    });
});

describe('buildSeries', () => {
    const rows = [
        ['2026-08-23', '10', '1,000'],
        ['2026-08-24', 'N/A', '0'],
        ['2026-08-24', '11', '2000'],   // duplicate date — the later row wins
        ['2026-08-25', '12', ''],
    ];

    it('collapses duplicate dates to the last row for that date', () => {
        expect(buildSeries(rows, 1)).toEqual(S([
            ['2026-08-23', 10], ['2026-08-24', 11], ['2026-08-25', 12],
        ]));
    });

    it('drops absent values entirely rather than carrying them', () => {
        expect(buildSeries(rows, 2)).toEqual(S([['2026-08-23', 1000], ['2026-08-24', 2000]]));
    });

    it('returns an empty series for a column that never had data', () => {
        expect(buildSeries(rows, 9)).toEqual([]);
    });
});

describe('historyFor', () => {
    it('takes the baseline from the last day strictly before today', () => {
        const s = S([['2026-08-23', 6.47], ['2026-08-24', 6.50], ['2026-08-25', 6.55]]);
        expect(historyFor(s, '2026-08-25')).toMatchObject({ baseline: 6.50, baselineDate: '2026-08-24' });
    });

    it("falls back to the newest prior day when today's row is not written yet", () => {
        const s = S([['2026-08-22', 6.40], ['2026-08-23', 6.47]]);
        expect(historyFor(s, '2026-08-25')).toMatchObject({ baseline: 6.47, baselineDate: '2026-08-23' });
    });

    it('reports how long the baseline value had already been standing', () => {
        const s = S([
            ['2026-06-01', 8.19], ['2026-06-03', 8.19],
            ['2026-06-18', 8.19], ['2026-06-24', 8.19],
        ]);
        expect(historyFor(s, '2026-06-25').heldFrom).toBe('2026-06-01');
    });

    it('computes sigma only with enough prior moves', () => {
        const few = flat(10, 1, '2026-08-24').map((p, i) => ({ ...p, value: 1 + i * 0.01 }));
        expect(historyFor(few, '2026-08-25').sigma).toBeNull();

        const many = flat(40, 1, '2026-08-24').map((p, i) => ({ ...p, value: 1 + (i % 2) * 0.01 }));
        expect(historyFor(many, '2026-08-25').sigma).toBeGreaterThan(0);
    });

    it('returns null with no prior day at all', () => {
        expect(historyFor(S([['2026-08-25', 5]]), '2026-08-25')).toBeNull();
        expect(historyFor([], '2026-08-25')).toBeNull();
    });
});

describe('markFor — print metrics', () => {
    const entry = (over = {}) => ({
        baseline: 8.19, baselineDate: '2026-06-24', heldFrom: '2026-06-03',
        runs: [4.47, -2.71, 8.12, 8.19], sigma: null, dailyRuns: [8.19, 8.19], ...over,
    });

    it('marks a live value that differs from yesterday', () => {
        const m = markFor('durable', 5.0, entry(), '2026-06-25');
        expect(m).toMatchObject({ kind: 'print', prev: 8.19, value: 5.0, dir: -1, heldFrom: '2026-06-03' });
        expect(m.heldDays).toBe(22);
        expect(m.runs[m.runs.length - 1]).toBe(5.0);
    });

    it('is silent when the live value matches yesterday', () => {
        expect(markFor('durable', 8.19, entry(), '2026-06-25')).toBeNull();
    });

    it('marks as soon as the print lands, before the scraper has caught up', () => {
        // today's row does not exist yet; the baseline is still yesterday's number
        const m = markFor('claims', 224.25, entry({ baseline: 223.25, baselineDate: '2026-06-24' }), '2026-06-25');
        expect(m).toMatchObject({ prev: 223.25, value: 224.25, dir: 1 });
    });

    it('refuses a unit change', () => {
        expect(markFor('claims', 212, entry({ baseline: 212000 }), '2026-03-18')).toBeNull();
    });

    it('refuses when the live value is missing', () => {
        [null, undefined, NaN].forEach(v => expect(markFor('durable', v, entry(), '2026-06-25')).toBeNull());
    });

    it('refuses when there is no history entry', () => {
        expect(markFor('durable', 5.0, null, '2026-06-25')).toBeNull();
    });

    it('ignores float noise below the epsilon', () => {
        expect(markFor('profitMargin', 14.917120500741223,
            entry({ baseline: 14.917120500741222 }), '2026-08-25')).toBeNull();
    });

    it('never marks a metric classed none, however much it moved', () => {
        expect(markFor('btc', 78981, entry({ baseline: 61526 }), '2026-08-25')).toBeNull();
        expect(markFor('usdbdt', 122.45, entry({ baseline: 100 }), '2026-08-25')).toBeNull();
    });
});

describe('markFor — move metrics', () => {
    const entry = (over = {}) => ({
        baseline: 1.56, baselineDate: '2026-06-24', heldFrom: '2026-06-24',
        runs: [], sigma: 0.02, dailyRuns: [1.61, 1.58, 1.60, 1.55, 1.57, 1.56], ...over,
    });

    it('fires only past 2 sigma', () => {
        expect(markFor('copperGold', 1.55, entry(), '2026-06-25')).toBeNull();       // 0.5σ
        expect(markFor('copperGold', 1.50, entry(), '2026-06-25'))                   // 3σ
            .toMatchObject({ kind: 'move', prev: 1.56, dir: -1 });
    });

    it('is silent when sigma could not be computed', () => {
        expect(markFor('copperGold', 9.9, entry({ sigma: null }), '2026-06-25')).toBeNull();
        expect(markFor('copperGold', 9.9, entry({ sigma: 0 }), '2026-06-25')).toBeNull();
    });

    it('carries the daily series for the sparkline, live value last', () => {
        const m = markFor('copperGold', 1.50, entry(), '2026-06-25');
        expect(m.runs[m.runs.length - 1]).toBe(1.50);
    });

    it('reports direction', () => {
        expect(markFor('yieldCurve', 1.9, entry({ baseline: 1.0, sigma: 0.05 }), '2026-08-25').dir).toBe(1);
    });
});

describe('buildDigest', () => {
    const rows = [
        ['Date', 'Yield Curve', 'Profit Margin', 'Sahm Rule'],
        ['2026-08-23', '0.50', '14.83', '0.10'],
        ['2026-08-24', '0.50', '14.92', '0.10'],
    ];

    it('emits one entry per markable metric and none for the never-marked ones', () => {
        const d = buildDigest(rows, new Date('2026-08-25T14:00:00Z'));
        expect(d.today).toBe('2026-08-25');
        expect(d.metrics.profitMargin).toMatchObject({ kind: 'print', baseline: 14.92 });
        expect(d.metrics.btc).toBeUndefined();
        expect(d.metrics.vixCurrent).toBeUndefined();
    });

    it('keeps the payload lean — print entries carry no sigma, move entries no distinct runs', () => {
        const d = buildDigest(rows, new Date('2026-08-25T14:00:00Z'));
        expect(d.metrics.profitMargin.sigma).toBeUndefined();
        expect(d.metrics.yieldCurve.runs).toBeUndefined();
    });

    it('survives a malformed sheet without throwing', () => {
        expect(() => buildDigest([], new Date())).not.toThrow();
        expect(() => buildDigest([['garbage']], new Date())).not.toThrow();
        expect(buildDigest([], new Date()).metrics).toEqual({});
    });
});
