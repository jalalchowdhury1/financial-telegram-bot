/**
 * Tests for assemblePills (the pure assembly logic used by /api/jev-pills/route.js).
 *
 * These tests do NOT import next/server — they test the pure function exported from
 * lib/jevPills.js.
 */

import { assemblePills } from '../jevPills';

// ---------------------------------------------------------------------------
// Sample raw payloads (realistic shapes matching the /api/* routes)
// ---------------------------------------------------------------------------

const sampleRaw = {
    spy: {
        current: 561.69,
        dailyChange: { pct: 0.12 },
        ma200: { pct: 6.3 },
        week52High: { pct: -2.1 },
        rsi: 48.5,
        _meta: { source: 'lambda+polygon' },
    },
    fg: {
        score: 55,
        rating: 'Neutral',
        previousWeek: 50,
        previousMonth: 48,
        _meta: { source: 'cnn' },
    },
    vol: {
        tickers: [
            { ticker: 'SPY', iv: 15.2, ivRank1y: 35, ivPctile1y: 32, rv21: 12.5, vrp: 2.7 },
        ],
        _meta: { source: 'VIX:cboe · SPY:cnbc' },
    },
    fred: {
        yieldCurve: { value: 0.36 },
        indicators: {
            sahmRule: { value: 0.1 },
            claims: { value: 220 },
            creditSpread: { value: 1.2 },
            realYields: { value: 1.5 },
            copperGold: { value: 1.4 },
            sentiment: { value: 70 },
        },
        checklist: { nfci: { value: -0.3 } },
        _meta: { source: 'St. Louis Fed' },
    },
    breadth: {
        pairs: {
            rspSpy: { ratio: 1.02, chg20Pct: 1.5, chg60Pct: 3.0, vs50dPct: 0.5 },
            iwmSpy: { ratio: 0.95, chg20Pct: 2.0, chg60Pct: 4.0, vs50dPct: 1.0 },
            xlkXlu: { ratio: 1.10, chg20Pct: 0.8, chg60Pct: 2.0, vs50dPct: 0.3 },
            hygLqd: { ratio: 0.98, chg20Pct: 0.2, chg60Pct: 0.5, vs50dPct: 0.1 },
        },
        _meta: { source: 'RSP:polygon · SPY:polygon' },
    },
    sheets: {
        AAIIDiff: '5.50%',
        _meta: { source: 'google-sheets' },
    },
    t10y3m: 0.15,
};

// ---------------------------------------------------------------------------
// assemblePills basic shape
// ---------------------------------------------------------------------------

describe('assemblePills', () => {
    test('returns the expected payload shape with mode on', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'on',
        });

        // Top-level keys
        expect(result).toHaveProperty('enabled', true);
        expect(result).toHaveProperty('mode', 'on');
        expect(result).toHaveProperty('asOf');
        expect(result).toHaveProperty('state');
        expect(result).toHaveProperty('pills');
        expect(result).toHaveProperty('conflictPairs');
        expect(result).toHaveProperty('since');
        expect(result).toHaveProperty('_meta');

        // Five pills
        expect(result.pills).toHaveProperty('regime');
        expect(result.pills).toHaveProperty('recession');
        expect(result.pills).toHaveProperty('breadth');
        expect(result.pills).toHaveProperty('hedging');
        expect(result.pills).toHaveProperty('conflict');

        // Each pill has verdict, p, by, reason
        for (const key of ['regime', 'recession', 'breadth', 'hedging', 'conflict']) {
            expect(result.pills[key]).toHaveProperty('verdict');
            expect(result.pills[key]).toHaveProperty('p');
            expect(result.pills[key]).toHaveProperty('by');
            expect(result.pills[key]).toHaveProperty('reason');
        }

        // Since has direction, changed, noBaseline
        expect(result.since).toHaveProperty('direction');
        expect(result.since).toHaveProperty('changed');
        expect(result.since).toHaveProperty('noBaseline');

        // _meta has jev and sources
        expect(result._meta).toHaveProperty('jev');
        expect(result._meta).toHaveProperty('sources');
    });

    test('mode=rules produces rules-only verdicts (all by: rule)', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        expect(result.mode).toBe('rules');
        expect(result._meta.jev).toBe('rules');
        for (const key of ['regime', 'recession', 'breadth', 'hedging', 'conflict']) {
            expect(result.pills[key].by).toBe('rule');
        }
    });

    test('mode=on with no jevAnswers still has by: rule (Jev unavailable)', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'on',
        });

        expect(result.mode).toBe('on');
        for (const key of ['regime', 'recession', 'breadth', 'hedging', 'conflict']) {
            expect(result.pills[key].by).toBe('rule');
        }
    });

    test('since has noBaseline when yesterday is null', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'on',
        });

        expect(result.since.noBaseline).toBe(true);
        expect(result.since.direction).toBe('none');
        expect(result.since.date).toBeNull();
    });

    test('since shows diff when yesterday has different verdicts', () => {
        const yesterday = {
            date: '2026-09-19',
            pills: {
                regime: { verdict: 'risk-off' },
                recession: { verdict: 'low' },
                breadth: { verdict: 'broad' },
                hedging: { verdict: 'fair' },
                conflict: { verdict: 'aligned' },
            },
        };

        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday,
            mode: 'on',
        });

        // With sampleRaw: regimeScore=3(ma200>0=+1,fg=55>=50=+1,hygLqd=0.2>0=+1) ⇒ risk-on
        // Yesterday was risk-off, so regime changed toward mild
        // recession was low → still low, no change
        // etc.
        expect(result.since.noBaseline).toBe(false);
        expect(result.since.date).toBe('2026-09-19');
        // At minimum regime softened
        expect(result.since.changed.length).toBeGreaterThanOrEqual(1);
        expect(result.since.direction).toMatch(/softening|mixed|none/);
    });

    test('sources collected from raw route _meta', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'on',
        });

        expect(result._meta.sources.spy).toBe('lambda+polygon');
        expect(result._meta.sources.fg).toBe('cnn');
        expect(result._meta.sources.breadth).toBe('RSP:polygon · SPY:polygon');
    });

    test('inputSources carried into _meta', () => {
        const inputSources = { t10y3m: 'fredcsv', nfci: 'fredcsv', claims: 'fred-route', sahm: 'fred-route' };
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'on',
            inputSources,
        });

        expect(result._meta).toHaveProperty('inputSources');
        expect(result._meta.inputSources).toEqual(inputSources);
    });

    test('inputSources defaults to empty object when not provided', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'on',
        });

        expect(result._meta.inputSources).toEqual({});
    });
});

// ---------------------------------------------------------------------------
// Rule verdict behavior with the sample data
// ---------------------------------------------------------------------------

describe('assemblePills — rule verdicts', () => {
    test('sample data produces risk-on regime', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        // ma200Pct=6.3>0 → +1, fg.score=55≥50 → +1, hygLqd.chg20Pct=0.2>0 → +1
        // Score 3 → risk-on
        expect(result.pills.regime.verdict).toBe('risk-on');
    });

    test('sample data produces low recession', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        // sahmRule=0.1 < 0.2, yieldCurve=0.36 > 0, claims=220 < 260, nfci=-0.3 < 0
        expect(result.pills.recession.verdict).toBe('low');
    });

    test('sample data produces broad breadth', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        // rspSpy.chg20Pct=1.5 > 0, iwmSpy.chg20Pct=2.0 > 0 → broad
        expect(result.pills.breadth.verdict).toBe('broad');
    });

    test('sample data produces fair hedging', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        // IV percentile 32 (not <20, not >70), VRP 2.7 (not >10, not <6 is only required
        // for cheap, and cheap needs BOTH ivPctile<20 AND vrp<6)
        // Not expensive (neither ivPctile>70 nor vrp>10) → fair
        expect(result.pills.hedging.verdict).toBe('fair');
    });

    test('sample data produces aligned conflict (no divergences)', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        // fg.score=55 (35-65 range, neither fear nor greed extreme)
        // yieldCurve=0.36 > 0, t10y3m=0.15 > 0 → same sign
        // ma200Pct=6.3 > 0, hygLqd=0.2 > -1.5 → no credit vs equities
        // high52Pct=-2.1 < -2 (not > -2)
        expect(result.pills.conflict.verdict).toBe('aligned');
    });
});

// ---------------------------------------------------------------------------
// Jev override
// ---------------------------------------------------------------------------

describe('assemblePills — Jev override', () => {
    test('jevAnswers with p >= 0.6 override rule', () => {
        const jevAnswers = {
            regime: { verdict: 'risk-off', p: 0.82 },
            recession: { verdict: 'rising', p: 0.71 },
        };

        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers,
            yesterday: null,
            mode: 'on',
        });

        expect(result.pills.regime.by).toBe('jev');
        expect(result.pills.regime.verdict).toBe('risk-off');
        expect(result.pills.regime.p).toBe(0.82);

        expect(result.pills.recession.by).toBe('jev');
        expect(result.pills.recession.verdict).toBe('rising');
    });

    test('jevAnswers with p < 0.6 are ignored', () => {
        const jevAnswers = {
            regime: { verdict: 'risk-off', p: 0.45 },
        };

        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers,
            yesterday: null,
            mode: 'on',
        });

        // Rule says risk-on, jev says risk-off at p=0.45 → rule wins
        expect(result.pills.regime.by).toBe('rule');
        expect(result.pills.regime.verdict).toBe('risk-on');
    });

    test('jevAnswers with unknown verdict are ignored', () => {
        const jevAnswers = {
            regime: { verdict: 'super-risk-on', p: 0.9 },
        };

        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers,
            yesterday: null,
            mode: 'on',
        });

        expect(result.pills.regime.by).toBe('rule');
        expect(result.pills.regime.verdict).toBe('risk-on');
    });
});

// ---------------------------------------------------------------------------
// Edge cases: missing/null data
// ---------------------------------------------------------------------------

describe('assemblePills — edge cases', () => {
    test('handles null raw data', () => {
        const result = assemblePills({
            raw: null,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        expect(result.enabled).toBe(true);
        expect(result.pills.regime.verdict).toBe('n/a');
        expect(result.pills.regime.reason).toContain('no data');
    });

    test('handles empty raw object', () => {
        const result = assemblePills({
            raw: {},
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        expect(result.enabled).toBe(true);
        // All rule verdicts with missing inputs should default
        expect(result.pills.regime.verdict).toBe('neutral');
        expect(result.pills.recession.verdict).toBe('low');
        // No breadth data → narrow
        expect(result.pills.breadth.verdict).toBe('narrow');
    });

    test('handles missing fields gracefully', () => {
        const raw = {
            spy: { current: 500 },
            // no fg, no vol, no fred, no breadth
        };

        const result = assemblePills({
            raw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        // Should not throw; missing fields just become nulls in the data contract
        expect(result.pills.regime.verdict).toBe('neutral'); // all inputs null → score 0
    });

    test('buildState never contains NotSoBoring or FrontRunner', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'on',
        });

        expect(result.state).not.toContain('NotSoBoring');
        expect(result.state).not.toContain('FrontRunner');
    });

    test('conflictPairs listed in output', () => {
        // Force a conflict by making fg score extreme and ma200 strong
        const raw = {
            ...sampleRaw,
            fg: { score: 30, rating: 'Fear' },
            spy: { ...sampleRaw.spy, ma200: { pct: 5 } },
        };

        const result = assemblePills({
            raw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        // Should have at least the sentiment vs price conflict
        expect(result.conflictPairs.length).toBeGreaterThanOrEqual(1);
        expect(result.conflictPairs[0].pair).toMatch(/sentiment vs price/);
    });
});

// ---------------------------------------------------------------------------
// factors in the payload
// ---------------------------------------------------------------------------

describe('assemblePills — factors', () => {
    test('payload includes factors key with all five pills', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        expect(result).toHaveProperty('factors');
        expect(result.factors).toHaveProperty('regime');
        expect(result.factors).toHaveProperty('recession');
        expect(result.factors).toHaveProperty('breadth');
        expect(result.factors).toHaveProperty('hedging');
        expect(result.factors).toHaveProperty('conflict');
    });

    test('each factor has summary and rows', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'on',
        });

        for (const pill of ['regime', 'recession', 'breadth', 'hedging', 'conflict']) {
            expect(result.factors[pill]).toHaveProperty('summary');
            expect(result.factors[pill]).toHaveProperty('rows');
            expect(Array.isArray(result.factors[pill].rows)).toBe(true);
            for (const row of result.factors[pill].rows) {
                expect(row).toHaveProperty('label');
                expect(row).toHaveProperty('value');
                expect(row).toHaveProperty('test');
                expect(row).toHaveProperty('hit');
                expect(row).toHaveProperty('effect');
            }
        }
    });

    test('null raw data yields factors with no-data summaries', () => {
        const result = assemblePills({
            raw: null,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        expect(result.factors).toBeDefined();
        for (const pill of ['regime', 'recession', 'breadth', 'hedging', 'conflict']) {
            expect(result.factors[pill].summary).toBe('no data');
        }
    });
});

// ---------------------------------------------------------------------------
// jev field on each pill
// ---------------------------------------------------------------------------

describe('assemblePills — jev field on pills', () => {
    test('jev answers present yield jev field on each pill', () => {
        const jevAnswers = {
            regime: { verdict: 'risk-off', p: 0.82 },
            recession: { verdict: 'rising', p: 0.71 },
            breadth: { verdict: 'narrow', p: 0.65 },
            hedging: { verdict: 'expensive', p: 0.73 },
            conflict: { verdict: 'mild-divergence', p: 0.68 },
        };

        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers,
            yesterday: null,
            mode: 'on',
        });

        for (const pill of ['regime', 'recession', 'breadth', 'hedging', 'conflict']) {
            expect(result.pills[pill]).toHaveProperty('jev');
            expect(result.pills[pill].jev).toEqual({
                verdict: jevAnswers[pill].verdict,
                p: jevAnswers[pill].p,
            });
        }
    });

    test('partial jev answers: missing pills get null jev', () => {
        const jevAnswers = {
            regime: { verdict: 'risk-off', p: 0.82 },
            // recession, breadth, hedging, conflict NOT provided
        };

        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers,
            yesterday: null,
            mode: 'on',
        });

        expect(result.pills.regime.jev).toEqual({ verdict: 'risk-off', p: 0.82 });
        expect(result.pills.recession.jev).toBeNull();
        expect(result.pills.breadth.jev).toBeNull();
        expect(result.pills.hedging.jev).toBeNull();
        expect(result.pills.conflict.jev).toBeNull();
    });

    test('null jevAnswers gives null jev on every pill', () => {
        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers: null,
            yesterday: null,
            mode: 'rules',
        });

        for (const pill of ['regime', 'recession', 'breadth', 'hedging', 'conflict']) {
            expect(result.pills[pill].jev).toBeNull();
        }
    });

    test('jev below floor (p < 0.6) still stores jev field (does NOT change mergeVerdicts)', () => {
        const jevAnswers = {
            regime: { verdict: 'risk-off', p: 0.45 },
        };

        const result = assemblePills({
            raw: sampleRaw,
            jevAnswers,
            yesterday: null,
            mode: 'on',
        });

        // jev field stores the raw answer even when below floor
        expect(result.pills.regime.jev).toEqual({ verdict: 'risk-off', p: 0.45 });
        // But verdict is still rule's (risk-on) since p < 0.6
        expect(result.pills.regime.verdict).toBe('risk-on');
        expect(result.pills.regime.by).toBe('rule');
    });
});
describe('breadth legs pass through toData into the regime row', () => {
    test('HYG/LQD row carries both legs from the raw breadth payload', () => {
        const raw = JSON.parse(JSON.stringify(sampleRaw));
        raw.breadth.pairs.hygLqd = { ...raw.breadth.pairs.hygLqd, chg20Pct: -0.0125, legs: { HYG: -1.29, LQD: -1.28 } };
        const out = assemblePills({ raw, jevAnswers: null, yesterday: null, mode: 'rules' });
        const row = out.factors.regime.rows.find((r) => r.label === 'HYG/LQD 20d');
        expect(row.value).toBe('-0.01%');
        expect(row.note).toBe('HYG -1.3% · LQD -1.3%');
        expect(out.pills.regime.reason).toContain('legs 20d: HYG -1.3% · LQD -1.3%');
    });
});

// ---------------------------------------------------------------------------
// repairPillInputs — route glue
// ---------------------------------------------------------------------------

jest.mock('../../lib/sources', () => ({
    fredObservations: jest.fn(),
    fredGraphCsv: jest.fn(),
    treasuryYieldCurveCsv: jest.fn(),
}));
jest.mock('../../lib/sheetLkg', () => ({
    fetchSheetLkg: jest.fn(),
}));
jest.mock('../../lib/store', () => ({
    loadLastGood: jest.fn(),
    saveLastGood: jest.fn(),
}));

import { fredObservations, fredGraphCsv, treasuryYieldCurveCsv } from '../../lib/sources';
import { fetchSheetLkg } from '../../lib/sheetLkg';
import { loadLastGood, saveLastGood } from '../../lib/store';
import { repairPillInputs } from '../../app/api/jev-pills/route';

describe('repairPillInputs', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        // Default mock: return empty CSV for any series not explicitly set up;
        // tests that need real data override this.
        fredGraphCsv.mockImplementation((seriesId) => {
            // Return a 3-row CSV with the series id as header, dates recent enough
            // to be fresh (within 7 days of any reasonable test now).
            const today = '2026-01-09';
            const yesterday = '2026-01-08';
            const twoDaysAgo = '2026-01-07';
            return Promise.resolve(`observation_date,${seriesId}\n${twoDaysAgo},1.0\n${yesterday},1.1\n${today},1.2`);
        });
        fredObservations.mockResolvedValue([]);
        treasuryYieldCurveCsv.mockRejectedValue(new Error('treasury down'));
        fetchSheetLkg.mockResolvedValue(null);
    });

    test('sets t10y3m from fred when fredKey is present and API returns data', async () => {
        // fredObservations returns DESCENDING (newest first)
        fredObservations.mockResolvedValue([
            { date: '2026-01-05', value: 0.87 },
            { date: '2026-01-04', value: 0.85 },
            { date: '2026-01-03', value: 0.83 },
        ]);
        const raw = {};
        const inputSources = await repairPillInputs(raw, { fredKey: 'test-key', now: new Date('2026-01-07') });
        expect(raw.t10y3m).toBe(0.87);
        expect(inputSources.t10y3m).toBe('fred');
    });

    test('falls back to fredcsv when no fredKey', async () => {
        fredGraphCsv.mockResolvedValue('observation_date,T10Y3M\n2026-01-04,0.80\n2026-01-05,0.87');
        const raw = {};
        const inputSources = await repairPillInputs(raw, { fredKey: null, now: new Date('2026-01-07') });
        expect(raw.t10y3m).toBe(0.87);
        expect(inputSources.t10y3m).toBe('fredcsv');
    });

    test('repairs NFCI when fred.checklist.nfci.value is missing', async () => {
        fredGraphCsv.mockImplementation((seriesId) => {
            if (seriesId === 'NFCI') {
                return Promise.resolve('observation_date,NFCI\n2026-01-02,-0.5\n2026-01-09,-0.45');
            }
            // Default: return data fresh enough
            return Promise.resolve('observation_date,T10Y3M\n2026-01-02,0.5\n2026-01-09,0.6');
        });
        const raw = { fred: { checklist: {} } };
        const inputSources = await repairPillInputs(raw, { now: new Date('2026-01-12') });
        expect(raw.fred.checklist.nfci.value).toBe(-0.45);
        expect(raw.fred.checklist.nfci.source).toBe('fredcsv');
        expect(inputSources.nfci).toBe('fredcsv');
    });

    test('does NOT repair NFCI when fred.checklist.nfci.value is already a finite number', async () => {
        const raw = { fred: { checklist: { nfci: { value: -0.3 } } } };
        const inputSources = await repairPillInputs(raw, {});
        // fredGraphCsv should NOT be called for NFCI (but may be called for T10Y3M)
        expect(inputSources.nfci).toBe('fred-route');
    });

    test('repairs claims from horsemen history when available', async () => {
        const raw = {
            fred: {
                indicators: {},
                horsemen: {
                    claims: {
                        history: [
                            { date: '2026-01-01', value: 210000 },
                            { date: '2026-01-08', value: 215000 },
                            { date: '2026-01-15', value: 220000 },
                            { date: '2026-01-22', value: 225000 },
                        ],
                    },
                },
            },
        };
        const inputSources = await repairPillInputs(raw, { now: new Date('2026-01-25') });
        // 4-week avg = (210000+215000+220000+225000)/4000 = 217.5
        expect(raw.fred.indicators.claims.value).toBe(217.5);
        expect(raw.fred.indicators.claims.source).toBe('horsemen');
        expect(inputSources.claims).toBe('horsemen');
    });

    test('repairs sahm from horsemen unemployment history when available', async () => {
        // Need 12 months of data. Min of 12 = 3.5, last 3 mean = (4.0+4.1+4.2)/3 = 4.1
        // Sahm = 4.1 - 3.5 = 0.6
        const values = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 3.9, 3.8, 3.9, 4.0, 4.1, 4.2];
        const history = values.map((v, i) => ({ date: `2026-${String(i + 1).padStart(2, '0')}-01`, value: v }));
        const raw = {
            fred: {
                indicators: {},
                horsemen: {
                    unemployment: { history },
                },
            },
        };
        const inputSources = await repairPillInputs(raw, { now: new Date('2027-01-15') });
        expect(raw.fred.indicators.sahmRule.value).toBeCloseTo(0.6, 5);
        expect(raw.fred.indicators.sahmRule.source).toBe('horsemen');
        expect(inputSources.sahm).toBe('horsemen');
    });

    test('returns fred-route label when sibling already has the data', async () => {
        const raw = {
            fred: {
                indicators: { claims: { value: 220 }, sahmRule: { value: 0.1 } },
                checklist: { nfci: { value: -0.3 } },
            },
            t10y3m: null,
        };
        const inputSources = await repairPillInputs(raw, { fredKey: 'test-key', now: new Date('2026-01-07') });
        // claims, sahm, nfci should stay at fred-route since their values are finite
        expect(inputSources.claims).toBe('fred-route');
        expect(inputSources.sahm).toBe('fred-route');
        expect(inputSources.nfci).toBe('fred-route');
        // t10y3m is always resolved, so it won't be fred-route if fred was used
        // Since fredKey was provided, it will try fred first
    });

    test('never throws — returns inputSources even with null raw', async () => {
        const result = await repairPillInputs(null, {});
        expect(result).toEqual({ t10y3m: 'fred-route', nfci: 'fred-route', claims: 'fred-route', sahm: 'fred-route' });
    });
});

// ---------------------------------------------------------------------------
// repairPillInputs — v3.1 tiers (treasury, sheet, seeded last-good, faults, diag)
// ---------------------------------------------------------------------------

describe('repairPillInputs — v3.1 tiers', () => {
    const TREASURY = 'Date,"1 Mo","3 Mo","2 Yr","10 Yr"\n01/06/2026,4.00,4.14,4.76,5.01\n01/05/2026,4.00,4.15,4.77,5.03';
    const mkStore = () => ({ load: jest.fn(async () => null), save: jest.fn(async () => true) });

    beforeEach(() => {
        jest.clearAllMocks();
        fredGraphCsv.mockRejectedValue(new Error('Fetch timed out for fredgraph after 4000ms'));
        fredObservations.mockRejectedValue(new Error('FRED API down'));
        treasuryYieldCurveCsv.mockResolvedValue(TREASURY);
        fetchSheetLkg.mockResolvedValue({
            indicators: { claims: { value: 203.25, asOf: '2026-01-03' }, sahmRule: { value: 0.03, asOf: '2025-12-01' } },
            checklist: { nfci: { value: -0.56, asOf: '2026-01-02' } },
            horsemen: { claims: { history: [{ date: '2026-01-03', value: 196000 }] } },
        });
    });

    test('t10y3m: FRED API dead → Treasury 10 Yr − 3 Mo (0.87), source treasury', async () => {
        const raw = {};
        const diag = {};
        const store = mkStore();
        const src = await repairPillInputs(raw, { fredKey: 'k', now: new Date('2026-01-07'), store, diag });
        expect(raw.t10y3m).toBe(0.87);
        expect(src.t10y3m).toBe('treasury');
        expect(diag.tried.t10y3m[0]).toMatch(/^fred:err\(FRED API down\)/);
        expect(treasuryYieldCurveCsv).toHaveBeenCalledWith(2026, expect.objectContaining({ timeout: 5000 }));
    });

    test('sibling missing entirely (timed out) → nfci/claims/sahm from the Sheet snapshot, fredcsv never reached', async () => {
        const raw = { fred: null };
        const diag = {};
        const src = await repairPillInputs(raw, { now: new Date('2026-01-07'), store: mkStore(), diag });
        expect(src).toMatchObject({ nfci: 'sheet', claims: 'sheet', sahm: 'sheet', t10y3m: 'treasury' });
        expect(raw.fred.checklist.nfci).toMatchObject({ value: -0.56, asOf: '2026-01-02', source: 'sheet' });
        expect(raw.fred.indicators.claims).toMatchObject({ value: 203.25, source: 'sheet' });
        expect(raw.fred.indicators.sahmRule).toMatchObject({ value: 0.03, source: 'sheet' });
        expect(fetchSheetLkg).toHaveBeenCalledTimes(1); // one snapshot shared by all three
        expect(diag.tried.claims).toEqual(['horsemen:empty', 'sheet:ok']);
    });

    test('hm_horsemen + hm_sheet faults skip those tiers; hm_fredcsv skips the phantom; lastgood serves', async () => {
        const store = mkStore();
        store.load.mockResolvedValue({ data: { value: 210, asOf: '2026-01-03', source: 'fred-route' }, savedAt: '2026-01-06T00:00:00Z' });
        const raw = { fred: { indicators: {}, horsemen: { claims: { history: [] } } } };
        const diag = {};
        const faults = new Set(['fred', 'hm_horsemen', 'hm_sheet', 'hm_fredcsv', 'hm_treasury', 'hm_fred']);
        const src = await repairPillInputs(raw, { fredKey: 'k', faults, now: new Date('2026-01-07'), store, diag });
        expect(diag.tried.claims).toEqual(['horsemen:off', 'sheet:off', 'fredcsv:off', 'lastgood:ok(2026-01-06T00:00:00Z)']);
        expect(src.claims).toBe('lastgood');
        expect(raw.fred.indicators.claims.value).toBe(210);
        expect(src.t10y3m).toBe('lastgood');
        expect(store.save).not.toHaveBeenCalled(); // fault mode never writes
    });

    test("the sibling's own `sheetlkg` fault name also disables the Sheet tier", async () => {
        const raw = { fred: null };
        const diag = {};
        await repairPillInputs(raw, { faults: new Set(['sheetlkg', 'lastgood']), now: new Date('2026-01-07'), store: mkStore(), diag });
        expect(diag.tried.nfci[0]).toMatch(/^sheet:err\(\[injected fault: sheetlkg\]\)/);
        expect(fetchSheetLkg).not.toHaveBeenCalled();
    });

    test('lastgood fault → null everywhere once live tiers are off', async () => {
        const store = mkStore();
        store.load.mockResolvedValue({ data: { value: 1, asOf: '2026-01-03' }, savedAt: '2026-01-06T00:00:00Z' });
        const raw = { fred: null };
        const faults = new Set(['hm_horsemen', 'hm_sheet', 'hm_fredcsv', 'hm_treasury', 'hm_fred', 'lastgood']);
        const src = await repairPillInputs(raw, { fredKey: 'k', faults, now: new Date('2026-01-07'), store });
        expect(src).toEqual({ t10y3m: null, nfci: null, claims: null, sahm: null });
        expect(store.load).not.toHaveBeenCalled();
    });

    test('healthy sibling → last-good SEEDED from its values (value + asOf, source fred-route)', async () => {
        const store = mkStore();
        fredObservations.mockResolvedValue([{ date: '2026-01-06', value: 0.9 }, { date: '2026-01-05', value: 0.85 }]);
        const raw = {
            fred: {
                indicators: { claims: { value: 203.25, asOf: '2026-01-03' }, sahmRule: { value: 0.03, asOf: '2025-12-01' } },
                checklist: { nfci: { value: -0.56, asOf: '2026-01-02' } },
            },
        };
        const diag = {};
        const src = await repairPillInputs(raw, { fredKey: 'k', now: new Date('2026-01-07'), store, diag });
        expect(src).toEqual({ t10y3m: 'fred', nfci: 'fred-route', claims: 'fred-route', sahm: 'fred-route' });
        expect(store.save).toHaveBeenCalledWith('jev-claims', { value: 203.25, asOf: '2026-01-03', source: 'fred-route' });
        expect(store.save).toHaveBeenCalledWith('jev-sahm', { value: 0.03, asOf: '2025-12-01', source: 'fred-route' });
        expect(store.save).toHaveBeenCalledWith('jev-nfci', { value: -0.56, asOf: '2026-01-02', source: 'fred-route' });
        expect(store.save).toHaveBeenCalledWith('jev-t10y3m', { value: 0.9, asOf: '2026-01-06', source: 'fred' });
        expect(diag.seeded.sort()).toEqual(['claims', 'nfci', 'sahm']);
        expect(fetchSheetLkg).not.toHaveBeenCalled(); // no miss → no Sheet fetch
    });

    test('seeding is skipped in fault mode', async () => {
        const store = mkStore();
        const raw = { fred: { indicators: { claims: { value: 1, asOf: 'x' }, sahmRule: { value: 1, asOf: 'x' } }, checklist: { nfci: { value: 1, asOf: 'x' } } } };
        await repairPillInputs(raw, { faults: new Set(['hm_fred', 'hm_treasury', 'hm_fredcsv', 'lastgood']), now: new Date('2026-01-07'), store });
        expect(store.save).not.toHaveBeenCalled();
    });
});

describe('assemblePills — dataAsOf covers every feed a pill reads', () => {
    test('spy through = last chartHistory date; fg through = today', () => {
        const raw = {
            ...sampleRaw,
            spy: { ...sampleRaw.spy, chartHistory: [{ date: '2026-09-17', price: 1 }, { date: '2026-09-18', price: 2 }] },
        };
        const p = assemblePills({ raw, jevAnswers: null, yesterday: null, mode: 'rules' });
        expect(p._meta.dataAsOf.spy).toBe('2026-09-18');
        expect(p._meta.dataAsOf.fg).toBe(p.asOf.slice(0, 10));
    });

    test('no chartHistory / no score → null, never a made-up date', () => {
        const raw = { ...sampleRaw, spy: { current: 1 }, fg: {} };
        const p = assemblePills({ raw, jevAnswers: null, yesterday: null, mode: 'rules' });
        expect(p._meta.dataAsOf.spy).toBeNull();
        expect(p._meta.dataAsOf.fg).toBeNull();
    });
});
