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