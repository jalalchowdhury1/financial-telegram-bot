import {
    PILLS,
    VERDICT_ORDER,
    JEV_QUESTIONS,
    buildState,
    ruleVerdicts,
    conflictPairs,
    mergeVerdicts,
    diffSinceYesterday,
    toData,
    pillFactors,
} from '../jevBrief';

// ---------------------------------------------------------------------------
// Helper — build a data-contract object with explicit values
// ---------------------------------------------------------------------------

function data({ overrides = {}, defaults = {} } = {}) {
    const d = {
        spy: { price: 500, chgPct: 0.5, ma200Pct: 5, high52Pct: -1, rsi: 55 },
        fg: { score: 50, rating: 'Neutral', prevWeek: 48, prevMonth: 45 },
        vol: {
            spy: { iv: 15, ivRank1y: 35, ivPctile1y: 35, rv21: 12, vrp: 4 },
            qqq: { iv: 20, ivRank1y: 40, ivPctile1y: 40, rv21: 16, vrp: 5 },
        },
        fred: {
            yieldCurve: 0.5, sahmRule: 0.1, claims: 220,
            creditSpread: 1.2, realYields: 1.5, copperGold: 1.4,
            sentiment: 70, nfci: -0.3,
        },
        t10y3m: 0.3,
        breadth: {
            rspSpy: { ratio: 1.02, chg20Pct: 1.5, chg60Pct: 3, vs50dPct: 0.5 },
            iwmSpy: { ratio: 0.95, chg20Pct: 2, chg60Pct: 4, vs50dPct: 1 },
            xlkXlu: { ratio: 1.1, chg20Pct: 0.8, chg60Pct: 2, vs50dPct: 0.3 },
            hygLqd: { ratio: 0.98, chg20Pct: 0.2, chg60Pct: 0.5, vs50dPct: 0.1 },
        },
        aaiiDiff: 5.5,
        ...defaults,
    };
    // Merge overrides top-level
    if (overrides.spy) Object.assign(d.spy, overrides.spy);
    if (overrides.fg) Object.assign(d.fg, overrides.fg);
    if (overrides.vol) {
        if (overrides.vol.spy) Object.assign(d.vol.spy, overrides.vol.spy);
        if (overrides.vol.qqq) Object.assign(d.vol.qqq, overrides.vol.qqq);
    }
    if (overrides.fred) Object.assign(d.fred, overrides.fred);
    if (overrides.breadth) {
        for (const k of Object.keys(overrides.breadth)) {
            if (d.breadth[k]) Object.assign(d.breadth[k], overrides.breadth[k]);
            else d.breadth[k] = overrides.breadth[k];
        }
    }
    if ('t10y3m' in overrides) d.t10y3m = overrides.t10y3m;
    if ('aaiiDiff' in overrides) d.aaiiDiff = overrides.aaiiDiff;
    return d;
}

// ---------------------------------------------------------------------------
// PILLS / constants
// ---------------------------------------------------------------------------

describe('PILLS and VERDICT_ORDER', () => {
    test('exports the five pills in order', () => {
        expect(PILLS).toEqual(['regime', 'recession', 'breadth', 'hedging', 'conflict']);
    });

    test('each pill has a verdict order', () => {
        for (const pill of PILLS) {
            expect(VERDICT_ORDER[pill]).toBeDefined();
            expect(Array.isArray(VERDICT_ORDER[pill])).toBe(true);
            expect(VERDICT_ORDER[pill].length).toBeGreaterThanOrEqual(2);
        }
    });

    test('JEV_QUESTIONS exports all five pills', () => {
        for (const pill of PILLS) {
            expect(JEV_QUESTIONS[pill]).toBeDefined();
            expect(JEV_QUESTIONS[pill].type).toBe('choice');
            expect(JEV_QUESTIONS[pill].criteria).toBeDefined();
        }
    });
});

// ---------------------------------------------------------------------------
// ruleVerdicts — regime
// ---------------------------------------------------------------------------

describe('ruleVerdicts — regime', () => {
    test('score >= 2 ⇒ risk-on', () => {
        // ma200Pct>0 (+1), fg>=50 (+1), hygLqd>0 (+1) = 3
        const d = data();
        expect(ruleVerdicts(d).regime.verdict).toBe('risk-on');
    });

    test('score <= -1 ⇒ risk-off', () => {
        // ma200Pct<0 (+1→no, -1), fg<30 (-1), hygLqd<-1 (-1) = -3
        const d = data({ overrides: { spy: { ma200Pct: -3 }, fg: { score: 25 }, breadth: { hygLqd: { chg20Pct: -2 } } } });
        expect(ruleVerdicts(d).regime.verdict).toBe('risk-off');
    });

    test('score 0 ⇒ neutral', () => {
        // ma200Pct>0 (+1), fg 30-49 (0), hygLqd -1 to 0 (0) = 1 → neutral (<2 and >-1)
        const d = data({ overrides: { fg: { score: 40 }, breadth: { hygLqd: { chg20Pct: -0.5 } } } });
        expect(ruleVerdicts(d).regime.verdict).toBe('neutral');
    });

    test('score 1 ⇒ neutral (border: below 2, above -1)', () => {
        // ma200Pct>0 (+1), fg 30-49 (0), hygLqd>0 (+1) = 2 → risk-on
        // For score 1: ma200Pct>0 (+1), fg 30-49 (0), hygLqd -1 to 0 (0) = 1
        const d = data({ overrides: { fg: { score: 40 }, breadth: { hygLqd: { chg20Pct: 0 } } } });
        expect(ruleVerdicts(d).regime.verdict).toBe('neutral');
    });

    test('missing inputs count as 0', () => {
        const d = data({ overrides: {
            spy: { ma200Pct: null },
            fg: { score: null },
            breadth: { hygLqd: { chg20Pct: null } },
        } });
        // All 0 → neutral
        expect(ruleVerdicts(d).regime.verdict).toBe('neutral');
    });

    test('regime includes a reason quoting numbers', () => {
        const r = ruleVerdicts(data());
        expect(r.regime.reason).toContain('Score');
    });
});

// ---------------------------------------------------------------------------
// ruleVerdicts — recession (edge thresholds)
// ---------------------------------------------------------------------------

describe('ruleVerdicts — recession', () => {
    test('sahmRule >= 0.5 ⇒ high', () => {
        const d = data({ overrides: { fred: { sahmRule: 0.5 } } });
        expect(ruleVerdicts(d).recession.verdict).toBe('high');
    });

    test('yieldCurve < 0 AND claims >= 260 ⇒ high', () => {
        const d = data({ overrides: { fred: { yieldCurve: -0.2, claims: 260 } } });
        expect(ruleVerdicts(d).recession.verdict).toBe('high');
    });

    test('sahmRule >= 0.2 ⇒ rising', () => {
        const d = data({ overrides: { fred: { sahmRule: 0.2 } } });
        expect(ruleVerdicts(d).recession.verdict).toBe('rising');
    });

    test('yieldCurve < 0 alone ⇒ rising', () => {
        const d = data({ overrides: { fred: { yieldCurve: -0.1, sahmRule: 0.15, claims: 250 } } });
        expect(ruleVerdicts(d).recession.verdict).toBe('rising');
    });

    test('claims >= 260 alone ⇒ rising', () => {
        const d = data({ overrides: { fred: { claims: 260, yieldCurve: 0.5, sahmRule: 0.15, nfci: -0.5 } } });
        expect(ruleVerdicts(d).recession.verdict).toBe('rising');
    });

    test('nfci > 0 alone ⇒ rising', () => {
        const d = data({ overrides: { fred: { nfci: 0.1, sahmRule: 0.15, yieldCurve: 0.5, claims: 250 } } });
        expect(ruleVerdicts(d).recession.verdict).toBe('rising');
    });

    test('sahmRule=0.19, yieldCurve>0, claims<260, nfci<=0 ⇒ low', () => {
        const d = data({ overrides: { fred: { sahmRule: 0.19, nfci: -0.1 } } });
        expect(ruleVerdicts(d).recession.verdict).toBe('low');
    });

    test('all recession inputs null ⇒ low', () => {
        const d = data({ overrides: {
            fred: { sahmRule: null, yieldCurve: null, claims: null, nfci: null },
        } });
        // nulls don't trigger any condition, so it should be "low"
        expect(ruleVerdicts(d).recession.verdict).toBe('low');
    });
});

// ---------------------------------------------------------------------------
// ruleVerdicts — breadth (edge thresholds)
// ---------------------------------------------------------------------------

describe('ruleVerdicts — breadth', () => {
    test('rspSpy.chg20Pct>0 and iwmSpy.chg20Pct>0 ⇒ broad', () => {
        const d = data();
        expect(ruleVerdicts(d).breadth.verdict).toBe('broad');
    });

    test('rspSpy.chg20Pct<-1.5 and vs50dPct<0 ⇒ rolling-over', () => {
        const d = data({ overrides: { breadth: { rspSpy: { chg20Pct: -2, vs50dPct: -1 } } } });
        expect(ruleVerdicts(d).breadth.verdict).toBe('rolling-over');
    });

    test('rspSpy<-1.5 but vs50dPct>=0 ⇒ narrow (not rolling-over)', () => {
        const d = data({ overrides: { breadth: { rspSpy: { chg20Pct: -2, vs50dPct: 0.1 } } } });
        expect(ruleVerdicts(d).breadth.verdict).toBe('narrow');
    });

    test('rspSpy chg20Pct <= 0 with iwmSpy > 0 ⇒ narrow', () => {
        const d = data({ overrides: { breadth: { rspSpy: { chg20Pct: -1 } } } });
        expect(ruleVerdicts(d).breadth.verdict).toBe('narrow');
    });

    test('no breadth data at all ⇒ narrow with reason', () => {
        const d = data({ overrides: {
            breadth: { rspSpy: { chg20Pct: null, vs50dPct: null }, iwmSpy: { chg20Pct: null } },
        } });
        const r = ruleVerdicts(d);
        expect(r.breadth.verdict).toBe('narrow');
        expect(r.breadth.reason).toContain('no breadth data');
    });
});

// ---------------------------------------------------------------------------
// ruleVerdicts — hedging (edge thresholds)
// ---------------------------------------------------------------------------

describe('ruleVerdicts — hedging', () => {
    test('ivPctile1y<20 and vrp<6 ⇒ cheap', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: 15, vrp: 4 } } } });
        expect(ruleVerdicts(d).hedging.verdict).toBe('cheap');
    });

    test('ivPctile>70 ⇒ expensive', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: 75 } } } });
        expect(ruleVerdicts(d).hedging.verdict).toBe('expensive');
    });

    test('vrp>10 ⇒ expensive (even if ivPctile is low)', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: 10, vrp: 12 } } } });
        expect(ruleVerdicts(d).hedging.verdict).toBe('expensive');
    });

    test('vrp=10 is NOT expensive (threshold is `>`), ivPctile=70 is NOT expensive', () => {
        // vrp=10: not >10. ivPctile=70: not >70. Both at thresholds → fair
        const d = data({ overrides: { vol: { spy: { ivPctile1y: 70, vrp: 10 } } } });
        expect(ruleVerdicts(d).hedging.verdict).toBe('fair');
    });

    test('ivPctile<20 but vrp>=6 ⇒ fair (cheap requires both)', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: 15, vrp: 7 } } } });
        expect(ruleVerdicts(d).hedging.verdict).toBe('fair');
    });

    test('missing vol data ⇒ fair with reason', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: null, vrp: null } } } });
        const r = ruleVerdicts(d);
        expect(r.hedging.verdict).toBe('fair');
        expect(r.hedging.reason).toContain('default');
    });
});

// ---------------------------------------------------------------------------
// ruleVerdicts — conflict
// ---------------------------------------------------------------------------

describe('ruleVerdicts — conflict', () => {
    test('0 pairs ⇒ aligned', () => {
        const d = data();
        expect(ruleVerdicts(d).conflict.verdict).toBe('aligned');
    });

    test('1 pair ⇒ mild-divergence', () => {
        // Force "sentiment vs price": fg<35 and ma200Pct>3
        const d = data({ overrides: { fg: { score: 30 }, spy: { ma200Pct: 5 } } });
        expect(conflictPairs(d).length).toBeGreaterThanOrEqual(1);
        // Ensure no other pairs fire
        // yieldCurve > 0, t10y3m > 0 → same sign, no conflict
        // ma200Pct>0 but hygLqd.chg20Pct=0.2 > -1.5 → no credit vs equities
        // high52Pct=-1 (>-2) but rspSpy.chg20Pct=1.5 > -1.5 → no breadth vs index
        expect(ruleVerdicts(d).conflict.verdict).toBe('mild-divergence');
    });

    test('2+ pairs ⇒ major-divergence', () => {
        // Force sentiment vs price + credit vs equities
        // sentiment vs price: fg<35 and ma200Pct>3
        // credit vs equities: ma200Pct>0 and hygLqd.chg20Pct<-1.5
        const d = data({ overrides: {
            fg: { score: 30 },
            spy: { ma200Pct: 5 },
            breadth: { hygLqd: { chg20Pct: -2 } },
        } });
        const cps = conflictPairs(d);
        expect(cps.length).toBeGreaterThanOrEqual(2);
        expect(ruleVerdicts(d).conflict.verdict).toBe('major-divergence');
    });
});

// ---------------------------------------------------------------------------
// conflictPairs — each pair fires independently
// ---------------------------------------------------------------------------

describe('conflictPairs — individual pairs', () => {
    test('sentiment vs price: fear + strong trend', () => {
        const d = data({ overrides: { fg: { score: 30 }, spy: { ma200Pct: 5 } } });
        const pairs = conflictPairs(d);
        expect(pairs.some((p) => p.pair === 'sentiment vs price')).toBe(true);
    });

    test('sentiment vs price: greed + weak trend', () => {
        const d = data({ overrides: { fg: { score: 70 }, spy: { ma200Pct: -5 } } });
        const pairs = conflictPairs(d);
        expect(pairs.some((p) => p.pair === 'sentiment vs price')).toBe(true);
    });

    test('2s10s vs 3m10y: opposite signs', () => {
        // yieldCurve > 0, t10y3m < 0
        const d = data({ overrides: { fred: { yieldCurve: 0.5 }, t10y3m: -0.2 } });
        const pairs = conflictPairs(d);
        expect(pairs.some((p) => p.pair === '2s10s vs 3m10y')).toBe(true);
    });

    test('2s10s vs 3m10y: same sign → no conflict', () => {
        const d = data({ overrides: { fred: { yieldCurve: 0.5 }, t10y3m: 0.3 } });
        const pairs = conflictPairs(d);
        expect(pairs.some((p) => p.pair === '2s10s vs 3m10y')).toBe(false);
    });

    test('credit vs equities: SPY above MA but credit rolling over', () => {
        const d = data({ overrides: { spy: { ma200Pct: 5 }, breadth: { hygLqd: { chg20Pct: -2 } } } });
        const pairs = conflictPairs(d);
        expect(pairs.some((p) => p.pair === 'credit vs equities')).toBe(true);
    });

    test('breadth vs index: near high but breadth rolling', () => {
        const d = data({ overrides: { spy: { high52Pct: -1 }, breadth: { rspSpy: { chg20Pct: -2 } } } });
        const pairs = conflictPairs(d);
        expect(pairs.some((p) => p.pair === 'breadth vs index')).toBe(true);
    });

    test('breadth vs index: near high but breadth fine → no conflict', () => {
        const d = data({ overrides: { spy: { high52Pct: -1 }, breadth: { rspSpy: { chg20Pct: 0.5 } } } });
        const pairs = conflictPairs(d);
        expect(pairs.some((p) => p.pair === 'breadth vs index')).toBe(false);
    });
});

// ---------------------------------------------------------------------------
// mergeVerdicts
// ---------------------------------------------------------------------------

describe('mergeVerdicts', () => {
    const sampleRule = ruleVerdicts(data());

    test('jev null ⇒ rule verdict', () => {
        const merged = mergeVerdicts(sampleRule, null);
        for (const pill of PILLS) {
            expect(merged[pill].by).toBe('rule');
            expect(merged[pill].verdict).toBe(sampleRule[pill].verdict);
        }
    });

    test('jev with p >= 0.6 overrides rule', () => {
        const jev = { regime: { verdict: 'risk-off', p: 0.8 } };
        const merged = mergeVerdicts(sampleRule, jev);
        expect(merged.regime.by).toBe('jev');
        expect(merged.regime.verdict).toBe('risk-off');
        expect(merged.regime.p).toBe(0.8);
    });

    test('jev with p < 0.6 does not override', () => {
        const jev = { regime: { verdict: 'risk-off', p: 0.5 } };
        const merged = mergeVerdicts(sampleRule, jev);
        expect(merged.regime.by).toBe('rule');
        expect(merged.regime.verdict).toBe(sampleRule.regime.verdict);
    });

    test('jev with unknown verdict does not override', () => {
        const jev = { regime: { verdict: 'super-risk-on', p: 0.9 } };
        const merged = mergeVerdicts(sampleRule, jev);
        expect(merged.regime.by).toBe('rule');
    });

    test('jev with missing p does not override', () => {
        const jev = { regime: { verdict: 'risk-off' } };
        const merged = mergeVerdicts(sampleRule, jev);
        expect(merged.regime.by).toBe('rule');
    });

    test('jev with p=0.6 exactly meets floor', () => {
        const jev = { regime: { verdict: 'risk-off', p: 0.6 } };
        const merged = mergeVerdicts(sampleRule, jev, 0.6);
        expect(merged.regime.by).toBe('jev');
    });
});

// ---------------------------------------------------------------------------
// diffSinceYesterday
// ---------------------------------------------------------------------------

describe('diffSinceYesterday', () => {
    test('no yesterday ⇒ noBaseline: true', () => {
        const diff = diffSinceYesterday({}, null);
        expect(diff.direction).toBe('none');
        expect(diff.noBaseline).toBe(true);
        expect(diff.changed).toEqual([]);
    });

    test('identical today/yesterday ⇒ none', () => {
        const today = { regime: { verdict: 'risk-on' }, recession: { verdict: 'low' } };
        const yesterday = { regime: { verdict: 'risk-on' }, recession: { verdict: 'low' } };
        const diff = diffSinceYesterday(today, yesterday);
        expect(diff.direction).toBe('none');
        expect(diff.changed).toEqual([]);
    });

    test('both moves toward severe ⇒ hardening', () => {
        // regime: risk-on → risk-off (more severe). recession: low → high (more severe)
        const today = { regime: { verdict: 'risk-off' }, recession: { verdict: 'high' } };
        const yesterday = { regime: { verdict: 'risk-on' }, recession: { verdict: 'low' } };
        const diff = diffSinceYesterday(today, yesterday);
        expect(diff.direction).toBe('hardening');
        expect(diff.changed.length).toBe(2);
    });

    test('both moves toward mild ⇒ softening', () => {
        // regime: risk-off → risk-on (less severe). recession: high → low (less severe)
        const today = { regime: { verdict: 'risk-on' }, recession: { verdict: 'low' } };
        const yesterday = { regime: { verdict: 'risk-off' }, recession: { verdict: 'high' } };
        const diff = diffSinceYesterday(today, yesterday);
        expect(diff.direction).toBe('softening');
    });

    test('mixed moves ⇒ mixed', () => {
        // regime toward severe: risk-on → risk-off. recession toward mild: high → low
        const today = { regime: { verdict: 'risk-off' }, recession: { verdict: 'low' } };
        const yesterday = { regime: { verdict: 'risk-on' }, recession: { verdict: 'high' } };
        const diff = diffSinceYesterday(today, yesterday);
        expect(diff.direction).toBe('mixed');
    });

    test('reports changed pills with from/to', () => {
        const today = { regime: { verdict: 'risk-off' } };
        const yesterday = { regime: { verdict: 'risk-on' } };
        const diff = diffSinceYesterday(today, yesterday);
        expect(diff.changed[0]).toEqual({ pill: 'regime', from: 'risk-on', to: 'risk-off' });
    });
});

// ---------------------------------------------------------------------------
// buildState — privacy constraint
// ---------------------------------------------------------------------------

describe('buildState', () => {
    test('builds a multiline string with numbered facts', () => {
        const d = data();
        const state = buildState(d);
        const lines = state.split('\n').filter(Boolean);
        expect(lines.length).toBeGreaterThanOrEqual(10);
        expect(lines[0]).toContain('SPY');
        expect(lines[1]).toContain('Fear & Greed');
    });

    test('never contains "NotSoBoring" or "FrontRunner"', () => {
        const d = data();
        const state = buildState(d);
        expect(state).not.toContain('NotSoBoring');
        expect(state).not.toContain('FrontRunner');
    });

    test('handles null/undefined data gracefully', () => {
        expect(buildState(null)).toBe('');
        expect(buildState(undefined)).toBe('');
    });

    test('uses n/a for missing values', () => {
        const d = data({ overrides: {
            spy: { price: null, chgPct: null, ma200Pct: null, high52Pct: null, rsi: null },
            fg: { score: null, rating: null, prevWeek: null, prevMonth: null },
        } });
        const state = buildState(d);
        expect(state).toContain('n/a');
    });

    test('handles fg.score "N/A" string and aaiiDiff NaN without throwing or printing "NaN"', () => {
        const d = data({ overrides: {
            fg: { score: 'N/A' },
            aaiiDiff: NaN,
        } });
        const state = buildState(d);
        expect(state).not.toContain('NaN');
    });
});

// ---------------------------------------------------------------------------
// toData — normalise raw payloads
// ---------------------------------------------------------------------------

describe('toData', () => {
    test('extracts spy from /api/spy shape (current + dailyChange + ma200 + week52High)', () => {
        const raw = {
            spy: {
                current: 500,
                dailyChange: { pct: 0.5 },
                ma200: { pct: 5 },
                week52High: { pct: -1 },
                rsi: 55,
            },
        };
        const d = toData(raw);
        expect(d.spy.price).toBe(500);
        expect(d.spy.chgPct).toBe(0.5);
        expect(d.spy.ma200Pct).toBe(5);
        expect(d.spy.high52Pct).toBe(-1);
        expect(d.spy.rsi).toBe(55);
    });

    test('extracts spy from flat shape as well', () => {
        const raw = {
            spy: { price: 500, chgPct: 0.5, ma200Pct: 5, high52Pct: -1, rsi: 55 },
        };
        const d = toData(raw);
        expect(d.spy.price).toBe(500);
    });

    test('extracts fear/greed with previousWeek/previousMonth', () => {
        const raw = {
            fg: { score: 25, rating: 'Fear', previousWeek: 30, previousMonth: 40 },
        };
        const d = toData(raw);
        expect(d.fg.score).toBe(25);
        expect(d.fg.rating).toBe('Fear');
        expect(d.fg.prevWeek).toBe(30);
        expect(d.fg.prevMonth).toBe(40);
    });

    test('extracts vol from tickers array', () => {
        const raw = {
            vol: {
                tickers: [
                    { ticker: 'spy', iv: 15, ivRank1y: 35, ivPctile1y: 35, rv21: 12, vrp: 4 },
                    { ticker: 'qqq', iv: 20, ivRank1y: 40, ivPctile1y: 40, rv21: 16, vrp: 5 },
                ],
            },
        };
        const d = toData(raw);
        expect(d.vol.spy.iv).toBe(15);
        expect(d.vol.qqq.iv).toBe(20);
    });

    test('lowercases UPPERCASE ticker keys (real /api/vol shape)', () => {
        const raw = {
            vol: {
                tickers: [
                    { ticker: 'SPY', iv: 15, ivRank1y: 35, ivPctile1y: 35, rv21: 12, vrp: 4 },
                    { ticker: 'QQQ', iv: 20, ivRank1y: 40, ivPctile1y: 40, rv21: 16, vrp: 5 },
                ],
            },
        };
        const d = toData(raw);
        expect(d.vol.spy.ivPctile1y).toBe(35);
        expect(d.vol.qqq.iv).toBe(20);
    });

    test('extracts fred nested indicators', () => {
        const raw = {
            fred: {
                yieldCurve: { value: 0.5 },
                indicators: {
                    sahmRule: { value: 0.1 },
                    claims: { value: 220 },
                    creditSpread: { value: 1.2 },
                    realYields: { value: 1.5 },
                    copperGold: { value: 1.4 },
                    sentiment: { value: 70 },
                },
                checklist: { nfci: { value: -0.3 } },
            },
        };
        const d = toData(raw);
        expect(d.fred.yieldCurve).toBe(0.5);
        expect(d.fred.sahmRule).toBe(0.1);
        expect(d.fred.nfci).toBe(-0.3);
    });

    test('extracts breadth from pairs', () => {
        const raw = {
            breadth: {
                pairs: {
                    rspSpy: { ratio: 1.02, chg20Pct: 1.5, chg60Pct: 3, vs50dPct: 0.5 },
                    iwmSpy: { ratio: 0.95, chg20Pct: 2, chg60Pct: 4, vs50dPct: 1 },
                },
            },
        };
        const d = toData(raw);
        expect(d.breadth.rspSpy.ratio).toBe(1.02);
        expect(d.breadth.iwmSpy.chg20Pct).toBe(2);
    });

    test('parses aaII diff from string "%" format', () => {
        const raw = { sheets: { AAIIDiff: '24.50%' } };
        expect(toData(raw).aaiiDiff).toBe(24.5);
    });

    test('parses aaII diff from number format', () => {
        const raw = { sheets: { AAIIDiff: 24.5 } };
        expect(toData(raw).aaiiDiff).toBe(24.5);
    });

    test('t10y3m passes through', () => {
        const raw = { t10y3m: -0.2 };
        expect(toData(raw).t10y3m).toBe(-0.2);
    });

    test('null values in raw become null in output', () => {
        const raw = { spy: { current: null }, fg: null, fred: null, breadth: null };
        const d = toData(raw);
        expect(d.spy.price).toBeNull();
        expect(d.fg.score).toBeNull();
        expect(d.fred.yieldCurve).toBeNull();
        // breadth missing entirely → empty object, chained lookups fine
        expect(d.breadth.rspSpy).toBeUndefined();
    });

    test('returns null when raw is null/undefined', () => {
        expect(toData(null)).toBeNull();
        expect(toData(undefined)).toBeNull();
    });

    test('num helper: fg.score "N/A" gives null', () => {
        const raw = { fg: { score: 'N/A' } };
        expect(toData(raw).fg.score).toBeNull();
    });

    test('num helper: AAIIDiff "N/A" gives null', () => {
        const raw = { sheets: { AAIIDiff: 'N/A' } };
        expect(toData(raw).aaiiDiff).toBeNull();
    });

    test('num helper: AAIIDiff "24.50%" still gives 24.5', () => {
        const raw = { sheets: { AAIIDiff: '24.50%' } };
        expect(toData(raw).aaiiDiff).toBe(24.5);
    });
});

// ---------------------------------------------------------------------------
// pillFactors — shape and content
// ---------------------------------------------------------------------------

describe('pillFactors', () => {
    test('returns all five pills with rows and summary', () => {
        const d = data();
        const pf = pillFactors(d);

        for (const pill of PILLS) {
            expect(pf[pill]).toBeDefined();
            expect(pf[pill]).toHaveProperty('summary');
            expect(pf[pill]).toHaveProperty('rows');
            expect(Array.isArray(pf[pill].rows)).toBe(true);
        }
    });

    test('each row has the correct shape', () => {
        const pf = pillFactors(data());
        for (const pill of PILLS) {
            for (const row of pf[pill].rows) {
                expect(row).toHaveProperty('label');
                expect(row).toHaveProperty('value');
                expect(row).toHaveProperty('test');
                expect(row).toHaveProperty('hit');
                expect(row).toHaveProperty('effect');
                expect(typeof row.label).toBe('string');
                expect(typeof row.value).toBe('string');
                expect(typeof row.test).toBe('string');
                expect(typeof row.hit).toBe('boolean');
                expect(typeof row.effect).toBe('string');
            }
        }
    });

    test('regime rows in order: SPY vs 200d, F&G, HYG/LQD', () => {
        const pf = pillFactors(data());
        const rows = pf.regime.rows;
        expect(rows.length).toBe(3);
        expect(rows[0].label).toContain('SPY vs 200-day');
        expect(rows[1].label).toContain('Fear & Greed');
        expect(rows[2].label).toContain('HYG/LQD');
    });

    test('regime summary quotes score', () => {
        const pf = pillFactors(data());
        expect(pf.regime.summary).toMatch(/Score \d/);
    });

    test('regime hit flags on risk-on data', () => {
        // Default data: ma200=+5%>0, fg=50≥50, hyg=0.2>0 → all three hit → score 3 → risk-on
        const pf = pillFactors(data());
        const rows = pf.regime.rows;
        expect(rows[0].hit).toBe(true);
        expect(rows[1].hit).toBe(true);
        expect(rows[2].hit).toBe(true);
        expect(pf.regime.summary).toContain('risk-on');
    });

    test('regime hit flags on risk-off data', () => {
        // ma200=-3%<0, fg=25<30, hyg=-2<-1 → all hit → score -3 → risk-off
        const d = data({ overrides: { spy: { ma200Pct: -3 }, fg: { score: 25 }, breadth: { hygLqd: { chg20Pct: -2 } } } });
        const pf = pillFactors(d);
        const rows = pf.regime.rows;
        expect(rows[0].hit).toBe(true);
        expect(rows[0].effect).toBe('−1');
        expect(rows[1].hit).toBe(true);
        expect(rows[1].effect).toBe('−1');
        expect(rows[2].hit).toBe(true);
        expect(rows[2].effect).toBe('−1');
        expect(pf.regime.summary).toContain('risk-off');
    });

    test('regime: fg=40 (30-49) does not hit', () => {
        const d = data({ overrides: { fg: { score: 40 } } });
        const pf = pillFactors(d);
        expect(pf.regime.rows[1].hit).toBe(false);
        expect(pf.regime.rows[1].effect).toBe('0');
    });

    test('regime: hygChg20 between -1 and 0 does not hit', () => {
        const d = data({ overrides: { breadth: { hygLqd: { chg20Pct: -0.5 } } } });
        const pf = pillFactors(d);
        expect(pf.regime.rows[2].hit).toBe(false);
        expect(pf.regime.rows[2].effect).toBe('0');
    });

    test('recession has 4 rows, one per input, no repeated input', () => {
        const pf = pillFactors(data());
        const labels = pf.recession.rows.map((r) => r.label);
        expect(labels).toEqual(['Sahm rule', 'Yield curve (2s10s)', 'Jobless claims', 'NFCI']);
        expect(new Set(labels).size).toBe(labels.length);
    });

    test('recession: low verdict summary', () => {
        const pf = pillFactors(data());
        expect(pf.recession.summary).toContain('none tripped → low');
        expect(pf.recession.rows.every((r) => !r.hit)).toBe(true);
    });

    test('recession: sahm >= 0.5 → Sahm row fires high', () => {
        const pf = pillFactors(data({ overrides: { fred: { sahmRule: 0.5 } } }));
        expect(pf.recession.rows[0]).toMatchObject({ hit: true, effect: 'high' });
        expect(pf.recession.summary).toBe('Sahm rule tripped → high');
    });

    test('recession: sahm 0.2–0.49 → Sahm row fires rising', () => {
        const pf = pillFactors(data({ overrides: { fred: { sahmRule: 0.2 } } }));
        expect(pf.recession.rows[0]).toMatchObject({ hit: true, effect: 'rising' });
    });

    test('recession: inverted curve + claims >= 260k → both rows fire high', () => {
        const pf = pillFactors(data({ overrides: { fred: { yieldCurve: -0.2, claims: 260 } } }));
        expect(pf.recession.rows[1]).toMatchObject({ hit: true, effect: 'high' });
        expect(pf.recession.rows[2]).toMatchObject({ hit: true, effect: 'high' });
        expect(pf.recession.summary).toContain('→ high');
    });

    test('recession: inverted curve alone → rising; claims alone → rising', () => {
        const yc = pillFactors(data({ overrides: { fred: { yieldCurve: -0.1 } } }));
        expect(yc.recession.rows[1]).toMatchObject({ hit: true, effect: 'rising' });
        expect(yc.recession.rows[2].hit).toBe(false);
        const cl = pillFactors(data({ overrides: { fred: { claims: 270 } } }));
        expect(cl.recession.rows[2]).toMatchObject({ hit: true, effect: 'rising' });
        expect(cl.recession.rows[1].hit).toBe(false);
    });

    test('breadth has 3 rows', () => {
        const pf = pillFactors(data());
        expect(pf.breadth.rows.length).toBe(3);
        expect(pf.breadth.rows[0].label).toContain('RSP/SPY 20d');
        expect(pf.breadth.rows[1].label).toContain('RSP/SPY vs 50d');
        expect(pf.breadth.rows[2].label).toContain('IWM/SPY 20d');
    });

    test('breadth: rolling-over data', () => {
        const d = data({ overrides: { breadth: { rspSpy: { chg20Pct: -2, vs50dPct: -1 } } } });
        const pf = pillFactors(d);
        expect(pf.breadth.rows[0].hit).toBe(true);
        expect(pf.breadth.rows[0].effect).toBe('rolling-over');
        expect(pf.breadth.rows[1].hit).toBe(true);
        expect(pf.breadth.rows[2].hit).toBe(false);
        expect(pf.breadth.summary).toContain('Rolling over');
    });

    test('breadth: broad data', () => {
        const d = data();  // default: rsp=1.5>0, iwm=2>0
        const pf = pillFactors(d);
        expect(pf.breadth.rows[0].hit).toBe(true);
        expect(pf.breadth.rows[0].effect).toBe('broad');
        expect(pf.breadth.rows[1].hit).toBe(false);
        expect(pf.breadth.rows[2].hit).toBe(true);
        expect(pf.breadth.rows[2].effect).toBe('broad');
        expect(pf.breadth.summary).toContain('Broad');
    });

    test('breadth: narrow data', () => {
        const d = data({ overrides: { breadth: { rspSpy: { chg20Pct: -1 } } } });
        const pf = pillFactors(d);
        expect(pf.breadth.rows[0].hit).toBe(false);
        expect(pf.breadth.rows[1].hit).toBe(false);
        expect(pf.breadth.rows[2].hit).toBe(false);
        expect(pf.breadth.summary).toContain('Narrow');
    });

    test('hedging has 2 rows', () => {
        const pf = pillFactors(data());
        expect(pf.hedging.rows.length).toBe(2);
        expect(pf.hedging.rows[0].label).toContain('IV percentile');
        expect(pf.hedging.rows[1].label).toBe('VRP');
    });

    test('hedging: cheap data', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: 15, vrp: 4 } } } });
        const pf = pillFactors(d);
        expect(pf.hedging.rows[0].hit).toBe(true);
        expect(pf.hedging.rows[0].effect).toBe('cheap');
        expect(pf.hedging.rows[1].hit).toBe(true);
        expect(pf.hedging.rows[1].effect).toBe('cheap');
        expect(pf.hedging.summary).toContain('cheap');
    });

    test('hedging: expensive via ivPctile only', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: 75, vrp: 4 } } } });
        const pf = pillFactors(d);
        expect(pf.hedging.rows[0].hit).toBe(true);
        expect(pf.hedging.rows[0].effect).toBe('expensive');
        expect(pf.hedging.rows[1].hit).toBe(false);  // vrp=4 NOT >10, NOT <6 with iv<20 (iv=75)
        expect(pf.hedging.rows[1].effect).toBe('');
        expect(pf.hedging.summary).toContain('expensive');
    });

    test('hedging: expensive via vrp only', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: 30, vrp: 12 } } } });
        const pf = pillFactors(d);
        expect(pf.hedging.rows[0].hit).toBe(false);  // iv 30 not >70 and not <20
        expect(pf.hedging.rows[1].hit).toBe(true);
        expect(pf.hedging.rows[1].effect).toBe('expensive');
        expect(pf.hedging.summary).toContain('expensive');
    });

    test('conflict has exactly 4 rows', () => {
        const pf = pillFactors(data());
        expect(pf.conflict.rows.length).toBe(4);
        expect(pf.conflict.rows[0].label).toBe('sentiment vs price');
        expect(pf.conflict.rows[1].label).toBe('2s10s vs 3m10y');
        expect(pf.conflict.rows[2].label).toBe('credit vs equities');
        expect(pf.conflict.rows[3].label).toBe('breadth vs index');
    });

    test('conflict effect is divergence when hit', () => {
        // Force sentiment vs price
        const d = data({ overrides: { fg: { score: 30 }, spy: { ma200Pct: 5 } } });
        const pf = pillFactors(d);
        const sRow = pf.conflict.rows.find((r) => r.label === 'sentiment vs price');
        expect(sRow.hit).toBe(true);
        expect(sRow.effect).toBe('divergence');
    });

    test('conflict effect is empty when not hit', () => {
        const pf = pillFactors(data());
        for (const row of pf.conflict.rows) {
            expect(row.hit).toBe(false);
            expect(row.effect).toBe('');
        }
        expect(pf.conflict.summary).toContain('0 of 4');
        expect(pf.conflict.summary).toContain('aligned');
    });

    test('null data returns all pills with no-data summary', () => {
        const pf = pillFactors(null);
        for (const pill of PILLS) {
            expect(pf[pill]).toBeDefined();
            expect(pf[pill].summary).toBe('no data');
            expect(pf[pill].rows).toEqual([]);
        }
    });

    test('formatting: percentages with sign and one decimal', () => {
        const pf = pillFactors(data());
        // Default spy ma200Pct=5 → "+5.0%" 
        const spyRow = pf.regime.rows[0];
        expect(spyRow.value).toMatch(/^[+-]\d+\.\d%/);
    });

    test('formatting: Sahm with two decimals', () => {
        const d = data({ overrides: { fred: { sahmRule: 0.5 } } });
        const pf = pillFactors(d);
        // Sahm row is first in recession
        expect(pf.recession.rows[0].value).toBe('0.50');
    });

    test('formatting: claims as 203k', () => {
        const d = data({ overrides: { fred: { claims: 203 } } });
        const pf = pillFactors(d);
        const claimsRow = pf.recession.rows[2];
        expect(claimsRow.value).toBe('203k');
    });

    test('formatting: F&G as integer', () => {
        const pf = pillFactors(data());
        expect(pf.regime.rows[1].value).toBe('50');
    });
});

// ---------------------------------------------------------------------------
// Consistency test — rule verdicts match implied verdict from factor hits
// ---------------------------------------------------------------------------

describe('pillFactors — consistency with ruleVerdicts', () => {
    function impliedVerdict(factorRows) {
        // Scan rows in order; first hit determines verdict via its effect
        for (const row of factorRows) {
            if (row.hit) {
                const e = row.effect;
                if (e === 'high' || e === 'rising' || e === 'rolling-over' ||
                    e === 'broad' || e === 'expensive' || e === 'cheap' ||
                    e === '+1' || e === '−1' || e === 'divergence') {
                    // For regime, effect is '+1' or '−1' — need score approach
                    // For non-regime pills, effect IS the verdict
                    // Return effect and let the caller interpret
                }
            }
        }
        return 'unknown';
    }

    function regimeVerdictFromHits(rows) {
        // Score from hits: +1 if hit and effect='+1', -1 if hit and effect='−1'
        let score = 0;
        for (const row of rows) {
            if (row.hit) {
                if (row.effect === '+1') score += 1;
                else if (row.effect === '−1') score -= 1;
            }
        }
        if (score >= 2) return 'risk-on';
        if (score <= -1) return 'risk-off';
        return 'neutral';
    }

    function recessionVerdictFromHits(rows) {
        const fired = rows.filter((r) => r.hit);
        if (fired.some((r) => r.effect === 'high')) return 'high';
        return fired.length ? 'rising' : 'low';
    }

    function breadthVerdictFromHits(rows) {
        for (const row of rows) {
            if (row.hit && (row.effect === 'rolling-over' || row.effect === 'broad')) {
                return row.effect;
            }
        }
        return 'narrow';
    }

    function hedgingVerdictFromHits(rows) {
        // expensive if EITHER row says expensive, cheap if any says cheap and none says expensive
        let hasExpensive = false;
        let hasCheap = false;
        for (const row of rows) {
            if (row.hit) {
                if (row.effect === 'expensive') hasExpensive = true;
                if (row.effect === 'cheap') hasCheap = true;
            }
        }
        if (hasExpensive) return 'expensive';
        if (hasCheap) return 'cheap';
        return 'fair';
    }

    function conflictVerdictFromHits(rows) {
        const numDivergence = rows.filter((r) => r.hit && r.effect === 'divergence').length;
        if (numDivergence === 0) return 'aligned';
        if (numDivergence === 1) return 'mild-divergence';
        return 'major-divergence';
    }

    function verdictFromFactors(pf) {
        return {
            regime: regimeVerdictFromHits(pf.regime.rows),
            recession: recessionVerdictFromHits(pf.recession.rows),
            breadth: breadthVerdictFromHits(pf.breadth.rows),
            hedging: hedgingVerdictFromHits(pf.hedging.rows),
            conflict: conflictVerdictFromHits(pf.conflict.rows),
        };
    }

    // Battery of data objects
    const fixtures = [
        { name: 'default data', data: data() },
        { name: 'regime risk-off', data: data({ overrides: { spy: { ma200Pct: -3 }, fg: { score: 25 }, breadth: { hygLqd: { chg20Pct: -2 } } } }) },
        { name: 'recession high via sahm', data: data({ overrides: { fred: { sahmRule: 0.5 } } }) },
        { name: 'recession high via yc+claims', data: data({ overrides: { fred: { yieldCurve: -0.2, claims: 260 } } }) },
        { name: 'recession rising via sahm', data: data({ overrides: { fred: { sahmRule: 0.2 } } }) },
        { name: 'recession rising via yc', data: data({ overrides: { fred: { yieldCurve: -0.1, sahmRule: 0.15 } } }) },
        { name: 'recession rising via nfci', data: data({ overrides: { fred: { nfci: 0.1, sahmRule: 0.15, yieldCurve: 0.5, claims: 250 } } }) },
        { name: 'broad breadth', data: data() },
        { name: 'rolling-over breadth', data: data({ overrides: { breadth: { rspSpy: { chg20Pct: -2, vs50dPct: -1 } } } }) },
        { name: 'narrow breadth', data: data({ overrides: { breadth: { rspSpy: { chg20Pct: -1 } } } }) },
        { name: 'cheap hedging', data: data({ overrides: { vol: { spy: { ivPctile1y: 15, vrp: 4 } } } }) },
        { name: 'expensive hedging via iv', data: data({ overrides: { vol: { spy: { ivPctile1y: 75, vrp: 4 } } } }) },
        { name: 'expensive hedging via vrp', data: data({ overrides: { vol: { spy: { ivPctile1y: 30, vrp: 12 } } } }) },
        { name: 'fair hedging thresholds', data: data({ overrides: { vol: { spy: { ivPctile1y: 70, vrp: 10 } } } }) },
        { name: 'conflict 1 pair', data: data({ overrides: { fg: { score: 30 }, spy: { ma200Pct: 5 } } }) },
        { name: 'conflict 2+ pairs', data: data({ overrides: { fg: { score: 30 }, spy: { ma200Pct: 5 }, breadth: { hygLqd: { chg20Pct: -2 } } } }) },
        { name: 'all null', data: data({ overrides: {
            spy: { ma200Pct: null, high52Pct: null },
            fg: { score: null },
            fred: { sahmRule: null, yieldCurve: null, claims: null, nfci: null },
            vol: { spy: { ivPctile1y: null, vrp: null } },
            breadth: { rspSpy: { chg20Pct: null, vs50dPct: null }, iwmSpy: { chg20Pct: null }, hygLqd: { chg20Pct: null } },
        } }) },
    ];

    for (const { name, data: d } of fixtures) {
        test(`consistency: ${name}`, () => {
            const rule = ruleVerdicts(d);
            const pf = pillFactors(d);
            const implied = verdictFromFactors(pf);

            for (const pill of PILLS) {
                if (rule[pill] && rule[pill].verdict && rule[pill].verdict !== 'n/a') {
                    expect(implied[pill]).toBe(rule[pill].verdict);
                }
            }
        });
    }

    test('consistency: all-null regime is neutral (score 0)', () => {
        const d = data({ overrides: {
            spy: { ma200Pct: null },
            fg: { score: null },
            breadth: { hygLqd: { chg20Pct: null } },
        } });
        const rule = ruleVerdicts(d);
        const pf = pillFactors(d);
        const implied = verdictFromFactors(pf);
        expect(implied.regime).toBe(rule.regime.verdict);
    });
});

describe('percent formatting never prints negative zero', () => {
    test('HYG/LQD 20d of -0.004% shows as 0.00% in factors and in the regime reason', () => {
        const d = data({ overrides: { breadth: { hygLqd: { chg20Pct: -0.004 } } } });
        expect(pillFactors(d).regime.rows[2].value).toBe('0.00%');
        expect(ruleVerdicts(d).regime.reason).toContain('hygLqd=0.00%');
        expect(ruleVerdicts(d).regime.reason).not.toContain('-0.00%');
    });
});

describe('HYG/LQD display', () => {
    test('two decimals and the legs line when legs are present', () => {
        const d = data({ overrides: { breadth: { hygLqd: { ratio: 0.75, chg20Pct: -0.0125, chg60Pct: 2.77, vs50dPct: 0.26, legs: { HYG: -1.29, LQD: -1.28 } } } } });
        const row = pillFactors(d).regime.rows.find((r) => r.label === 'HYG/LQD 20d');
        expect(row.value).toBe('-0.01%');
        expect(row.note).toBe('HYG -1.3% · LQD -1.3%');
        expect(row.hit).toBe(false);
        expect(ruleVerdicts(d).regime.reason).toContain('hygLqd=-0.01% (-1 to 0, 0)');
        expect(ruleVerdicts(d).regime.reason).toContain('legs 20d: HYG -1.3% · LQD -1.3%');
    });
    test('no legs → plain two-decimal value, no legs line', () => {
        const d = data({ overrides: { breadth: { hygLqd: { ratio: 0.75, chg20Pct: 0.2 } } } });
        const row = pillFactors(d).regime.rows.find((r) => r.label === 'HYG/LQD 20d');
        expect(row.value).toBe('+0.20%');
        expect(row.note).toBeUndefined();
        expect(ruleVerdicts(d).regime.reason).not.toContain('legs 20d');
    });
});

// ---------------------------------------------------------------------------
// Null-input edges flagged by the 2026-09-19 review
// ---------------------------------------------------------------------------

describe('null-input edges', () => {
    test('hedging: ivPctile>70 with vrp=null is still expensive', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: 75, vrp: null } } } });
        expect(ruleVerdicts(d).hedging.verdict).toBe('expensive');
    });

    test('hedging: ivPctile=null with vrp>10 is still expensive', () => {
        const d = data({ overrides: { vol: { spy: { ivPctile1y: null, vrp: 12 } } } });
        expect(ruleVerdicts(d).hedging.verdict).toBe('expensive');
    });

    test('2s10s vs 3m10y: t10y3m=null never fires, even with an inverted 2s10s', () => {
        const d = data({ overrides: { fred: { yieldCurve: -0.5 }, t10y3m: null } });
        expect(conflictPairs(d).some((p) => p.pair === '2s10s vs 3m10y')).toBe(false);
    });
});
