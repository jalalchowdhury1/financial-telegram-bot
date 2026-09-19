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