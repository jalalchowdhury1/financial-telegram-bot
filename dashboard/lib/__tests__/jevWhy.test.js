import { pillWhy, WHY_LABELS } from '../jevWhy';
import { PILLS, pillFactors } from '../jevBrief';
import live from './fixtures/jev-pills-2026-10-04.json';

// Live /api/jev-pills saved 2026-10-04 (public market numbers only).
describe('pillWhy — live payload', () => {
    test('regime: the votes that fired', () => {
        expect(pillWhy(live, 'regime')).toBe('2 of 3 votes: uptrend + junk bonds beating safe bonds');
    });
    test('recession: how many warnings tripped', () => {
        expect(pillWhy(live, 'recession')).toBe('0 of 4 warnings tripped');
    });
    test('breadth: the fired rows, with their real values', () => {
        expect(pillWhy(live, 'breadth')).toBe('Average stock −4.3% vs SPY in 20 days · below its 50-day trend');
    });
    test('hedging: where option prices sit in their year', () => {
        expect(pillWhy(live, 'hedging')).toBe('Options cheap: bottom 15% of the year');
    });
    test('conflict: the pairs that disagree', () => {
        expect(pillWhy(live, 'conflict')).toBe('Fearful crowd in an uptrend · near the high, average stock slipping');
    });
});

// Feed real-shaped inputs through the SAME pillFactors the route uses, so a label
// rename or a new row in lib/jevBrief.js fails here instead of silently dropping the line.
function data(over = {}) {
    return {
        spy: { ma200Pct: 5, high52Pct: -1, ...over.spy },
        fg: { score: 50, ...over.fg },
        vol: { spy: { ivPctile1y: 35, vrp: 4, ...over.vol } },
        fred: { yieldCurve: 0.5, sahmRule: 0.1, claims: 220, nfci: -0.3, ...over.fred },
        t10y3m: 'ten' in over ? over.ten : 0.3,
        breadth: {
            rspSpy: { chg20Pct: 1.5, vs50dPct: 0.5, ...over.rsp },
            iwmSpy: { chg20Pct: 2, ...over.iwm },
            hygLqd: { chg20Pct: 0.2, ...over.hyg },
        },
    };
}
const why = (d, pill) => pillWhy({ factors: pillFactors(d), pills: {} }, pill);

describe('pillWhy — every row label pillFactors emits has a template', () => {
    test('labels match lib/jevBrief.js pillFactors exactly', () => {
        const f = pillFactors(data());
        for (const pill of PILLS) {
            expect([...WHY_LABELS[pill]].sort()).toEqual(f[pill].rows.map((r) => r.label).sort());
        }
    });
});

describe('pillWhy — each verdict reads from the rows that fired', () => {
    test('regime with a vote against', () => {
        expect(why(data({ fg: { score: 22 } }), 'regime')).toBe('2 of 3 votes: uptrend + junk bonds beating safe bonds · against: fearful crowd');
        expect(why(data({ spy: { ma200Pct: -4 }, fg: { score: 22 }, hyg: { chg20Pct: -2 } }), 'regime'))
            .toBe('0 of 3 votes · against: downtrend + fearful crowd + junk bonds lagging safe bonds');
        expect(why(data({ fg: { score: 70 } }), 'regime')).toBe('3 of 3 votes: uptrend + greedy crowd + junk bonds beating safe bonds');
    });
    test('recession warnings name their numbers', () => {
        expect(why(data({ fred: { sahmRule: 0.52 } }), 'recession')).toBe('1 of 4 warnings tripped: Sahm 0.52');
        expect(why(data({ fred: { yieldCurve: -0.3, claims: 275, nfci: 0.1 } }), 'recession'))
            .toBe('3 of 4 warnings tripped: curve inverted (−0.30) + claims 275k + money tight (NFCI 0.10)');
    });
    test('breadth: broad, rolling over, and narrow (nothing fired -> the lead fact)', () => {
        expect(why(data(), 'breadth')).toBe('Average stock +1.5% vs SPY in 20 days · small caps +2.0%');
        expect(why(data({ rsp: { chg20Pct: -2, vs50dPct: -1 } }), 'breadth')).toBe('Average stock −2.0% vs SPY in 20 days · below its 50-day trend');
        expect(why(data({ rsp: { chg20Pct: -0.5 } }), 'breadth')).toBe('Average stock −0.5% vs SPY in 20 days');
    });
    test('hedging: cheap, pricey, VRP-only pricey, and fair', () => {
        expect(why(data({ vol: { ivPctile1y: 12, vrp: 3 } }), 'hedging')).toBe('Options cheap: bottom 12% of the year');
        expect(why(data({ vol: { ivPctile1y: 82, vrp: 4 } }), 'hedging')).toBe('Options pricey: top 18% of the year');
        expect(why(data({ vol: { ivPctile1y: 50, vrp: 12.3 } }), 'hedging')).toBe('Options cost 12.30 pts over real moves');
        expect(why(data({ vol: { ivPctile1y: 43, vrp: 4 } }), 'hedging')).toBe('Options at the 43rd percentile of the year');
        // fair with IV in the bottom fifth (VRP >= 6 blocks "cheap"): no "mid-priced" claim.
        expect(why(data({ vol: { ivPctile1y: 15, vrp: 7 } }), 'hedging')).toBe('Options at the 15th percentile of the year');
    });
    test('conflict: each diverging pair, or all agree', () => {
        expect(why(data(), 'conflict')).toBe('All 4 pairs agree');
        expect(why(data({ fg: { score: 72 }, spy: { ma200Pct: -4 } }), 'conflict')).toBe('Greedy crowd in a downtrend');
        expect(why(data({ hyg: { chg20Pct: -2 } }), 'conflict')).toBe('Junk bonds lagging safe bonds in an uptrend');
        expect(why(data({ ten: -0.2 }), 'conflict')).toBe('Two yield curves disagree');
    });
});

describe('pillWhy — missing data shows nothing, never a guess', () => {
    test('no payload, no factors, no rows', () => {
        expect(pillWhy(null, 'regime')).toBeNull();
        expect(pillWhy({}, 'regime')).toBeNull();
        expect(pillWhy({ factors: {} }, 'regime')).toBeNull();
        expect(pillWhy({ factors: { regime: { summary: 'x', rows: [] } } }, 'regime')).toBeNull();
        expect(pillWhy({ factors: { regime: { rows: 'junk' } } }, 'regime')).toBeNull();
    });
    test('n/a inputs: no lead fact from a missing number', () => {
        const f = pillFactors(data({ rsp: { chg20Pct: null, vs50dPct: null }, iwm: { chg20Pct: null }, vol: { ivPctile1y: null, vrp: null } }));
        expect(pillWhy({ factors: f }, 'breadth')).toBeNull();
        expect(pillWhy({ factors: f }, 'hedging')).toBeNull();
    });
    test('a fired row with no template falls back to the summary, then the reason', () => {
        const factors = { breadth: { summary: 'Rolling over: RSP/SPY declining', rows: [{ label: 'Renamed row', value: '-3%', hit: true, effect: 'rolling-over' }] } };
        expect(pillWhy({ factors }, 'breadth')).toBe('Rolling over: RSP/SPY declining');
        const noSummary = { breadth: { rows: factors.breadth.rows } };
        expect(pillWhy({ factors: noSummary, pills: { breadth: { reason: 'RSP/SPY 20d -3%' } } }, 'breadth')).toBe('RSP/SPY 20d -3%');
        expect(pillWhy({ factors: noSummary }, 'breadth')).toBeNull();
    });
});

// Review fix: the line explains the RULE's rows. When Jev (p >= 0.6) overrides the rule with
// a different verdict, the bare line would argue against the badge above it.
describe('pillWhy — a Jev override never sits over a line that argues the other way', () => {
    const withPill = (pill, p) => ({ ...live, pills: { ...live.pills, [pill]: { ...live.pills[pill], ...p } } });

    test('Jev agrees with the rule: the line is unchanged', () => {
        // Live: recession low by Jev 0.88, hedging cheap by Jev 1.00 — both match the rule.
        expect(live.pills.recession.by).toBe('jev');
        expect(pillWhy(live, 'recession')).toBe('0 of 4 warnings tripped');
        expect(pillWhy(live, 'hedging')).toBe('Options cheap: bottom 15% of the year');
    });
    test('Jev disagrees: the line says it is the rule talking', () => {
        expect(pillWhy(withPill('regime', { verdict: 'risk-off', by: 'jev', p: 0.72 }), 'regime'))
            .toBe('Rule says risk-on · 2 of 3 votes: uptrend + junk bonds beating safe bonds');
        expect(pillWhy(withPill('recession', { verdict: 'high', by: 'jev', p: 0.8 }), 'recession'))
            .toBe('Rule says low · 0 of 4 warnings tripped');
        expect(pillWhy(withPill('breadth', { verdict: 'narrow', by: 'jev', p: 0.7 }), 'breadth'))
            .toBe('Rule says rolling over · average stock −4.3% vs SPY in 20 days · below its 50-day trend');
        expect(pillWhy(withPill('hedging', { verdict: 'fair', by: 'jev', p: 0.65 }), 'hedging'))
            .toBe('Rule says cheap · options cheap: bottom 15% of the year');
        expect(pillWhy(withPill('conflict', { verdict: 'aligned', by: 'jev', p: 0.9 }), 'conflict'))
            .toBe('Rule says major divergence · fearful crowd in an uptrend · near the high, average stock slipping');
    });
    test('a rule verdict the rows do not support (mismatched payload): no line', () => {
        expect(pillWhy(withPill('regime', { verdict: 'risk-off', by: 'rule', p: null }), 'regime')).toBeNull();
        expect(pillWhy(withPill('recession', { verdict: 'n/a', by: 'rule', p: null }), 'recession')).toBeNull();
    });
    test('Jev disagrees and only the rule summary is left to say: no line', () => {
        const factors = { breadth: { summary: 'Rolling over: RSP/SPY declining', rows: [{ label: 'Renamed row', value: '-3%', hit: true, effect: 'rolling-over' }] } };
        expect(pillWhy({ factors, pills: { breadth: { verdict: 'narrow', by: 'jev', p: 0.7 } } }, 'breadth')).toBeNull();
        expect(pillWhy({ factors, pills: { breadth: { verdict: 'rolling-over', by: 'rule' } } }, 'breadth')).toBe('Rolling over: RSP/SPY declining');
    });
});

// Review fix: a missing input used to count as "not tripped", so a FRED outage read as an
// all-clear ("0 of 4 warnings tripped"). Count only what was measured; say what was not.
describe('pillWhy — missing inputs are never counted as calm', () => {
    test('nothing measured at all: no line for any pill', () => {
        for (const d of [{}, { fred: { yieldCurve: 'N/A' } }]) {
            const f = pillFactors(d);
            for (const pill of PILLS) expect(pillWhy({ factors: f }, pill)).toBeNull();
        }
    });
    test('one recession input missing: counted out of what was measured, and named', () => {
        expect(why(data({ fred: { nfci: null } }), 'recession')).toBe('0 of 3 warnings tripped (NFCI n/a)');
        expect(why(data({ fred: { yieldCurve: 'N/A', sahmRule: 0.52 } }), 'recession')).toBe('1 of 3 warnings tripped: Sahm 0.52 (curve n/a)');
        expect(why(data({ fred: { sahmRule: null, claims: null } }), 'recession')).toBe('0 of 2 warnings tripped (Sahm, claims n/a)');
    });
    test('one regime input missing', () => {
        expect(why(data({ fg: { score: null } }), 'regime')).toBe('2 of 2 votes: uptrend + junk bonds beating safe bonds (F&G n/a)');
        expect(why(data({ spy: { ma200Pct: null }, fg: { score: 22 } }), 'regime')).toBe('1 of 2 votes: junk bonds beating safe bonds · against: fearful crowd (SPY trend n/a)');
    });
    test('conflict: "all agree" counts only the pairs that could be checked', () => {
        expect(why(data({ ten: null }), 'conflict')).toBe('All 3 pairs agree (1 pair n/a)');
        expect(why(data({ ten: null, fg: { score: null } }), 'conflict')).toBe('All 2 pairs agree (2 pairs n/a)');
    });
    test('the payload with every input present is unchanged', () => {
        expect(why(data(), 'recession')).toBe('0 of 4 warnings tripped');
        expect(why(data(), 'regime')).toBe('3 of 3 votes: uptrend + greedy crowd + junk bonds beating safe bonds');
        expect(why(data(), 'conflict')).toBe('All 4 pairs agree');
    });
});
