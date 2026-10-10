import { resolvePeRatio, parseMultplPe, parseYahooPe } from '../peRatio';

const MULTPL = '<div id="current"><b>Current S&P 500 PE Ratio:</b> 26.66 <span>-0.10</span></div>';
const YAHOO = '<td>PE Ratio (TTM)</td><td>25.00</td>';
const LEGS = { spx: 7811.54, spxDate: '2026-10-09', eps: 295.36, epsDate: '2026-06-30' };

const fetchers = (over = {}) => ({
    multplHtml: async () => MULTPL,
    yahooHtml: async () => YAHOO,
    computed: async () => LEGS,
    ...over,
});
const boom = async () => { throw new Error('HTTP 503'); };

describe('parsers', () => {
    test('multpl and yahoo', () => {
        expect(parseMultplPe(MULTPL)).toBe(26.66);
        expect(parseYahooPe(YAHOO)).toBeCloseTo(26.75, 2); // 25 × 1.07
        expect(parseMultplPe('<html>changed layout</html>')).toBeNull();
        expect(parseYahooPe(null)).toBeNull();
    });
});

describe('resolvePeRatio', () => {
    test('happy path: multpl, TTM, not CAPE', async () => {
        const r = await resolvePeRatio(fetchers(), new Set());
        expect(r).toMatchObject({ peRatio: 26.66, peSource: 'multpl', peIsCape: false, peAsOf: null });
    });

    test('?_fail=pe_multpl → Yahoo', async () => {
        const r = await resolvePeRatio(fetchers(), new Set(['pe_multpl']));
        expect(r.peSource).toBe('yahoo');
        expect(r.messages[0]).toMatch(/injected fault: pe_multpl/);
    });

    test('?_fail=pe_multpl,pe_yahoo → computed S&P ÷ EPS, labelled computed, dated by the price', async () => {
        const r = await resolvePeRatio(fetchers(), new Set(['pe_multpl', 'pe_yahoo']));
        expect(r).toMatchObject({ peRatio: 26.45, peSource: 'computed', peIsCape: false, peAsOf: '2026-10-09' });
        expect(r.messages.join(' ')).toMatch(/P\/E computed: S&P 7811.54 \(2026-10-09\) ÷ EPS 295.36/);
    });

    test('?_fail=pe_multpl,pe_yahoo,pe_computed → null (N/A), never an invented number', async () => {
        const r = await resolvePeRatio(fetchers(), new Set(['pe_multpl', 'pe_yahoo', 'pe_computed']));
        expect(r).toMatchObject({ peRatio: null, peSource: null, peIsCape: false });
        expect(r.messages[r.messages.length - 1]).toBe('P/E unavailable — all layers failed');
    });

    test('real failures cascade the same way and never throw; key is masked', async () => {
        const r = await resolvePeRatio(fetchers({
            multplHtml: boom,
            yahooHtml: async () => '<html>consent wall</html>',
            computed: async () => { throw new Error('bad url ?api_key=SECRET'); },
        }), new Set(), { maskKey: (s) => s.replace(/api_key=[^&\s]+/g, 'api_key=***') });
        expect(r.peRatio).toBeNull();
        expect(r.messages.join(' ')).not.toContain('SECRET');
        expect(r.messages).toContain('P/E Yahoo failed: no match');
    });

    test('a bad leg (EPS 0, or a P/E outside 5–80) → unavailable, not a silly number', async () => {
        const bad = (legs) => fetchers({ multplHtml: boom, yahooHtml: boom, computed: async () => ({ ...LEGS, ...legs }) });
        expect((await resolvePeRatio(bad({ eps: 0 }), new Set())).peSource).toBeNull();
        expect((await resolvePeRatio(bad({ eps: 29.5 }), new Set())).peSource).toBeNull();
        const r = await resolvePeRatio(bad({}), new Set());
        expect(r.peSource).toBe('computed');
    });
});
