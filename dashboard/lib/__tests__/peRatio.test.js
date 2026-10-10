import { resolvePeRatio, parseMultplPe, parseYahooPe } from '../peRatio';

const MULTPL = '<div id="current"><b>Current S&P 500 PE Ratio:</b> 26.66 <span>-0.10</span></div>';
const YAHOO = '<td>PE Ratio (TTM)</td><td>25.00</td>';
const CAPE = [{ date: '2026-09-01', value: 38.4 }, { date: '2026-08-01', value: 38.1 }];

const fetchers = (over = {}) => ({
    multplHtml: async () => MULTPL,
    yahooHtml: async () => YAHOO,
    capeObs: async () => CAPE,
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

    test('?_fail=pe_multpl,pe_yahoo → CAPE, LABELLED as CAPE with its own monthly as-of', async () => {
        const r = await resolvePeRatio(fetchers(), new Set(['pe_multpl', 'pe_yahoo']));
        expect(r).toMatchObject({ peRatio: 38.4, peSource: 'cape', peIsCape: true, peAsOf: '2026-09-01' });
        expect(r.messages.join(' ')).toMatch(/Shiller CAPE .*NOT trailing-twelve-month/);
    });

    test('?_fail=pe_multpl,pe_yahoo,pe_fred → null (N/A), never an invented number', async () => {
        const r = await resolvePeRatio(fetchers(), new Set(['pe_multpl', 'pe_yahoo', 'pe_fred']));
        expect(r).toMatchObject({ peRatio: null, peSource: null, peIsCape: false });
        expect(r.messages[r.messages.length - 1]).toBe('P/E unavailable — all layers failed');
    });

    test('real failures cascade the same way and never throw; key is masked', async () => {
        const r = await resolvePeRatio(fetchers({
            multplHtml: boom,
            yahooHtml: async () => '<html>consent wall</html>',
            capeObs: async () => { throw new Error('bad url ?api_key=SECRET'); },
        }), new Set(), { maskKey: (s) => s.replace(/api_key=[^&\s]+/g, 'api_key=***') });
        expect(r.peRatio).toBeNull();
        expect(r.messages.join(' ')).not.toContain('SECRET');
        expect(r.messages).toContain('P/E Yahoo failed: no match');
    });

    test('empty CAPE observations → unavailable', async () => {
        const r = await resolvePeRatio(fetchers({ multplHtml: boom, yahooHtml: boom, capeObs: async () => [] }), new Set());
        expect(r.peSource).toBeNull();
    });
});
