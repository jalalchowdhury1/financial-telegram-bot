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

describe('multpl page date (audit B6)', () => {
    // multpl's real layout (1 Nov 2026): the timestamp has no year.
    const dated = (stamp) => `<meta name="description" content="Current S&P 500 PE Ratio is 25.86, a change of -0.48 from previous market close." />\n<div id="current"><b>Current<span class="currentTitle">\nS&P 500 PE Ratio</span>:</b>\n25.86\n<div id="timestamp">\n\n\n${stamp}\n\n</div></div>`;
    const now = new Date('2026-11-01T15:00:00Z');

    test('the page\'s own date is the as-of (a Sunday load says Friday, not Sunday)', async () => {
        const r = await resolvePeRatio(fetchers({ multplHtml: async () => dated('4:00 PM EDT, Fri Oct 30') }), new Set(), { now });
        expect(r).toMatchObject({ peRatio: 25.86, peSource: 'multpl', peAsOf: '2026-10-30' });
        expect(r.stale).toBeFalsy();
    });

    test('a page dated 10 days back is demoted to the computed tier', async () => {
        const r = await resolvePeRatio(fetchers({ multplHtml: async () => dated('4:00 PM EDT, Thu Oct 22') }), new Set(['pe_yahoo']), { now });
        expect(r.peSource).toBe('computed');
        expect(r.messages.join(' ')).toMatch(/multpl stale \(2026-10-22\)/);
    });

    test('...and served last, labelled stale with its own date, only when nothing else answers', async () => {
        const r = await resolvePeRatio(fetchers({ multplHtml: async () => dated('4:00 PM EDT, Thu Oct 22'), yahooHtml: boom, computed: boom }), new Set(), { now });
        expect(r).toMatchObject({ peRatio: 25.86, peSource: 'multpl', peAsOf: '2026-10-22', stale: true });
    });

    test('a December stamp read in January is last year', async () => {
        const r = await resolvePeRatio(fetchers({ multplHtml: async () => dated('4:00 PM EST, Wed Dec 31') }), new Set(), { now: new Date('2027-01-02T15:00:00Z') });
        expect(r.peAsOf).toBe('2026-12-31');
    });
});
