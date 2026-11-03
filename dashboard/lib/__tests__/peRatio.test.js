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

    // 2 Nov 2026: multpl's whole site sat on Fri Oct 30 all Monday night; 5-day leeway kept it
    // winning, so the tile froze while the computed tier had Monday's close (Watch c79734d8).
    test('a page one close behind, 6 h past that close, is demoted (Mon 11 pm ET)', async () => {
        const mon11pm = new Date('2026-11-03T04:00:00Z');
        const r = await resolvePeRatio(fetchers({ multplHtml: async () => dated('4:00 PM EDT, Fri Oct 30'), computed: async () => ({ ...LEGS, spxDate: '2026-11-02' }) }), new Set(['pe_yahoo']), { now: mon11pm });
        expect(r).toMatchObject({ peSource: 'computed', peAsOf: '2026-11-02' });
        expect(r.messages.join(' ')).toMatch(/multpl stale \(2026-10-30\)/);
    });

    test('...but inside the 6 h grace after the close it still wins (Mon 6 pm ET)', async () => {
        const r = await resolvePeRatio(fetchers({ multplHtml: async () => dated('4:00 PM EDT, Fri Oct 30') }), new Set(), { now: new Date('2026-11-02T23:00:00Z') });
        expect(r).toMatchObject({ peSource: 'multpl', peAsOf: '2026-10-30' });
    });

    test('a December stamp read in January is last year', async () => {
        const r = await resolvePeRatio(fetchers({ multplHtml: async () => dated('4:00 PM EST, Wed Dec 31') }), new Set(), { now: new Date('2027-01-02T15:00:00Z') });
        expect(r.peAsOf).toBe('2026-12-31');
    });
});

// Tripwire: the cascade's "multpl is fresh enough" and /api/freshness's `pe` grade must agree.
// If the cascade ever keeps a multpl page the checker calls red, the tile freezes with a fresh
// backup unused (Watch c79734d8, 2 Nov 2026). Sweeps every hour of two weeks × page dates.
describe('cascade and freshness checker agree', () => {
    const { peItem, isRed } = require('../servedFreshness');
    const page = (d) => `Current S&P 500 PE Ratio: 25.86 <div id="timestamp">4:00 PM EDT, ${new Date(`${d}T12:00:00Z`).toUTCString().replace(/^(\w{3}), (\d{2}) (\w{3}).*/, (_, w, dd, m) => `${w} ${m} ${+dd}`)}</div>`;
    test('a multpl page the cascade keeps is never red', async () => {
        const bad = [];
        for (let h = 0; h < 14 * 24; h++) {
            const now = new Date(Date.parse('2026-10-26T04:00:00Z') + h * 3600e3);
            for (let back = 0; back < 6; back++) {
                const d = new Date(now.getTime() - back * 864e5).toISOString().slice(0, 10);
                const r = await resolvePeRatio(fetchers({ multplHtml: async () => page(d), yahooHtml: boom, computed: boom }), new Set(), { now });
                if (r.peSource !== 'multpl' || r.stale) continue;
                const item = peItem({ peRatio: r.peRatio, peIsCape: false, peRatioAsOf: r.peAsOf }, now.getTime());
                if (isRed(item)) bad.push(`${now.toISOString()} page ${d}`);
            }
        }
        expect(bad).toEqual([]);
    });
});
