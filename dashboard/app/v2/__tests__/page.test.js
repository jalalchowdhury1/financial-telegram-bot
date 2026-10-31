/**
 * v2 (app/v2/page.js) keeps v1's loading guarantees: a bad or missing feed breaks only
 * its own panel, one hung route never holds the header, refresh skips the edge cache,
 * and every chip that jumps to a card (What moved, the read, status line) finds it.
 */
import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react';
import DashboardV2 from '../page';

const spyPayload = {
    current: 612.34,
    ma200: { value: 580.1, pct: 5.56 },
    week52High: { value: 620.5, pct: -1.3 },
    return3y: 81.05,
    rsi: 55.2,
    dailyChange: { value: 1.2, pct: 0.2 },
    chartHistory: [{ date: '2026-01-02', price: 600, ma200: 578.4 }, { date: '2026-01-05', price: 612.34, ma200: 580.1 }],
    _meta: { source: 'Polygon', hasErrors: false },
};

function mockRoutes(overrides = {}) {
    const never = new Promise(() => {});
    global.fetch = jest.fn((url) => {
        const path = String(url).split('?')[0];
        if (path in overrides) return overrides[path];
        if (path === '/api/spy') return Promise.resolve({ status: 200, json: async () => spyPayload });
        if (path === '/api/market-extra') return never;
        return Promise.resolve({ status: 200, json: async () => ({}) });
    });
}

beforeEach(() => {
    window.localStorage.clear();
    window.scrollTo = jest.fn();
});
afterEach(() => jest.restoreAllMocks());

it('draws SPY as soon as its feed answers, while other feeds are empty or hung', async () => {
    mockRoutes();
    render(<DashboardV2 />);
    await waitFor(() => expect(screen.getByText('$612.34')).toBeInTheDocument());
    expect(screen.getByRole('heading', { name: 'Signals' })).toBeInTheDocument();
    expect(screen.getByText('+5.56%')).toBeInTheDocument();
});

it('every route failing with a 500 still draws the page, with no NaN', async () => {
    global.fetch = jest.fn(() => Promise.resolve({ status: 500, json: async () => ({ message: 'boom' }) }));
    jest.useFakeTimers();
    render(<DashboardV2 />);
    await act(async () => { await jest.advanceTimersByTimeAsync(4000); });
    jest.useRealTimers();
    expect(screen.getByText('SPY unavailable · tap ↻')).toBeInTheDocument();
    expect(screen.getByText('F&G unavailable · tap ↻')).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/NaN/);
});

it('a hung market-extra never holds the header on "Loading"', async () => {
    mockRoutes();
    render(<DashboardV2 />);
    await waitFor(() => expect(screen.getByText(/just now|ago/)).toBeInTheDocument());
    expect(document.querySelector('[data-jump="Markets"] .skeleton')).not.toBeNull();
});

it('automatic loads carry no cache-buster; the refresh button does', async () => {
    mockRoutes({ '/api/market-extra': Promise.resolve({ status: 200, json: async () => ({}) }) });
    render(<DashboardV2 />);
    await waitFor(() => expect(screen.getByText('$612.34')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByLabelText('Refresh all data')).not.toBeDisabled());
    expect(global.fetch.mock.calls.every(([u]) => !/[?&]_t=/.test(String(u)))).toBe(true);
    global.fetch.mockClear();
    fireEvent.click(screen.getByLabelText('Refresh all data'));
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    expect(global.fetch.mock.calls.some(([u]) => /[?&]_t=/.test(String(u)))).toBe(true);
});

it('keeps every jump target v1 has, so chips land on the right panel', async () => {
    mockRoutes();
    render(<DashboardV2 />);
    await waitFor(() => expect(screen.getByText('$612.34')).toBeInTheDocument());
    const jumps = new Set([...document.querySelectorAll('[data-jump]')].map((e) => e.getAttribute('data-jump')));
    for (const label of ['SPY overview', 'Fear & Greed', 'Indicators', 'Yield curve', 'Profit margin', 'S&P 500 EPS',
        'Economy', 'Recession watch', 'Rubber band', 'SPY chart', 'Volatility', 'Bull checklist', 'Markets', 'Polymarket']) {
        expect([label, jumps.has(label)]).toEqual([label, true]);
    }
});

it('each economy chart says what stands out vs its whole history, and lights up when it matters', async () => {
    const months = (n, f) => Array.from({ length: n }, (_, i) => ({ date: `${2000 + Math.floor(i / 12)}-${String((i % 12) + 1).padStart(2, '0')}-01`, value: f(i) }));
    const fred = {
        yieldCurve: { current: -0.2, asOf: '2004-12-01', history: months(60, (i) => (i < 54 ? 1 : -0.2)) },
        profitMargin: { current: 12, asOf: '2004-12-01', history: months(60, (i) => (i === 59 ? 12 : 5 + (i % 7))) },
        spEps: { current: 90, asOf: '2004-12-01', history: months(60, (i) => (i === 59 ? 90 : 100)) },
        recessions: [],
    };
    mockRoutes({ '/api/fred': Promise.resolve({ status: 200, json: async () => fred }) });
    render(<DashboardV2 />);
    const tile = (jump) => document.querySelector(`.v2-fred[data-jump="${jump}"]`);
    await waitFor(() => expect(tile('Yield curve').querySelector('.v2-fred-badge').textContent).toBe('Inverted since Jul 2004'));
    expect(tile('Yield curve').className).toMatch(/lit-bad is-lit/);
    expect(tile('Profit margin').querySelector('.v2-fred-badge').textContent).toBe('Record high since 2000');
    expect(tile('S&P 500 EPS').querySelector('.v2-fred-badge').textContent).toBe('10% off high · −10% y/y');
    expect(tile('S&P 500 EPS').className).toMatch(/lit-bad is-lit/);
});

it('USD/BDT adds the compound yearly rise (and its dashed steady path) on 5Y and longer only', async () => {
    const baked = { from: '2016-10-30', t: [0, 1820, 3400, 3653], v: [80, 100, 150, 160] };
    mockRoutes({ '/v2/usdbdt.json': Promise.resolve({ ok: true, status: 200, json: async () => baked }) });
    render(<DashboardV2 />);
    await waitFor(() => expect(document.querySelector('.v2-taka-now')).not.toBeNull());
    const card = within(document.querySelector('.v2-taka'));
    expect(document.querySelector('.v2-taka-cagr')).toBeNull(); // 1Y
    fireEvent.click(card.getByRole('button', { name: '5Y' }));
    expect(document.querySelector('.v2-taka-cagr').textContent).toBe('+9.8%/yr'); // 100 → 160 in 1,833 days
    fireEvent.click(card.getByRole('button', { name: 'All' }));
    expect(document.querySelector('.v2-taka-cagr').textContent).toBe('+7.2%/yr'); // 80 → 160 in 10.0 years
    expect(document.querySelector('.v2-taka-pace')).not.toBeNull();
});

it('Fear & Greed draws its Prev / 1W / 1M / 1Y readings as one trail, each with its move to now', async () => {
    const fg = { score: 26.2, rating: 'FEAR', previousClose: 34.6, previousWeek: 16.7, previousMonth: 30.3, previousYear: 'N/A', asOf: '2026-10-30T23:59:48+00:00', _meta: { source: 'CNN' } };
    mockRoutes({ '/api/fear-greed': Promise.resolve({ status: 200, json: async () => fg }) });
    render(<DashboardV2 />);
    await waitFor(() => expect(document.querySelector('.v2-fgt')).not.toBeNull());
    expect([...document.querySelectorAll('.v2-fgt-axis span')].map((e) => e.textContent)).toEqual(['1M▼4', '1W▲9', '1D▼9', 'Now']); // no 1Y: the proxy tier's 'N/A'
    expect([...document.querySelectorAll('.v2-fgt-pt b')].map((e) => e.textContent)).toEqual(['30', '17', '35']);
    expect(document.querySelector('.v2-fgt-pt.is-now').className).toMatch(/tone-caution/);
});

it('the AAII tile opens what SPY did after every 20+ reading', async () => {
    const aaii = { from: '2020-01-02', t: [0, 7], v: [21, 5] };
    const spy = { from: '2020-01-01', t: Array.from({ length: 500 }, (_, i) => i), v: Array.from({ length: 500 }, (_, i) => 100 + i) };
    const ok = (body) => Promise.resolve({ ok: true, status: 200, json: async () => body });
    mockRoutes({ '/api/sheets': ok({ AAIIDiff: '3.60%' }), '/history/aaiiDiff.json': ok(aaii), '/history/spy-daily.json': ok(spy) });
    render(<DashboardV2 />);
    await waitFor(() => expect(screen.getByText('AAII', { selector: '.v2-signal-label span' })).toBeInTheDocument());
    const tile = screen.getByText('AAII', { selector: '.v2-signal-label span' }).closest('[role="button"]');
    fireEvent.click(tile);
    const dialog = await screen.findByRole('dialog', { name: 'AAII fear at 20+' });
    await waitFor(() => expect(within(dialog).getAllByText('+180.2%')).toHaveLength(2)); // avg + median: Jan 2 (101) → Jul 2 (283)
    expect(within(dialog).getByText('100%')).toBeInTheDocument();
    expect(within(dialog).getAllByRole('button', { name: /SPY \+180\.2%/ })).toHaveLength(1);
    fireEvent.click(within(dialog).getByRole('button', { name: '12M' }));
    expect(window.localStorage.getItem('v2-aaii-months')).toBe('12');
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('dialog', { name: 'AAII fear at 20+' })).toBeNull();
});

it('What moved draws each move as its last month against a normal day (30 Oct: a stray USD/BDT line crashed the strip)', async () => {
    const day = (i) => new Date(Date.now() - (30 - i) * 86400000).toISOString().slice(0, 10);
    const chartHistory = Array.from({ length: 30 }, (_, i) => ({ date: day(i), price: i % 2 ? 101 : 100 }));
    const spy = { ...spyPayload, current: 102, dailyChange: { value: 2, pct: 2 }, chartHistory };
    mockRoutes({ '/api/spy': Promise.resolve({ status: 200, json: async () => spy }) });
    render(<DashboardV2 />);
    await waitFor(() => expect(document.querySelector('.v2-mv-past')).not.toBeNull());
    expect(document.querySelector('.v2-mv-past').getAttribute('d')).not.toMatch(/NaN/);
    expect(screen.queryByText(/Component Error/)).toBeNull();
});

it('the VIX / 3M tile opens what SPY did after each day VIX closed over VIX 3M, 1 month out first', async () => {
    const ratio = { from: '2020-01-02', t: [0, 1], v: [1.05, 0.9] };
    const spy = { from: '2020-01-01', t: Array.from({ length: 500 }, (_, i) => i), v: Array.from({ length: 500 }, (_, i) => 100 + i) };
    const ok = (body) => Promise.resolve({ ok: true, status: 200, json: async () => body });
    mockRoutes({ '/api/sheets': ok({ VIX: { current: '19.42', threeMonth: '20.78' } }), '/history/vix-ratio-daily.json': ok(ratio), '/history/spy-daily.json': ok(spy) });
    render(<DashboardV2 />);
    await waitFor(() => expect(screen.getByText('VIX / 3M')).toBeInTheDocument());
    fireEvent.click(screen.getByText('VIX / 3M').closest('[role="button"]'));
    const dialog = await screen.findByRole('dialog', { name: 'VIX over VIX 3M' });
    expect(within(dialog).getByText('Now 19.42 / 20.78 · 0.93')).toBeInTheDocument();
    await waitFor(() => expect(within(dialog).getAllByText('+30.7%')).toHaveLength(2)); // Jan 2 (101) → Feb 2 (132)
    expect(within(dialog).getByRole('button', { name: '1M' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(within(dialog).getByRole('button', { name: /Jan 2, 2020/ }));
    expect(within(dialog).getByText('VIX/3M 1.050')).toBeInTheDocument();
});

it('Risk: each card gives one answer closed, and More opens the proof (loop, fuses, each tell since 2000)', async () => {
    const ok = (body) => Promise.resolve({ ok: true, status: 200, json: async () => body });
    const monthly = (f) => Array.from({ length: 322 }, (_, i) => ({ date: `${2000 + Math.floor(i / 12)}-${String((i % 12) + 1).padStart(2, '0')}-01`, value: f(i) }));
    const history = Array.from({ length: 40 }, (_, i) => ({ d: `2026-0${1 + Math.floor(i / 10)}-${10 + (i % 10)}`, slow: 0.5, fast: 0.4, rip: -0.1 }));
    const rb = {
        asOf: '2026-10-30', history,
        verdict: { colour: 'green', text: 'Working' },
        dials: {
            slow: { colour: 'green', excess_pct: 0.66, hit: 0.633, red_days: 0, stop_after: 45, window: 60, n: 30, se_pct: 0.5 },
            fast: { colour: 'green', excess_pct: 0.641, n: 20, se_pct: 0.7 },
            age: { colour: 'green', years: 2.5, amber_years: 3.3, red_years: 4 },
            rip: { colour: 'green', excess_pct: -0.06, hot_days: 0, red_after: 45, window: 60, n: 30, se_pct: 0.2, hit: 0.5 },
            machines: { colour: 'green', reasons: [], legs: [{ name: 'C3', line_pct: -54, dd_pct: -1.48 }, { name: 'hedges', role: 'hedge', dd_pct: -15.9 }] },
        },
        defensive: { mode: 'INVESTED', streak: { machines: 0 }, green_streak: 39, rules: { fire_after: { slow: 1, rip: 1, machines: 5 }, reentry_closes: 10, reentry_edge_pct: 0.2 } },
    };
    const fred = {
        horsemen: {
            claims: { current: 200000, history: monthly(() => 200000) },
            unemployment: { current: 4.2, history: monthly(() => 4.2) },
            bankruptcies: { current: 23000, history: monthly((i) => (i === 321 ? 23000 : 20000)) },
        },
        yieldCurve: { current: 0.38, history: monthly(() => 0.38) },
        indicators: { sahmRule: { value: 0 } },
        recessions: [{ start: '2008-01-01', end: '2009-07-01' }, { start: '2020-03-01', end: '2020-05-01' }],
    };
    mockRoutes({ '/api/rubber-band': ok(rb), '/api/fred': ok(fred) });
    const { container } = render(<DashboardV2 />);
    const slot = (jump) => container.querySelector(`[data-jump="${jump}"]`);

    await waitFor(() => expect(within(slot('Rubber band')).getByText('Stay invested')).toBeInTheDocument());
    expect(slot('Rubber band')).toHaveTextContent('Do dips bounce back?+0.66%');
    expect(slot('Rubber band')).toHaveTextContent('Do rallies cool off?−0.06%');
    expect(slot('Rubber band')).toHaveTextContent('Machines healthy?2 of 2');
    fireEvent.click(within(slot('Rubber band')).getByRole('button', { name: 'More' }));
    expect(within(slot('Rubber band')).getByText('39 green days')).toBeInTheDocument();
    expect(slot('Rubber band')).toHaveTextContent('0/45 bad days');
    expect(slot('Rubber band')).toHaveTextContent('0/45 hot days');
    expect(slot('Rubber band')).toHaveTextContent('0/5 red closes');
    expect(within(slot('Rubber band')).getByText('trips at −54%')).toBeInTheDocument();
    expect(within(slot('Rubber band')).getByText('no limit yet')).toBeInTheDocument();

    await waitFor(() => expect(within(slot('Recession watch')).getByText('1 of 4 riding')).toBeInTheDocument());
    expect(within(slot('Recession watch')).getByText('▲ 15.0% vs 1y')).toBeInTheDocument();
    expect(within(slot('Recession watch')).getByText('Normal')).toBeInTheDocument();
    fireEvent.click(within(slot('Recession watch')).getByRole('button', { name: 'More' }));
    expect(within(slot('Recession watch')).getByText('200K now')).toBeInTheDocument();
    expect(slot('Recession watch')).toHaveTextContent('4.20% now · Sahm 0.00');
    expect(within(slot('Recession watch')).getByText('where past ones began')).toBeInTheDocument();
    expect(slot('Recession watch').querySelectorAll('.rk-band').length).toBeGreaterThanOrEqual(8); // 2 recessions × 4 charts
});

describe('the top line', () => {
    const ok = (body) => Promise.resolve({ ok: true, status: 200, json: async () => body });
    const calm = { AAIIDiff: '3.60%', VIX: { current: '19.42', threeMonth: '20.78' } };
    const flat = { from: '2025-01-01', t: Array.from({ length: 100 }, (_, i) => i * 7), v: Array(100).fill(5) };
    const files = Object.fromEntries(['aaiiDiff', 'cl', 'claims', 'usdcad', 'vixCurrent', 'housing', 'peRatio', 'tnx', 'spy-daily'].map((k) => [`/history/${k}.json`, ok(flat)]));
    const combosSettled = () => waitFor(() => expect([...document.querySelectorAll('.v2-cb-now')].every((e) => e.textContent !== '…')).toBe(true));

    it('says Nothing firing, and the closest signal, only when every signal was read', async () => {
        mockRoutes({ '/api/sheets': ok(calm), ...files });
        render(<DashboardV2 />);
        await waitFor(() => expect(screen.getByText('Nothing firing')).toBeInTheDocument());
        expect(document.querySelector('.v2-today button').textContent).toBe('closest VIX/3M 0.93 → 1');
    });

    it('a feed that failed is not a calm one: no Nothing firing while the combos are unread', async () => {
        mockRoutes({ '/api/sheets': ok(calm) });
        render(<DashboardV2 />);
        await waitFor(() => expect(screen.getByText('19.42')).toBeInTheDocument());
        await combosSettled();
        expect(document.querySelector('.v2-today')).not.toBeNull(); // the line is there, holding its place
        expect(screen.queryByText('Nothing firing')).toBeNull();
    });

    it.each([['failed', undefined], ['hangs', new Promise(() => {})]])('a signal that fires still shows when the combos feed %s', async (_, combos) => {
        mockRoutes({ '/api/sheets': ok({ ...calm, AAIIDiff: '25.00%' }), ...(combos && { '/history/spy-daily.json': combos }) });
        render(<DashboardV2 />);
        await waitFor(() => expect(document.querySelector('.v2-today button')?.textContent).toMatch(/^AAII bullish to [A-Z][a-z]{2} \d+$/));
    });
});
