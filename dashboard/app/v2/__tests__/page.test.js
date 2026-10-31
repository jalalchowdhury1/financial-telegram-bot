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
    await waitFor(() => expect(screen.getByText('AAII')).toBeInTheDocument());
    const tile = screen.getByText('AAII').closest('[role="button"]');
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
