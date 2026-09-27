/**
 * The dashboard page's loading behaviour (app/page.js): each card renders as soon as
 * ITS OWN feed answers, one slow route can't hold the page, a failed feed never blanks
 * a card, and a manual refresh (button or R) skips the edge cache.
 */
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import Dashboard from '../page';

const spyPayload = {
    current: 612.34,
    ma200: { value: 580.1, pct: 5.56 },
    week52High: { value: 620.5, pct: -1.3 },
    return3y: 81.05,
    rsi: 55.2,
    dailyChange: { value: 1.2, pct: 0.2 },
    _meta: { source: 'Polygon', hasErrors: false },
};

let never; // a promise that never settles: the "slow route"
function mockRoutes(overrides = {}) {
    never = new Promise(() => {});
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

it('a malformed payload breaks only its own card, never the page', async () => {
    // Every other route answers `{}` here — /api/fred without its `indicators` used to
    // throw inside EconomicIndicatorGrid's own render, outside any boundary: blank page.
    mockRoutes();
    render(<Dashboard />);
    await waitFor(() => expect(screen.getByText('$612.34')).toBeInTheDocument());
    expect(screen.getByText('Jalal\'s Financial Dashboard')).toBeInTheDocument();
});

it('shows the SPY price while a slow route (market-extra) is still loading', async () => {
    mockRoutes();
    render(<Dashboard />);
    await waitFor(() => expect(screen.getByText('$612.34')).toBeInTheDocument());
    expect(screen.getByText(/Loading live data/)).toBeInTheDocument(); // the page as a whole is still loading
});

it('automatic loads carry no cache-buster; the refresh button and R do', async () => {
    mockRoutes({ '/api/market-extra': Promise.resolve({ status: 200, json: async () => ({}) }) });
    render(<Dashboard />);
    await waitFor(() => expect(screen.getByText(/Updated/)).toBeInTheDocument());
    const first = global.fetch.mock.calls.map((c) => String(c[0])).filter((u) => /^\/api\/spy(\?|$)/.test(u));
    expect(first).toEqual(['/api/spy']);

    global.fetch.mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh all data' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh all data' })).not.toBeDisabled());
    const busted = global.fetch.mock.calls.map((c) => String(c[0])).filter((u) => /^\/api\/spy(\?|$)/.test(u));
    expect(busted).toHaveLength(1);
    expect(busted[0]).toMatch(/^\/api\/spy\?_t=\d+$/);

    global.fetch.mockClear();
    await act(async () => { fireEvent.keyDown(document.body, { key: 'r' }); });
    await waitFor(() => expect(global.fetch.mock.calls.some((c) => /^\/api\/sheets\?_t=/.test(String(c[0])))).toBe(true));

    // Cmd+R is the browser's reload, not ours.
    global.fetch.mockClear();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh all data' })).not.toBeDisabled());
    fireEvent.keyDown(document.body, { key: 'r', metaKey: true });
    expect(global.fetch).not.toHaveBeenCalled();
});

it('a feed that fails on refresh keeps the card it already had', async () => {
    mockRoutes({ '/api/market-extra': Promise.resolve({ status: 200, json: async () => ({}) }) });
    render(<Dashboard />);
    await waitFor(() => expect(screen.getByText('$612.34')).toBeInTheDocument());
    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh all data' })).not.toBeDisabled());

    global.fetch.mockImplementation(() => Promise.reject(new TypeError('Failed to fetch')));
    jest.useFakeTimers();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh all data' }));
    await act(async () => { await jest.advanceTimersByTimeAsync(2000); });
    jest.useRealTimers();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh all data' })).not.toBeDisabled());
    expect(screen.getByText('$612.34')).toBeInTheDocument();
});

it('👋 since last visit: compares against the last visit, and a tab coming back never reads "flat" before the refresh lands', async () => {
    const real = Date.now.bind(Date);
    const at = real() - 2 * 3600e3;
    window.localStorage.setItem('fd:seen:v1', JSON.stringify({ v: { spy: { x: 600, at } }, p: {} }));
    const extra = { '/api/market-extra': Promise.resolve({ status: 200, json: async () => ({}) }) };
    mockRoutes(extra);
    render(<Dashboard />);
    await waitFor(() => expect(screen.getByRole('note').textContent).toMatch(/SPY \+2\.1%/)); // 600 → 612.34

    // Two hours hidden, then back: the old numbers are still on screen, the refresh is in flight.
    let land;
    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh all data' })).not.toBeDisabled());
    mockRoutes({ ...extra, '/api/spy': new Promise((r) => { land = r; }) });
    jest.spyOn(Date, 'now').mockImplementation(() => real() + 2 * 3600e3);
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); });
    expect(screen.queryByRole('note')).toBeNull(); // not "SPY flat": nothing new has landed yet

    await act(async () => { land({ status: 200, json: async () => ({ ...spyPayload, current: 624.58 }) }); });
    await waitFor(() => expect(screen.getByRole('note').textContent).toMatch(/SPY \+2\.0%/)); // 612.34 → 624.58
});
