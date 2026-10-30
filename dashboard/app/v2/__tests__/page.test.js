/**
 * v2 (app/v2/page.js) keeps v1's loading guarantees: a bad or missing feed breaks only
 * its own panel, one hung route never holds the header, refresh skips the edge cache,
 * and every chip that jumps to a card (What moved, the read, status line) finds it.
 */
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import DashboardV2 from '../page';

const spyPayload = {
    current: 612.34,
    ma200: { value: 580.1, pct: 5.56 },
    week52High: { value: 620.5, pct: -1.3 },
    return3y: 81.05,
    rsi: 55.2,
    dailyChange: { value: 1.2, pct: 0.2 },
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
