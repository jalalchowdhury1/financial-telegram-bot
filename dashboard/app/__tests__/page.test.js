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

const jevLive = require('../../lib/__tests__/fixtures/jev-pills-2026-10-04.json');

it('⚡ the Jev card is saved when it lands, and the next open paints it from that copy with its 🕐 tag', async () => {
    const { SNAP_PREFIX } = require('../../lib/snapshot');
    mockRoutes({ '/api/jev-pills': Promise.resolve({ status: 200, json: async () => jevLive }) });
    const first = render(<Dashboard />);
    await waitFor(() => expect(screen.getByText('🧪 Jev Regime Pills')).toBeInTheDocument());
    await waitFor(() => expect(window.localStorage.getItem(`${SNAP_PREFIX}jev`)).not.toBeNull());
    first.unmount();

    // Warm second open: /api/jev-pills is slow, the saved copy paints at once, tagged.
    mockRoutes({ '/api/jev-pills': never });
    render(<Dashboard />);
    const h = screen.getByText('🧪 Jev Regime Pills'); // synchronously: painted before any feed answered
    expect(h.closest('.saved-wrap')).toHaveAttribute('data-cached');
});

describe('honest labels on the SPY and Fear & Greed cards', () => {
    const { todayET } = require('../../lib/marks');
    const withChart = (lastDate) => ({
        ...spyPayload,
        chartHistory: [
            { date: '2026-10-01', price: 606.1, ma50: 600, ma200: 580 },
            { date: lastDate, price: 612.34, ma50: 601, ma200: 580.1 },
        ],
    });
    const ok = (body) => Promise.resolve({ status: 200, json: async () => body });
    const badge = () => document.querySelector('.daily-change-badge');

    it('the SPY move names its session like What moved: "Fri" on a weekend, not "today"', async () => {
        mockRoutes({ '/api/spy': ok(withChart('2026-10-02')), '/api/spy-daily-move': ok({ value: '0.74%' }) });
        render(<Dashboard />);
        await waitFor(() => expect(badge()).not.toBeNull());
        expect(badge().textContent).toBe('▲ 0.74% Fri');
        // the floating glance bar names the session too
        expect(document.querySelector('.glance-main').textContent).toBe('SPY 612.34 ▲0.74% Fri');
    });

    it('during the session it still says "today"; with no market date it keeps "today"', async () => {
        mockRoutes({ '/api/spy': ok(withChart(todayET())), '/api/spy-daily-move': ok({ value: '-0.31%' }) });
        const r = render(<Dashboard />);
        await waitFor(() => expect(badge()).not.toBeNull());
        expect(badge().textContent).toBe('▼ -0.31% today');
        expect(document.querySelector('.glance-main').textContent).toBe('SPY 612.34 ▼0.31%');
        r.unmount();
        // the first render's saved copies are written off the render path (setTimeout 0): flush, then forget them
        await act(async () => { await new Promise((res) => setTimeout(res, 0)); });
        window.localStorage.clear();

        mockRoutes({ '/api/spy-daily-move': ok({ value: null }) }); // spyPayload: no chart, no vol
        render(<Dashboard />);
        await waitFor(() => expect(badge()).not.toBeNull());
        expect(badge().textContent).toBe('▲ $1.20 (+0.20%) today');
    });

    it("Fear & Greed on its VIX proxy (previousYear 'N/A') shows — in that cell, never NaN", async () => {
        mockRoutes({
            '/api/fear-greed': ok({
                score: 34.2, rating: 'Fear', previousClose: 36.1, previousWeek: 40.9, previousMonth: 52.3, previousYear: 'N/A',
                _meta: { source: 'Yahoo VIX proxy', hasErrors: false },
            }),
        });
        render(<Dashboard />);
        await waitFor(() => expect(document.querySelectorAll('.fg-history-item')).toHaveLength(4));
        const cells = [...document.querySelectorAll('.fg-history-item')].map((c) => c.textContent);
        expect(cells).toEqual(['Prev Close36▼2', '1 Week41▼7', '1 Month52▼18', '1 Year—']);
        expect(document.body.textContent).not.toMatch(/NaN/);
    });

    it('a VIX proxy says it is not CNN; a cached copy says STALE + its date; live CNN says neither', async () => {
        const fg = (meta) => ok({ score: 45, rating: 'FEAR', previousClose: 38, previousWeek: 40, previousMonth: 38, previousYear: 49, _meta: meta });
        mockRoutes({ '/api/fear-greed': fg({ source: 'Yahoo ^VIX Proxy', proxy: true, hasErrors: true }) });
        const { unmount } = render(<Dashboard />);
        await waitFor(() => expect(document.querySelector('[data-testid="fg-provenance"]')).not.toBeNull());
        expect(document.querySelector('[data-testid="fg-provenance"]').textContent).toBe('⚠ VIX proxy, not CNN Fear & Greed');
        unmount();
        window.localStorage.clear();

        mockRoutes({ '/api/fear-greed': fg({ source: 'Stale KV last-good (2026-10-08T21:00:00.000Z) ← CNN', stale: true, lastGoodAt: '2026-10-08T21:00:00.000Z', hasErrors: true }) });
        const second = render(<Dashboard />);
        await waitFor(() => expect(document.querySelector('[data-testid="fg-provenance"]')).not.toBeNull());
        expect(document.querySelector('[data-testid="fg-provenance"]').textContent).toBe('⚠ STALE · cached 2026-10-08');
        second.unmount();
        window.localStorage.clear();

        mockRoutes({ '/api/fear-greed': fg({ source: 'CNN', hasErrors: false }) });
        render(<Dashboard />);
        await waitFor(() => expect(document.querySelectorAll('.fg-history-item')).toHaveLength(4));
        expect(document.querySelector('[data-testid="fg-provenance"]')).toBeNull();
    });

    it('📈 the CNN score opens its chart (sheet + CNN since 2021); the VIX proxy is a different number: no chart', async () => {
        const fg = (meta) => ok({ score: 45, rating: 'FEAR', previousClose: 38, previousWeek: 40, previousMonth: 38, previousYear: 49, _meta: meta });
        const history = ok({ today: '2026-10-10', metrics: {}, series: { from: '2026-09-21', days: 20, v: { cnnFearGreed: Array.from({ length: 20 }, (_, i) => 40 + (i % 6)) } } });
        const score = () => [...document.querySelectorAll('.hero-price')].find((h) => h.textContent === '45');
        mockRoutes({ '/api/fear-greed': fg({ source: 'CNN', hasErrors: false }), '/api/history': history });
        const first = render(<Dashboard />);
        await waitFor(() => expect(score()?.querySelector('.chartable')).toBeTruthy());
        await act(async () => { fireEvent.click(score().querySelector('.chartable')); });
        expect(document.querySelector('.mark-pop').getAttribute('aria-label')).toBe('CNN Fear & Greed chart');
        first.unmount();
        window.localStorage.clear();

        mockRoutes({ '/api/fear-greed': fg({ source: 'Yahoo ^VIX Proxy', proxy: true, hasErrors: true }), '/api/history': history });
        render(<Dashboard />);
        await waitFor(() => expect(score()).toBeTruthy());
        await act(async () => { await new Promise((res) => setTimeout(res, 0)); });
        expect(score().querySelector('.chartable')).toBeNull();
    });
});

it('📡 Market Pulse shows the verdicts of the cards below, the Rubber Band one handed up by its own card', async () => {
    const ok = (body) => Promise.resolve({ status: 200, json: async () => body });
    mockRoutes({
        '/api/fred': ok(require('../../lib/__tests__/fixtures/pulse-fred-2026-10-04.json')),
        '/api/vol': ok(require('../../lib/__tests__/fixtures/pulse-vol-2026-10-04.json')),
        '/api/rubber-band': ok(require('../../lib/__tests__/fixtures/pulse-rubber-band-2026-10-04.json')),
    });
    const { container } = render(<Dashboard />);
    // a plain DOM query: getByRole over the whole dashboard took >1 s per poll under a full parallel run
    await waitFor(() => expect([...container.querySelectorAll('.market-pulse button')].map((b) => b.textContent)).toContain('Dips pay ✓'), { timeout: 5000 });
    const line = container.querySelector('.market-pulse');
    const chips = [...line.querySelectorAll('button')].map((b) => b.textContent);
    expect(chips).toEqual(['Vol calm', 'Horsemen 1/4', 'Curve +0.45%', 'Bull 7/8', 'Dips pay ✓']);
    expect(line.textContent).not.toMatch(/RSI|F&G|SPY/);
});

it('📡 cold open: vol answers first, fred later — Market Pulse holds its place, then paints every chip at once', async () => {
    const ok = (body) => Promise.resolve({ status: 200, json: async () => body });
    let fredIn;
    mockRoutes({
        '/api/fred': new Promise((r) => { fredIn = () => r({ status: 200, json: async () => require('../../lib/__tests__/fixtures/pulse-fred-2026-10-04.json') }); }),
        '/api/vol': ok(require('../../lib/__tests__/fixtures/pulse-vol-2026-10-04.json')),
        '/api/rubber-band': ok(require('../../lib/__tests__/fixtures/pulse-rubber-band-2026-10-04.json')),
    });
    const { container } = render(<Dashboard />);
    await waitFor(() => expect(screen.getByText('$612.34')).toBeInTheDocument());
    await act(async () => { await new Promise((r) => setTimeout(r, 50)); }); // vol + rubber band are in
    const line = () => container.querySelector('.market-pulse');
    expect(line()).toHaveClass('is-waiting');
    expect(line().querySelectorAll('button')).toHaveLength(0); // no lone "Vol calm" for fred to land around
    await act(async () => { fredIn(); });
    await waitFor(() => expect(line()).not.toHaveClass('is-waiting'));
    expect([...line().querySelectorAll('button')].map((b) => b.textContent))
        .toEqual(['Vol calm', 'Horsemen 1/4', 'Curve +0.45%', 'Bull 7/8', 'Dips pay ✓']);
});

it('↩ after a Market Pulse chip jump, the back pill offers the way back', async () => {
    const { clearJump } = require('../../lib/jumpBack');
    const ok = (body) => Promise.resolve({ status: 200, json: async () => body });
    jest.spyOn(Element.prototype, 'getClientRects').mockImplementation(() => [{ top: 0 }]);
    Element.prototype.scrollIntoView = jest.fn();
    window.scrollY = 0;
    mockRoutes({
        '/api/fred': ok(require('../../lib/__tests__/fixtures/pulse-fred-2026-10-04.json')),
        '/api/vol': ok(require('../../lib/__tests__/fixtures/pulse-vol-2026-10-04.json')),
    });
    render(<Dashboard />);
    await waitFor(() => expect(screen.getByRole('button', { name: /^Vol calm/ })).toBeInTheDocument());
    expect(document.querySelector('.back-pill')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: /^Vol calm/ }));
    // jsdom has no layout (every top is 0), so only the pill itself is checked here; BackPill.test checks the names
    expect(screen.getByRole('button', { name: /^Back to / })).toHaveClass('is-on');
    act(() => clearJump());
});
