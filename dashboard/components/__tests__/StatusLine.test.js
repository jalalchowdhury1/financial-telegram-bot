import { render, screen, fireEvent } from '@testing-library/react';
import StatusLine, { statusFeeds } from '../StatusLine';
import { jumpToCard } from '../WhatMoved';

jest.mock('../WhatMoved', () => ({ jumpToCard: jest.fn() }));

// _meta as the live site served them on 25 Oct 2026
const healthy = () => ({
    spy: { current: 759.97, rsi: 37.4, ma200: { value: 727, pct: 4.5 }, week52High: { value: 779.1, pct: -2.5 },
        _meta: { source: 'yfinance + Finnhub Spot', hasErrors: false } },
    fred: { yieldCurve: { current: 0.55 }, _meta: { source: 'St. Louis Fed', hasErrors: false, messages: ['Loaded 17/17 series', 'S&P EPS: multpl'] } },
    fg: { score: 21, _meta: { source: 'CNN', hasErrors: false } },
    sheets: { NotSoBoring: 'OFF', _meta: { source: 'Computed (daily prices) + Google Sheets (Live)', hasErrors: false } },
    extra: { fx: { usdcad: {} }, _meta: { source: 'yfinance/Polygon/Finnhub/FRED/ER-API', hasErrors: false, messages: ['Polygon: 4', 'CNBC: 1', 'FRED: 2', 'ER-API: 4'] } },
});
const line = () => document.querySelector('.status-items').textContent;

test('all healthy: one plain line that starts with ✓, no chips', () => {
    render(<StatusLine bodies={healthy()} saved={{}} />);
    expect([...document.querySelectorAll('.status-item')].map((e) => e.textContent))
        .toEqual(['✓ feeds live', 'SPY yfinance', 'FRED 17/17', 'F&G CNN', 'Sheets live', 'Markets Polygon: 4 · CNBC: 1 · FRED: 2 · ER-API: 4']);
    expect(screen.queryAllByRole('button')).toHaveLength(0);
});

test('a degraded SPY source is an amber chip that jumps to the SPY card', () => {
    const b = healthy();
    b.spy._meta = { source: 'FRED Fallback', hasErrors: false };
    render(<StatusLine bodies={b} saved={{}} />);
    const chip = screen.getByRole('button', { name: '⚠ SPY: FRED fallback' });
    expect(chip).toHaveClass('status-warn');
    expect(line()).not.toMatch(/✓/);
    fireEvent.click(chip);
    expect(jumpToCard).toHaveBeenCalledWith('SPY overview');
});

test('nothing answered (offline): every feed reads "no data" in red, never LIVE OK or 18/18', () => {
    render(<StatusLine bodies={{ spy: null, fred: null, fg: null, sheets: null, extra: null }} saved={{}} />);
    const chips = screen.getAllByRole('button');
    expect(chips.map((c) => c.textContent)).toEqual(['⚠ SPY: no data', '⚠ FRED: no data', '⚠ F&G: no data', '⚠ Sheets: no data', '⚠ Markets: no data']);
    chips.forEach((c) => expect(c).toHaveClass('status-error'));
    expect(line()).not.toMatch(/LIVE OK|18\/18|✓/);
});

test("a route's fallback body (error, or no data in it) is not green", () => {
    const f = statusFeeds({ ...healthy(), spy: { error: 'SPY temporarily unavailable' }, extra: { fx: {}, commodities: {}, rates: {} } }, {});
    expect(f.find((x) => x.key === 'spy')).toMatchObject({ text: 'no data', level: 'error' });
    expect(f.find((x) => x.key === 'extra')).toMatchObject({ text: 'no data', level: 'error' });
});

test('a saved copy on screen reads "🕐 saved 10:42", never live', () => {
    render(<StatusLine bodies={{ ...healthy(), fred: null }} saved={{ fred: '10:42' }} />);
    expect(screen.getByRole('button', { name: '🕐 FRED: saved 10:42' })).toHaveClass('status-warn');
    expect(line()).not.toMatch(/✓/);
});

test("SPY from the server's last-good copy is amber 'stale'", () => {
    const b = healthy();
    b.spy._meta = { source: 'KV last-good (2026-10-23T20:00:00.000Z) ← yfinance', stale: true, hasErrors: true };
    expect(statusFeeds(b, {}).find((x) => x.key === 'spy')).toMatchObject({ text: 'stale', level: 'warn' });
});

test('problem chips lead the line (red, then amber), so on a phone a down feed is never past the screen edge', () => {
    const b = healthy();
    b.extra = { fx: {}, commodities: {}, rates: {}, _meta: { hasErrors: true } };
    render(<StatusLine bodies={b} saved={{ sheets: '10:42' }} />);
    expect([...document.querySelectorAll('.status-item')].map((e) => e.textContent))
        .toEqual(['⚠ Markets: no data', '🕐 Sheets: saved 10:42', 'SPY yfinance', 'FRED 17/17', 'F&G CNN']);
});
