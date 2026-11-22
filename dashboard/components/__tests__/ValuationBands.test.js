/**
 * The SPY card's Bands view (his 22 Nov 2026 "S&P 500 Forward P/E Valuation Bands" chart): SPY on rungs of forward
 * earnings × a whole P/E, carried forward at the earnings growth analysts expect.
 */
import { render, screen } from '@testing-library/react';
import ValuationBands from '../ValuationBands';

const iso = (t) => new Date(t).toISOString().slice(0, 10);
// 300 days to 2026-11-20: forward earnings flat at $40 a share of SPY, the P/E climbing 16.2 → 18.4
const daily = Array.from({ length: 300 }, (_, i) => {
    const pe = 16.2 + (i / 299) * 2.2;
    return { date: iso(Date.UTC(2026, 10, 20) - (299 - i) * 864e5), price: 40 * pe, eps: 40, pe, a5: 19.8, a10: 19.1 };
});
const row = (date, g) => ({ date, pe: 18.4, a5: 19.8, a10: 19.1, ...(g != null && { g }) });
const show = (fwd) => render(<ValuationBands daily={daily} fwd={fwd} tf="ALL" onTf={() => {}} chip={() => ({})} />);
const text = () => document.body.textContent;

test('names the rungs SPY sits between and prices each rung where the analysts\' growth takes it', () => {
    show([row('2026-11-13'), row('2026-11-20', 10)]);
    expect(text()).toContain('$736 = 18.4x earnings');
    expect(text()).toContain('🪜 Between 18x and 19x · 5-yr avg 19.8x = $792');
    // 299 days on screen → 179 days ahead, earnings 10% a year higher
    const eps = 40 * 1.1 ** (179 / 365);
    expect(text()).toContain(`19x $${Math.round(eps * 19)}`);
    expect(text()).toContain(`16x $${Math.round(eps * 16)}`);
    expect(text()).toContain('analysts\' earnings +10.0%/yr → May 2027');
    expect(screen.getByRole('img').getAttribute('aria-label')).toBe('SPY at 18.4 times forward earnings, between the 18x and 19x bands');
});

test('no growth number in the last 45 days: the bands stop at today', () => {
    show([row('2026-09-25', 10), row('2026-11-20')]);
    expect(text()).not.toContain('analysts');
    expect(text()).toContain('19x $760');
});

test('FactSet unavailable says so', () => {
    render(<ValuationBands daily={[]} fwd={false} tf="ALL" onTf={() => {}} chip={() => ({})} />);
    expect(text()).toBe('Forward P/E unavailable right now');
});
