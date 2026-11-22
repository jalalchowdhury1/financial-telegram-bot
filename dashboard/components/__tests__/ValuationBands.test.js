/**
 * The SPY card's Bands view (his 22 Nov 2026 "S&P 500 Forward P/E Valuation Bands" chart): SPY on rungs of forward
 * earnings × a whole P/E, carried forward at the earnings growth analysts expect.
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ValuationBands from '../ValuationBands';

const iso = (t) => new Date(t).toISOString().slice(0, 10);
// 300 days to 2026-11-20: forward earnings flat at $40 a share of SPY, the P/E climbing 16.2 → 18.4
const daily = Array.from({ length: 300 }, (_, i) => {
    const pe = 16.2 + (i / 299) * 2.2;
    return { date: iso(Date.UTC(2026, 10, 20) - (299 - i) * 864e5), price: 40 * pe, eps: 40, pe, a5: 19.8, a10: 19.1 };
});
const row = (date, g, gy = 2027) => ({ date, pe: 18.4, a5: 19.8, a10: 19.1, ...(g != null && { g, gy }) });
const show = (fwd) => render(<ValuationBands daily={daily} fwd={fwd} tf="ALL" onTf={() => {}} chip={() => ({})} />);
const text = () => document.body.textContent;

test('names the rungs SPY sits between and prices each rung where the analysts\' growth takes it', () => {
    show([row('2026-11-13'), row('2026-11-20', 10)]);
    expect(text()).toContain('$736 = 18.4x earnings');
    expect(text()).toContain('Between 18x and 19x · 5-yr avg 19.8x = $792');
    // 299 days on screen → 179 days ahead, earnings 10% a year higher
    const eps = 40 * 1.1 ** (179 / 365);
    expect(text()).toContain(`19x $${Math.round(eps * 19)}`);
    expect(text()).toContain(`16x $${Math.round(eps * 16)}`);
    expect(text()).toContain('if earnings +10.0% (analysts\' 2027) → May 2027');
    expect(screen.getByRole('img').getAttribute('aria-label')).toBe('SPY at 18.4 times forward earnings, between the 18x and 19x bands');
});

test('no growth number in the last 45 days: the bands stop at today', () => {
    show([row('2026-09-25', 10), row('2026-11-20')]);
    expect(text()).not.toContain('analysts');
    expect(text()).toContain('19x $760');
});

test('a blog-read week naming only this year never replaces the report that names next year', () => {
    show([row('2026-11-13', 10), row('2026-11-20', 33.3, 2026)]);
    expect(text()).toContain('if earnings +10.0% (analysts\' 2027)');
});

test('on a window whose rungs step by 2 the lit band is still one P/E wide, the one the tile names', () => {
    const wide = daily.map((p, i) => ({ ...p, pe: 6 + (i / 299) * 12.4, price: 40 * (6 + (i / 299) * 12.4) }));
    render(<ValuationBands daily={wide} fwd={[row('2026-11-20', 10)]} tf="ALL" onTf={() => {}} chip={() => ({})} />);
    expect(text()).toContain('Between 18x and 19x');
    expect(text()).toContain('20x $');
    expect(text()).not.toContain('19x $');
});

test('FactSet unavailable says so', () => {
    render(<ValuationBands daily={[]} fwd={false} tf="ALL" onTf={() => {}} chip={() => ({})} />);
    expect(text()).toBe('Forward P/E unavailable right now');
});

test('🔭 runs the rungs out to the end of 2030: the analysts\' year, then forward earnings\' own 10-year pace', async () => {
    // 6 years of days: forward earnings double (12.2% a year), the P/E flat at 18.4
    const n = 2192, long = Array.from({ length: n }, (_, i) => {
        const eps = 20 * 2 ** (i / (n - 1));
        return { date: iso(Date.UTC(2026, 10, 20) - (n - 1 - i) * 864e5), price: eps * 18.4, eps, pe: 18.4, a5: 19.8, a10: 19.1 };
    });
    render(<ValuationBands daily={long} fwd={[row('2026-11-20', 10)]} tf="ALL" onTf={() => {}} chip={() => ({})} />);
    expect(text()).toContain('(analysts\' 2027) → Nov 2027'); // the default view: a year ahead
    const scope = screen.getByRole('button', { name: 'Look ahead to 2030' });
    expect(scope.getAttribute('aria-pressed')).toBe('false');
    fireEvent.click(scope);
    await waitFor(() => expect(text()).toContain('if +10.0% (analysts\' 2027), then +12.2%/yr (10-yr pace)'));
    const pace = 2 ** (365.25 / (n - 1)) - 1, days = Math.round((Date.UTC(2030, 11, 31) - Date.UTC(2026, 10, 20)) / 864e5);
    await waitFor(() => expect(text()).toContain(`18x $${Math.round(40 * 1.1 * (1 + pace) ** ((days - 365) / 365) * 18).toLocaleString('en-US')}`), { timeout: 3000 });
    fireEvent.click(scope);
    await waitFor(() => expect(text()).toContain('→ Nov 2027'), { timeout: 3000 });
    // he writes his own year: the rungs run to the end of it; a year out of range is pulled to the nearest allowed
    const box = screen.getByRole('textbox', { name: 'Look ahead to year' });
    expect(box.value).toBe('2030'); // the next year ending in 0 or 5
    fireEvent.change(box, { target: { value: '203' } });
    expect(text()).toContain('→ Nov 2027'); // nothing moves until the year is whole
    fireEvent.change(box, { target: { value: '2035' } });
    const to2035 = Math.round((Date.UTC(2035, 11, 31) - Date.UTC(2026, 10, 20)) / 864e5);
    await waitFor(() => expect(text()).toContain(`18x $${Math.round(40 * 1.1 * (1 + pace) ** ((to2035 - 365) / 365) * 18).toLocaleString('en-US')}`), { timeout: 3000 });
    expect(screen.getByRole('button', { name: 'Look ahead to 2035' }).getAttribute('aria-pressed')).toBe('true');
    fireEvent.change(box, { target: { value: '2099' } });
    expect(box.value).toBe('2056');
    fireEvent.change(box, { target: { value: '2020' } });
    expect(box.value).toBe('2028');
});

test('under 5 years of forward earnings on file: no 🔭', () => {
    show([row('2026-11-20', 10)]);
    expect(screen.queryByRole('button', { name: /Look ahead/ })).toBeNull();
});
