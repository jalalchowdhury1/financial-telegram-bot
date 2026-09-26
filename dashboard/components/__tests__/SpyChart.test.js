import { render, screen, fireEvent } from '@testing-library/react';
import SpyChart from '../SpyChart';

if (!window.PointerEvent) {
    window.PointerEvent = class PointerEvent extends MouseEvent {
        constructor(type, params = {}) { super(type, params); this.pointerType = params.pointerType || 'mouse'; }
    };
}

// n daily bars ending 2026-09-25, price p0 + i*step
const bars = (n, p0, step = 0.1) => {
    const out = [];
    const end = Date.UTC(2026, 8, 25);
    for (let i = 0; i < n; i++) {
        const d = new Date(end - (n - 1 - i) * 864e5).toISOString().slice(0, 10);
        const p = p0 + i * step;
        out.push({ date: d, price: p, ma50: p, ma200: p });
    }
    return out;
};

beforeEach(() => window.localStorage.clear());

test('14 months of bars (the Polygon path) cannot pose as a 5Y chart', () => {
    render(<SpyChart chartHistory={bars(302, 600)} current={630} />);
    expect(screen.getByRole('button', { name: '1Y' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: '5Y' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '10Y' })).toBeDisabled();
    // the active tab fell back to ALL
    expect(screen.getByRole('button', { name: 'ALL' }).style.color).toBe('rgb(56, 189, 248)');
    expect(document.querySelector('.chart-note')).toBeNull();
});

test('index-point history (Sheet path) is labelled as index points, not dollars', () => {
    const { container } = render(<SpyChart chartHistory={bars(1500, 6000, 1)} current={771.35} />);
    const labels = [...container.querySelectorAll('svg text')].map((t) => t.textContent);
    expect(labels.some((l) => l.startsWith('$'))).toBe(false);
    expect(container.querySelector('.chart-note').textContent).toMatch(/S&P 500 index points/);
});

test('tap readout shows date and SPY price; the choice is remembered', () => {
    const { container, unmount } = render(<SpyChart chartHistory={bars(1500, 400)} current={549.9} />);
    fireEvent.click(screen.getByRole('button', { name: '1Y' }));
    expect(window.localStorage.getItem('ftb:tf:spy')).toBe('1Y');
    const svg = container.querySelector('svg');
    svg.getBoundingClientRect = () => ({ left: 0, width: 540, top: 0, height: 220 });
    fireEvent.pointerDown(svg, { clientX: 532, pointerType: 'touch' });
    expect(container.querySelector('.chart-readout').textContent).toBe('Sep 25, 2026 · $549.90');
    unmount();
    render(<SpyChart chartHistory={bars(1500, 400)} current={549.9} />);
    expect(screen.getByRole('button', { name: '1Y' }).style.color).toBe('rgb(56, 189, 248)');
});
