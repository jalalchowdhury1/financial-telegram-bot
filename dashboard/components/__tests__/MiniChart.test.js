import { render, screen, fireEvent } from '@testing-library/react';
import MiniChart from '../MiniChart';

// Ascending monthly history spanning `years` back from 2026-06.
const monthly = (years) => {
    const out = [];
    for (let i = years * 12 - 1; i >= 0; i--) {
        const d = new Date(Date.UTC(2026, 5 - i, 1));
        out.push({ date: d.toISOString().slice(0, 10), value: 100 + (years * 12 - i) });
    }
    return out;
};

test('cadence="monthly" shows the full 1Y-30Y tab row', () => {
    render(<MiniChart history={monthly(40)} cadence="monthly" />);
    for (const tf of ['1Y', '3Y', '5Y', '10Y', '20Y', '30Y', 'ALL']) {
        expect(screen.getByRole('button', { name: tf })).toBeInTheDocument();
    }
});

test('auto cadence still detects quarterly data (<500 points → 10Y/20Y/30Y/ALL)', () => {
    render(<MiniChart history={monthly(10)} />); // 120 points → quarterly mode
    for (const tf of ['10Y', '20Y', '30Y', 'ALL']) {
        expect(screen.getByRole('button', { name: tf })).toBeInTheDocument();
    }
    expect(screen.queryByRole('button', { name: '1Y' })).not.toBeInTheDocument();
});

test('auto cadence still detects daily data (≥500 points → 1Y/5Y/10Y/ALL)', () => {
    render(<MiniChart history={monthly(50)} />); // 600 points → daily mode
    for (const tf of ['1Y', '5Y', '10Y', 'ALL']) {
        expect(screen.getByRole('button', { name: tf })).toBeInTheDocument();
    }
    expect(screen.queryByRole('button', { name: '30Y' })).not.toBeInTheDocument();
});

// The SVG line must stay drawable on EVERY tab — a NaN coordinate renders as a
// blank chart. Exercise the real production shape (~945 monthly points, like the
// S&P EPS card) plus hostile shapes: minimum length, and a flat series (range 0).
const expectDrawableSvg = (container) => {
    const line = container.querySelector('polyline');
    const area = container.querySelector('polygon');
    expect(line).not.toBeNull();
    expect(line.getAttribute('points')).not.toMatch(/NaN|Infinity/);
    expect(area.getAttribute('points')).not.toMatch(/NaN|Infinity/);
};

test('every monthly tab renders NaN-free SVG on a 945-point history', () => {
    const { container } = render(<MiniChart history={monthly(79)} cadence="monthly" />); // ≈ 1947→today
    for (const tf of ['1Y', '3Y', '5Y', '10Y', '20Y', '30Y', 'ALL']) {
        fireEvent.click(screen.getByRole('button', { name: tf }));
        expectDrawableSvg(container);
    }
});

test('monthly mode survives a 2-point history on every tab', () => {
    const { container } = render(<MiniChart history={monthly(40).slice(-2)} cadence="monthly" />);
    for (const tf of ['1Y', '3Y', '5Y', '10Y', '20Y', '30Y', 'ALL']) {
        fireEvent.click(screen.getByRole('button', { name: tf }));
        expectDrawableSvg(container);
    }
});

test('a perfectly flat series (range 0) still renders', () => {
    const flat = monthly(10).map((h) => ({ ...h, value: 100 }));
    const { container } = render(<MiniChart history={flat} cadence="monthly" />);
    fireEvent.click(screen.getByRole('button', { name: '1Y' }));
    expectDrawableSvg(container);
});

test('returns null (no crash) below 2 points', () => {
    const one = monthly(1).slice(-1);
    expect(render(<MiniChart history={one} cadence="monthly" />).container.firstChild).toBeNull();
    expect(render(<MiniChart history={[]} cadence="monthly" />).container.firstChild).toBeNull();
    expect(render(<MiniChart history={undefined} cadence="monthly" />).container.firstChild).toBeNull();
});
