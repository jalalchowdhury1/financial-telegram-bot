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

test('auto cadence still detects quarterly data (<500 points → 10Y/15Y/20Y/30Y/ALL)', () => {
    render(<MiniChart history={monthly(10)} />); // 120 points → quarterly mode
    for (const tf of ['10Y', '15Y', '20Y', '30Y', 'ALL']) {
        expect(screen.getByRole('button', { name: tf })).toBeInTheDocument();
    }
    expect(screen.queryByRole('button', { name: '1Y' })).not.toBeInTheDocument();
});

test('auto cadence still detects daily data (≥500 points → 1Y/5Y/10Y/15Y/20Y/30Y/ALL)', () => {
    render(<MiniChart history={monthly(50)} />); // 600 points → daily mode
    for (const tf of ['1Y', '5Y', '10Y', '15Y', '20Y', '30Y', 'ALL']) {
        expect(screen.getByRole('button', { name: tf })).toBeInTheDocument();
    }
    expect(screen.queryByRole('button', { name: '3Y' })).not.toBeInTheDocument(); // the monthly row's tab
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

// jsdom has no PointerEvent: without this, fireEvent.pointerDown drops clientX/pointerType.
if (!window.PointerEvent) {
    window.PointerEvent = class PointerEvent extends MouseEvent {
        constructor(type, params = {}) { super(type, params); this.pointerType = params.pointerType || 'mouse'; }
    };
}

describe('QoL: remembered tabs, honest tabs, tap readout', () => {
    beforeEach(() => window.localStorage.clear());

    test('the chosen timeframe is remembered per chart', () => {
        const { unmount } = render(<MiniChart history={monthly(40)} cadence="monthly" gradientId="epsGrad" />);
        fireEvent.click(screen.getByRole('button', { name: '10Y' }));
        expect(window.localStorage.getItem('ftb:tf:epsGrad')).toBe('10Y');
        unmount();
        render(<MiniChart history={monthly(40)} cadence="monthly" gradientId="epsGrad" />);
        expect(screen.getByRole('button', { name: '10Y' }).style.background).not.toMatch(/0\.05/);
    });

    test('tabs longer than the history are disabled', () => {
        render(<MiniChart history={monthly(4)} cadence="monthly" />);
        expect(screen.getByRole('button', { name: '1Y' })).not.toBeDisabled();
        expect(screen.getByRole('button', { name: '3Y' })).not.toBeDisabled();
        expect(screen.getByRole('button', { name: '5Y' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'ALL' })).not.toBeDisabled();
    });

    test('tapping the chart shows that point’s date and value', () => {
        const { container } = render(<MiniChart history={monthly(40)} cadence="monthly" defaultTimeframe="1Y" />);
        fireEvent.click(screen.getByRole('button', { name: '1Y' }));
        const svg = container.querySelector('svg');
        svg.getBoundingClientRect = () => ({ left: 0, width: 480, top: 0, height: 180 });
        // right edge of the plot = the newest point (2026-06-01, value 100 + 480)
        fireEvent.pointerDown(svg, { clientX: 472, pointerType: 'touch' });
        expect(container.querySelector('.chart-readout').textContent).toBe('Jun 1, 2026 · 580.00');
        expect(container.querySelector('.chart-cursor')).not.toBeNull();
        // touch readout survives the finger lifting; a mouse leaving clears it
        fireEvent.pointerLeave(svg, { pointerType: 'touch' });
        expect(container.querySelector('.chart-readout')).not.toBeNull();
        fireEvent.pointerLeave(svg, { pointerType: 'mouse' });
        expect(container.querySelector('.chart-readout')).toBeNull();
    });
});

describe('QoL: readable axis labels', () => {
    test('y ticks and years are HTML labels over the chart; taps still read the chart', () => {
        const { container } = render(<MiniChart history={monthly(40)} cadence="monthly" defaultTimeframe="10Y" />);
        fireEvent.click(screen.getByRole('button', { name: '10Y' }));
        expect(container.querySelectorAll('svg text').length).toBe(0);
        const texts = [...container.querySelectorAll('.axis-layer .axis-lbl')].map((l) => l.textContent);
        expect(texts.filter((t) => /^\d{4}$/.test(t)).length).toBeGreaterThanOrEqual(5); // years
        expect(texts.filter((t) => /^\d+\.\d+$/.test(t)).length).toBe(5); // 5 y ticks
        // the label layer sits beside the svg in one box and never takes the pointer
        const svg = container.querySelector('svg');
        expect(svg.parentElement.classList.contains('chart-plot')).toBe(true);
        svg.getBoundingClientRect = () => ({ left: 0, width: 480, top: 0, height: 180 });
        fireEvent.pointerDown(svg, { clientX: 472, pointerType: 'touch' });
        expect(container.querySelector('.chart-readout').textContent).toBe('Jun 1, 2026 · 580.00');
    });

    test('a custom fmt still formats the y ticks', () => {
        const big = monthly(10).map((h) => ({ ...h, value: h.value * 1000 }));
        const { container } = render(<MiniChart history={big} cadence="monthly" fmt={(v) => `${Math.round(v / 1000)}K`} />);
        const texts = [...container.querySelectorAll('.axis-lbl')].map((l) => l.textContent);
        expect(texts.some((t) => /^\d+K$/.test(t))).toBe(true);
    });
});

describe('v2: round y ticks and the inverted stretch', () => {
    const lbls = (c) => [...c.querySelectorAll('.axis-lbl')].map((l) => l.textContent).filter((t) => !/^\d{4}$/.test(t));
    // 5 years: +1.25 falling, under zero for months 12–35 (deepest −1.09 at month 26), then back up
    const curve = monthly(5).map((h, i) => ({ ...h, value: i < 12 ? 1.25 - i * 0.1 : i < 36 ? -1.08 * (1 - Math.abs(i - 26) / 15) - 0.01 : (i - 35) * 0.04 }));

    test('nice ticks land on round numbers (zero is 0.0, not 0.08); v1 keeps its five even splits', () => {
        const { container, unmount } = render(<MiniChart history={curve} cadence="monthly" defaultTimeframe="ALL" gradientId="tNice" nice />);
        expect(lbls(container)).toEqual(['-1.0', '-0.5', '0.0', '0.5', '1.0']);
        unmount();
        const eps = monthly(5).map((h, i) => ({ ...h, value: 195 + i * (103 / 59) }));
        const { container: c2, unmount: u2 } = render(<MiniChart history={eps} cadence="monthly" defaultTimeframe="ALL" gradientId="tEps" nice />);
        expect(lbls(c2)).toEqual(['200', '220', '240', '260', '280']);
        u2();
        const { container: c3 } = render(<MiniChart history={curve} cadence="monthly" defaultTimeframe="ALL" gradientId="tV1" />);
        expect(lbls(c3)).toEqual(['-1.09', '-0.51', '0.08', '0.66', '1.25']);
    });

    test('the inverted stretch is tinted and labelled at its deepest point; nothing on v1 or on a curve that never inverts', () => {
        const { container, unmount } = render(<MiniChart history={curve} cadence="monthly" defaultTimeframe="ALL" gradientId="tInv" showZero inverted="rgba(251,191,36,0.2)" />);
        expect(container.querySelector('clipPath#tInv-below')).toBeTruthy();
        expect(container.querySelector('polygon[clip-path="url(#tInv-below)"]').getAttribute('fill')).toBe('rgba(251,191,36,0.2)');
        const tag = screen.getByText('inverted');
        const x = (i) => ((42 + (i / 59) * 430) / 480) * 100; // toX in % (42 = the gutter before it is measured)
        expect(parseFloat(tag.style.left)).toBeCloseTo(x(26), 1);
        unmount();
        const { container: c2, unmount: u2 } = render(<MiniChart history={curve} cadence="monthly" defaultTimeframe="ALL" gradientId="tInvV1" showZero />);
        expect(c2.querySelector('clipPath')).toBeNull();
        expect(screen.queryByText('inverted')).toBeNull();
        u2();
        const { container: c3, unmount: u3 } = render(<MiniChart history={monthly(5)} cadence="monthly" defaultTimeframe="ALL" gradientId="tUp" inverted="red" />);
        expect(c3.querySelector('clipPath')).toBeNull();
        expect(screen.queryByText('inverted')).toBeNull();
        u3();
        // a 2px dip: tinted, but no room for the word
        const dip = curve.map((h) => ({ ...h, value: h.value < 0 ? -0.02 : h.value }));
        const { container: c4 } = render(<MiniChart history={dip} cadence="monthly" defaultTimeframe="ALL" gradientId="tDip" inverted="red" />);
        expect(c4.querySelector('clipPath')).toBeTruthy();
        expect(screen.queryByText('inverted')).toBeNull();
    });
});
