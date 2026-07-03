import { render, screen } from '@testing-library/react';
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
