import { render, screen } from '@testing-library/react';
import RunupBars from '../HorsemenRunup';

const monthly = (start, n, fn) => Array.from({ length: n }, (_, i) => ({
    date: new Date(Date.UTC(start + Math.floor(i / 12), i % 12, 1)).toISOString().slice(0, 10),
    value: fn(i),
}));

// Claims fall 20% over the last year; unemployment flat; bankruptcies climb.
const fred = {
    recessions: [{ start: '2001-04-01', end: '2001-11-01' }, { start: '2008-01-01', end: '2009-06-01' }],
    horsemen: {
        claims: { current: 203000, asOf: '2026-08-22', history: monthly(1995, 380, (i) => (i < 368 ? 300000 : 240000)) },
        unemployment: { current: 4.1, asOf: '2026-07-01', history: monthly(1995, 380, () => 4.1) },
        bankruptcies: { current: 26941, asOf: '2026-06-30', history: monthly(1995, 380, (i) => 20000 + i * 20) },
    },
    yieldCurve: { current: 0.4, asOf: '2026-09-01', history: monthly(1995, 380, (i) => (i > 320 && i < 340 ? -0.5 : 0.4)) },
};

describe('RunupBars', () => {
    test('shows one row per horseman', () => {
        render(<RunupBars fred={fred} />);
        expect(screen.getAllByTestId(/^fh-row-/)).toHaveLength(4);
    });

    test('a horseman moving the healthy way reads as improving', () => {
        render(<RunupBars fred={fred} />);
        expect(screen.getByTestId('fh-row-claims')).toHaveAttribute('data-status', 'improving');
    });

    test('a horseman moving the wrong way but short of the pre-recession move is a watch', () => {
        render(<RunupBars fred={fred} />);
        expect(screen.getByTestId('fh-row-bankruptcies')).toHaveAttribute('data-status', 'watch');
    });

    test('the yield curve gets its own row about the inversion, not a run-up bar', () => {
        render(<RunupBars fred={fred} />);
        const row = screen.getByTestId('fh-row-spread');
        expect(row).toHaveAttribute('data-status', 'inversion');
        expect(row.textContent).toMatch(/inverted/i);
    });

    test('states how many recessions each comparison rests on', () => {
        render(<RunupBars fred={fred} />);
        expect(screen.getByTestId('fh-row-claims').textContent).toMatch(/2 recessions/);
    });

    test('renders nothing rather than throwing when a series is missing', () => {
        const { container } = render(<RunupBars fred={{ recessions: [], horsemen: {} }} />);
        expect(container).toBeTruthy();
    });
});
