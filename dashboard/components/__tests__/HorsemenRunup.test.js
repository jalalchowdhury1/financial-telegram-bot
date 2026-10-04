import { render, screen } from '@testing-library/react';
import RunupBars from '../HorsemenRunup';
import live from '../../lib/__tests__/fixtures/fred-horsemen-2026-10-04.json';

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
        expect(screen.getByTestId('fh-row-claims').textContent).toMatch(/median of 2\)/);
    });

    test('renders nothing rather than throwing when a series is missing', () => {
        const { container } = render(<RunupBars fred={{ recessions: [], horsemen: {} }} />);
        expect(container).toBeTruthy();
    });

    // Real /api/fred slices saved 2026-10-04.
    const NOW = Date.UTC(2026, 9, 4);
    test('live data: each change is the latest print vs a year before it (matches the header)', () => {
        render(<RunupBars fred={live} now={NOW} />);
        expect(screen.getByTestId('fh-row-unemployment').textContent).toMatch(/−0\.2pp vs 1y/);
        expect(screen.getByTestId('fh-row-bankruptcies').textContent).toMatch(/\+17% vs 1y/);
        expect(screen.getByTestId('fh-row-claims').textContent).toMatch(/−12% vs 1y/);
    });

    test('live data: the curve reads as the one 2022–2024 inversion, not a September 2024 blip', () => {
        render(<RunupBars fred={live} now={NOW} />);
        expect(screen.getByTestId('fh-row-spread').textContent).toMatch(/inverted 2022–2024 · tell fired 25 months ago/);
    });

    test('notes are never cut: no nowrap/ellipsis, and the row layout lives in CSS (so phones can restack it)', () => {
        render(<RunupBars fred={live} now={NOW} />);
        for (const row of screen.getAllByTestId(/^fh-row-/)) {
            expect(row).toHaveClass('fh-row');
            expect(row.style.gridTemplateColumns).toBe('');
            const note = row.querySelector('.fh-note');
            expect(note).not.toBeNull();
            expect(note.style.whiteSpace).toBe('');
            expect(note.style.textOverflow).toBe('');
        }
    });

    test('the footer count uses the badge wording when the card passes it', () => {
        render(<RunupBars fred={live} now={NOW} ridingNote="Riding = test: 1 of 4 riding." />);
        expect(screen.getByText(/1 of 4 riding\./)).toBeInTheDocument();
        expect(screen.queryByText(/moving the wrong way/)).not.toBeInTheDocument();
    });
});
