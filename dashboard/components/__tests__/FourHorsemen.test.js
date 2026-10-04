import React from 'react';
import { render, screen } from '@testing-library/react';
import FourHorsemen from '../FourHorsemen';
import live from '../../lib/__tests__/fixtures/fred-horsemen-2026-10-04.json';

// Ascending history with real spaced dates: n points, stepDays apart, ending 2026-07-20.
const series = (n, stepDays, fn) => {
    const end = new Date('2026-07-20T00:00:00Z').getTime();
    return Array.from({ length: n }, (_, i) => ({
        date: new Date(end - (n - 1 - i) * stepDays * 86400000).toISOString().slice(0, 10),
        value: fn(i),
    }));
};

const mockFred = {
    recessions: [{ start: '2020-02-01', end: '2020-04-01' }],
    yieldCurve: {
        current: 0.52, asOf: '2026-07-21', stale: false,
        history: series(1200, 1, (i) => -0.5 + i * 0.002),          // daily, rising
    },
    indicators: { sahmRule: { value: 0.13 } },
    horsemen: {
        claims: {
            current: 221000, asOf: '2026-07-11', stale: false, unavailable: false,
            history: series(300, 7, (i) => 200000 + i * 1000),       // weekly, rising >10%/yr
        },
        unemployment: {
            current: 4.2, asOf: '2026-06-01', stale: false, unavailable: false,
            history: series(60, 30, (i) => 3.5 + i * 0.02),          // monthly, rising
        },
        bankruptcies: {
            current: 25960, total: 591850, asOf: '2026-03-31', stale: false, unavailable: false,
            changePct: 11.4, status: 'rising', source: 'uscourts',
            // quarterly, rising 11.4% a year (agrees with changePct), ending at `current`
            history: series(40, 91, (i) => Math.round(25960 / 1.114 ** ((39 - i) / 4))),
        },
    },
};

describe('FourHorsemen (run-up bars)', () => {
    test('renders the current level of each horseman', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        expect(screen.getByText(/Four Horsemen/)).toBeInTheDocument();
        for (const label of ['Initial Jobless Claims', 'Unemployment Rate', '10Y − 2Y Yield Spread', 'US Bankruptcies']) {
            expect(screen.getAllByText(label).length).toBeGreaterThanOrEqual(1);
        }
        expect(screen.getByText('221K')).toBeInTheDocument();
        expect(screen.getByText('4.20%')).toBeInTheDocument();
        expect(screen.getByText('+0.52%')).toBeInTheDocument();
        expect(screen.getByText('26K')).toBeInTheDocument();
    });

    test('shows a run-up row for each horseman instead of the old overlay', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        expect(screen.getAllByTestId(/^fh-row-/)).toHaveLength(4);
        expect(screen.getByTestId('fh-row-spread')).toHaveAttribute('data-status', 'inversion');
    });

    test('shows the riding count badge (claims + bankruptcies rising here)', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        expect(screen.getByText('2 of 4 riding')).toBeInTheDocument();
    });

    test('no zoom tabs — the card no longer plots a time series', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        for (const tf of ['ALL', '20Y', '10Y', '5Y', '1Y']) {
            expect(screen.queryByRole('button', { name: tf })).not.toBeInTheDocument();
        }
    });

    test('renders a skeleton while loading', () => {
        const { container } = render(<FourHorsemen fred={null} loading={true} />);
        expect(container.querySelector('.card')).toBeInTheDocument();
        expect(screen.queryByText('Initial Jobless Claims')).not.toBeInTheDocument();
    });

    test('bankruptcies N/A: chip shows N/A, the other rows still render', () => {
        const fred = {
            ...mockFred,
            horsemen: {
                ...mockFred.horsemen,
                bankruptcies: { current: null, total: null, asOf: null, stale: false, unavailable: true, changePct: null, status: 'unknown', source: null, history: [], tried: [] },
            },
        };
        render(<FourHorsemen fred={fred} loading={false} />);
        expect(screen.getByText('N/A')).toBeInTheDocument();                 // bankruptcies chip
        expect(screen.getByText('221K')).toBeInTheDocument();                // claims chip alive
        // The bankruptcies row says so rather than vanishing, and the rest still render.
        expect(screen.getAllByTestId(/^fh-row-/)).toHaveLength(4);
        expect(screen.getByTestId('fh-row-bankruptcies').textContent).toMatch(/not enough history/);
    });

    test('everything unavailable → single card-level N/A state', () => {
        const empty = { current: null, asOf: null, stale: false, unavailable: true, history: [] };
        const fred = {
            recessions: [], yieldCurve: { current: null, history: [] }, indicators: {},
            horsemen: { claims: empty, unemployment: empty, bankruptcies: { ...empty, changePct: null } },
        };
        render(<FourHorsemen fred={fred} loading={false} />);
        expect(screen.getByText(/source busy/)).toBeInTheDocument();
    });

    test('stale bankruptcies value still shows in the chip, marked with the clock', () => {
        const fred = {
            ...mockFred,
            horsemen: {
                ...mockFred.horsemen,
                bankruptcies: { ...mockFred.horsemen.bankruptcies, stale: true, asOf: '2025-12-31' },
            },
        };
        render(<FourHorsemen fred={fred} loading={false} />);
        expect(screen.getByText(/🕐\s*26K/)).toBeInTheDocument();
        expect(screen.getByText(/Last data .*(stale)/)).toBeInTheDocument();
    });

    // Real /api/fred slices saved 2026-10-04: header and rail must print the same changes.
    test('live data: header chips and rail notes agree (one helper for both)', () => {
        render(<FourHorsemen fred={live} loading={false} />);
        expect(screen.getByText('▼ 0.2pp vs 1y')).toBeInTheDocument();        // was ▼ 0.1pp (13 months)
        expect(screen.getByTestId('fh-row-unemployment').textContent).toMatch(/−0\.2pp vs 1y/);
        expect(screen.getByText('▲ 16.9% vs 1y')).toBeInTheDocument();
        expect(screen.getByTestId('fh-row-bankruptcies').textContent).toMatch(/\+17% vs 1y/);   // was +12%
        expect(screen.getByText('▼ 12.4% vs 1y')).toBeInTheDocument();
        expect(screen.getByTestId('fh-row-claims').textContent).toMatch(/−12% vs 1y/);
    });

    test('live data: riding count unchanged (bankruptcies only) and the footer says it the same way', () => {
        render(<FourHorsemen fred={live} loading={false} />);
        expect(screen.getByText('1 of 4 riding')).toBeInTheDocument();
        expect(screen.getByText(/: 1 of 4 riding\./)).toBeInTheDocument();
        expect(screen.getByText('Rising')).toBeInTheDocument();                // bankruptcies
        expect(screen.getByText('Contained')).toBeInTheDocument();             // claims
    });
});
