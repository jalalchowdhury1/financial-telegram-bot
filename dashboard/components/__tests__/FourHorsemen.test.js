import React from 'react';
import { render, screen } from '@testing-library/react';
import FourHorsemen, { trendOf } from '../FourHorsemen';

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
            history: series(40, 91, (i) => 20000 + i * 150),         // quarterly, rising
        },
    },
};

describe('FourHorsemen (overlay)', () => {
    test('renders the overlay with all four series labeled and headline values', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        expect(screen.getByText(/Four Horsemen/)).toBeInTheDocument();
        // Each label appears in the stat chips AND as an inline SVG label on the line.
        for (const label of ['Initial Jobless Claims', 'Unemployment Rate', '10Y − 2Y Yield Spread', 'US Bankruptcies']) {
            expect(screen.getAllByText(label).length).toBeGreaterThanOrEqual(2);
        }
        expect(screen.getByText('221K')).toBeInTheDocument();
        expect(screen.getByText('4.20%')).toBeInTheDocument();
        expect(screen.getByText('+0.52%')).toBeInTheDocument();
        expect(screen.getByText('26K')).toBeInTheDocument();
    });

    test('draws direction notes on the lines (all mock series rise)', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        expect(screen.getAllByText(/trending up/).length).toBeGreaterThanOrEqual(3);
    });

    test('shows the riding count badge (claims + bankruptcies rising here)', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        expect(screen.getByText('2 of 4 riding')).toBeInTheDocument();
    });

    test('offers shared timeframe tabs', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        for (const tf of ['ALL', '20Y', '10Y', '5Y', '1Y']) {
            expect(screen.getByRole('button', { name: tf })).toBeInTheDocument();
        }
    });

    test('renders a skeleton while loading', () => {
        const { container } = render(<FourHorsemen fred={null} loading={true} />);
        expect(container.querySelector('.card')).toBeInTheDocument();
        expect(screen.queryByText('Initial Jobless Claims')).not.toBeInTheDocument();
    });

    test('bankruptcies N/A: chip shows N/A, the other three lines still draw', () => {
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
        // Claims still labeled on the chart (chip + svg) even with bankruptcies gone
        expect(screen.getAllByText('Initial Jobless Claims').length).toBeGreaterThanOrEqual(2);
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
});

describe('trendOf (12-month fitted trend on raw history)', () => {
    test('rising series → up', () => {
        expect(trendOf(series(20, 30, (i) => 100 + i * 10))).toBe('up');
    });
    test('falling series → down', () => {
        expect(trendOf(series(20, 30, (i) => 300 - i * 10))).toBe('down');
    });
    test('flat big-number series → flat (relative threshold)', () => {
        expect(trendOf(series(20, 30, () => 200000))).toBe('flat');
    });
    test('flat rate-like series → flat (absolute threshold)', () => {
        expect(trendOf(series(20, 30, () => 4.2))).toBe('flat');
    });
    test('quarterly cadence still yields a verdict (fit uses ~5 points/yr)', () => {
        expect(trendOf(series(8, 91, (i) => 20000 + i * 500))).toBe('up');
    });
    test('a fit beats endpoint noise: rising year with one final down-tick is still up', () => {
        const pts = series(52, 7, (i) => 200000 + i * 1000);
        pts[pts.length - 1].value = pts[pts.length - 2].value - 3000; // noisy last print
        expect(trendOf(pts)).toBe('up');
    });
    test('verdict is identical on thinned data (sampling cannot flip it)', () => {
        const full = series(365, 1, (i) => 1 + i * 0.005);
        const thinned = full.filter((_, i) => i % 7 === 0).concat([full[full.length - 1]]);
        expect(trendOf(full)).toBe(trendOf(thinned));
    });
    test('only the last 12 months count: an old collapse cannot mask a fresh rise', () => {
        // 2 years: first year high plateau, then a year of steady climbing off a low.
        const pts = series(104, 7, (i) => (i < 52 ? 500000 : 200000 + (i - 52) * 2000));
        expect(trendOf(pts)).toBe('up');
    });
    test('too little history → null', () => {
        expect(trendOf([{ date: '2026-01-01', value: 1 }])).toBeNull();
        expect(trendOf(null)).toBeNull();
    });
});
