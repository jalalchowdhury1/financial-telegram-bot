import React from 'react';
import { render, screen } from '@testing-library/react';
import FourHorsemen from '../FourHorsemen';

const weekly = (n, v) => Array.from({ length: n }, (_, i) => ({
    date: `20${String(20 + Math.floor(i / 52)).padStart(2, '0')}-01-01`, value: v(i),
}));

const mockFred = {
    recessions: [{ start: '2020-02-01', end: '2020-04-01' }],
    yieldCurve: {
        current: 0.52, asOf: '2026-07-21', stale: false,
        history: Array.from({ length: 600 }, (_, i) => ({ date: `2024-01-${(i % 28) + 1}`, value: -0.5 + i * 0.002 })),
    },
    indicators: { sahmRule: { value: 0.13 } },
    horsemen: {
        claims: {
            current: 221000, asOf: '2026-07-11', stale: false, unavailable: false,
            history: weekly(120, (i) => 200000 + i * 500),
        },
        unemployment: {
            current: 4.2, asOf: '2026-06-01', stale: false, unavailable: false,
            history: Array.from({ length: 40 }, (_, i) => ({ date: `202${Math.floor(i / 12)}-0${(i % 12) < 9 ? (i % 12) + 1 : 9}-01`, value: 3.5 + i * 0.02 })),
        },
        bankruptcies: {
            current: 25960, total: 591850, asOf: '2026-03-31', stale: false, unavailable: false,
            changePct: 11.4, status: 'rising', source: 'uscourts',
            history: Array.from({ length: 40 }, (_, i) => ({ date: `20${16 + Math.floor(i / 4)}-03-31`, value: 20000 + i * 150, total: 450000 + i * 3000 })),
        },
    },
};

describe('FourHorsemen', () => {
    test('renders all four panels with headline values', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        expect(screen.getByText(/Four Horsemen/)).toBeInTheDocument();
        expect(screen.getByText('Initial Jobless Claims')).toBeInTheDocument();
        expect(screen.getByText('Unemployment Rate')).toBeInTheDocument();
        expect(screen.getByText('10Y − 2Y Yield Spread')).toBeInTheDocument();
        expect(screen.getByText('US Bankruptcies')).toBeInTheDocument();
        expect(screen.getByText('221K')).toBeInTheDocument();   // claims
        expect(screen.getByText('4.20%')).toBeInTheDocument();  // unemployment
        expect(screen.getByText('+0.52%')).toBeInTheDocument(); // spread
        // 25,960 → 26K (also appears as an axis tick, so allow multiple matches)
        expect(screen.getAllByText('26K').length).toBeGreaterThanOrEqual(1);
    });

    test('shows the riding count badge (claims + bankruptcies rising here)', () => {
        render(<FourHorsemen fred={mockFred} loading={false} />);
        // claims YoY > 10% and bankruptcies YoY > 10% → 2 of 4
        expect(screen.getByText('2 of 4 riding')).toBeInTheDocument();
    });

    test('renders a skeleton while loading', () => {
        const { container } = render(<FourHorsemen fred={null} loading={true} />);
        expect(container.querySelector('.card')).toBeInTheDocument();
        expect(screen.queryByText('Initial Jobless Claims')).not.toBeInTheDocument();
    });

    test('bankruptcies panel goes N/A when unavailable, others still render', () => {
        const fred = {
            ...mockFred,
            horsemen: {
                ...mockFred.horsemen,
                bankruptcies: { current: null, total: null, asOf: null, stale: false, unavailable: true, changePct: null, status: 'unknown', source: null, history: [], tried: [] },
            },
        };
        render(<FourHorsemen fred={fred} loading={false} />);
        expect(screen.getByText('US Bankruptcies')).toBeInTheDocument();
        expect(screen.getAllByText(/source busy/).length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText('221K')).toBeInTheDocument();
    });

    test('stale bankruptcies value still shows, marked with the clock', () => {
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
