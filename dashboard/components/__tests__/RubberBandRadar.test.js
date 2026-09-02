import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import RubberBandRadar from '../RubberBandRadar';

const snap = {
    asOf: '2026-09-01', generatedAt: '2026-09-02T04:34:10',
    spec: { version: '1.0', dip: { rsi_below: 32, slow_n: 30, fast_n: 20, stop_after_red_days: 60 }, rip: { rsi_above: 79, n: 30, red_after_hot_days: 60 }, age: { amber_years: 3.3, red_years: 4.0 } },
    verdict: { colour: 'green', text: 'The rubber band is working: the last 30 dips paid +0.63% more than an ordinary day. All legs inside their lines.' },
    dials: {
        slow: { colour: 'green', n: 30, excess_pct: 0.634, hit: 0.633, se_pct: 0.539, span_years: 2.8, red_days: 0, stop_after: 60, first_event: '2023-10-27', last_event: '2026-07-29',
            crosscheck: { kind: '10-day drawdown > 6%', excess_pct: 0.496, agrees: true } },
        fast: { colour: 'amber', n: 20, excess_pct: -0.1, hit: 0.5, red_days: 12, stop_after: 60, look_only: true },
        age: { colour: 'green', years: 2.8, amber_years: 3.3, red_years: 4.0, events_last_12m: 8 },
        rip: { colour: 'green', n: 30, excess_pct: -0.068, hit: 0.567, hot_days: 0, red_after: 60 },
        machines: { colour: 'amber', reasons: ['C8-T is within 10 points of its line (-25% vs -31%)'], lag_months: 0, lag_pair: ['m1', 'C3'],
            legs: [
                { name: 'Main', line_pct: -40, dd_pct: -8.78, months_underwater: 1, worst_dd_pct: -51.83 },
                { name: 'C3', line_pct: -54, dd_pct: -8.18, months_underwater: 1, worst_dd_pct: -43.81 },
                { name: 'm1', line_pct: null, dd_pct: -8.14, months_underwater: 1, worst_dd_pct: -44.83 },
                { name: 'C8-T', line_pct: -31, dd_pct: -25.0, months_underwater: 1, worst_dd_pct: -20.34 },
            ] },
    },
    history: Array.from({ length: 120 }, (_, i) => ({ d: `2026-0${1 + Math.floor(i / 30)}-${String(1 + (i % 30)).padStart(2, '0')}`, slow: 0.5 - i * 0.001, fast: 0.4, rip: -0.1 })),
    _meta: { source: 'gist', stale: false, messages: [] },
};

const mockFetch = (body, ok = true) => {
    global.fetch = jest.fn(() => Promise.resolve({ ok, json: () => Promise.resolve(body) }));
};

describe('RubberBandRadar', () => {
    afterEach(() => { delete global.fetch; });

    test('renders five dials, the verdict, the machine legs and the as-of date', async () => {
        mockFetch(snap);
        render(<RubberBandRadar />);
        await waitFor(() => expect(screen.getByText(/rubber band is working/i)).toBeInTheDocument());
        expect(screen.getAllByTestId('rb-dial')).toHaveLength(5);
        expect(screen.getByTestId('rb-dial-slow').parentElement).toHaveTextContent('+0.63%');   // slow excess
        expect(screen.getByText('C8-T')).toBeInTheDocument();
        expect(screen.getByText(/within 10 points/)).toBeInTheDocument();
        expect(screen.getByText(/2026-09-01/)).toBeInTheDocument();
        expect(screen.getByTestId('rb-dial-fast')).toHaveAttribute('data-colour', 'amber');
        expect(screen.getByTestId('rb-dial-machines')).toHaveAttribute('data-colour', 'amber');
    });

    test('flags a stale snapshot', async () => {
        mockFetch({ ...snap, _meta: { source: 'gist', stale: true, messages: ['old'] } });
        render(<RubberBandRadar />);
        await waitFor(() => expect(screen.getAllByText(/stale/i).length).toBeGreaterThanOrEqual(1));
        expect(screen.getByText('Stale')).toHaveClass('badge-red');
    });

    test('never crashes on an unavailable payload', async () => {
        mockFetch({ _meta: { source: 'Unavailable', hasErrors: true, messages: ['boom'] } });
        render(<RubberBandRadar />);
        await waitFor(() => expect(screen.getByText(/unavailable/i)).toBeInTheDocument());
    });

    test('never crashes when fetch itself throws', async () => {
        global.fetch = jest.fn(() => Promise.reject(new Error('net down')));
        render(<RubberBandRadar />);
        await waitFor(() => expect(screen.getByText(/unavailable/i)).toBeInTheDocument());
    });
});
