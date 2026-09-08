import { render, screen, fireEvent } from '@testing-library/react';
import RubberBandRadar from '../components/RubberBandRadar';

// A v1.1 snapshot the way scripts/rubber_band.py publishes it (trimmed to what the card reads).
const dip = (extra) => ({
    colour: 'green', n: 30, excess_pct: 0.63, se_pct: 0.54, hit: 0.63, first_event: '2023-10-27', last_event: '2026-07-29',
    red_days: 0, red_run: 0, window: 60, stop_after: 45, ...extra,
});
const fixture = {
    asOf: '2026-09-08',
    spec: {
        version: '1.1',
        dip: { rsi_below: 32, slow_n: 30, fast_n: 20, stop_of: 45, stop_window: 60 },
        rip: { rsi_above: 79, n: 30, hot_of: 45, hot_window: 60 },
        machines: { near_line_pts: 10, min_history_days: 250, record_amber_frac: 0.85, underwater_amber_frac: 0.75, fast_window_days: 20, book_drop_pct: -10 },
    },
    verdict: { colour: 'green', text: 'The rubber band is working.' },
    dials: {
        slow: dip({}),
        fast: dip({ n: 20, look_only: true }),
        age: { colour: 'green', years: 2.8, amber_years: 3.3, red_years: 4.0, events_last_12m: 8 },
        rip: { colour: 'green', n: 30, excess_pct: -0.07, se_pct: 0.18, hit: 0.57, hot_days: 0, hot_run: 0, window: 60, red_after: 45 },
        machines: {
            colour: 'green', reasons: [], hedge_check: 'book -4.3% / hedges -0.7% over 20d', lag_months: 0, lag_pair: ['m1', 'C3'], lag_is_info_only: true,
            legs: [
                { name: 'C3', role: 'core', dd_pct: -8.0, line_pct: -54, months_underwater: 2, worst_dd_prior_pct: -30, longest_underwater_prior_months: 14, days_before_peak: 3000 },
                { name: 'hedges', role: 'hedge', dd_pct: -15.6, line_pct: null, months_underwater: 6, worst_dd_prior_pct: -30.05, longest_underwater_prior_months: 102, days_before_peak: 3500 },
            ],
        },
    },
    history: [],
    defensive: {
        mode: 'INVESTED', since: null, last_asof: '2026-09-08', streak: { slow: 0, rip: 0, machines: 0 }, green_streak: 1, pending: null,
        rules: { fire_after: { slow: 1, rip: 1, machines: 5 }, reentry_closes: 10, reentry_edge_pct: 0.2 },
    },
    _meta: { stale: false },
};

beforeEach(() => {
    global.fetch = jest.fn(() => Promise.resolve({ json: () => Promise.resolve(fixture) }));
});

test('v1.1 card: tap opens one rule at a time, double-click opens, hedges leg and trigger mode show', async () => {
    render(<RubberBandRadar />);
    const slow = await screen.findByTestId('rb-dial-slow');
    expect(screen.queryByTestId('rb-explain-slow')).toBeNull();

    fireEvent.click(slow);                                        // tap → the rule appears
    expect((await screen.findByTestId('rb-explain-slow')).textContent).toMatch(/45 of the last 60/);
    fireEvent.click(slow);                                        // tap again → closes
    expect(screen.queryByTestId('rb-explain-slow')).toBeNull();
    fireEvent.doubleClick(slow);                                  // double-click → always open
    expect(screen.getByTestId('rb-explain-slow')).toBeTruthy();

    fireEvent.click(screen.getByTestId('rb-dial-machines'));      // one panel at a time
    expect(screen.queryByTestId('rb-explain-slow')).toBeNull();
    expect(screen.getByTestId('rb-explain-machines').textContent).toMatch(/hedges did not rise/);
    expect(screen.getByTestId('rb-explain-machines').textContent).toMatch(/C3 -54%/);

    expect(screen.getByText('hedges')).toBeTruthy();              // the new leg
    expect(screen.getByText('hedge')).toBeTruthy();               // its "line" cell
    expect(screen.getByTestId('rb-hedge-check').textContent).toMatch(/book -4.3% \/ hedges -0.7%/);
    expect(screen.getByText(/information only, not a rule/)).toBeTruthy();

    expect(screen.getByTestId('rb-mode').textContent).toBe('INVESTED');
    fireEvent.click(screen.getByText(/the rule in one breath/));
    expect(screen.getByTestId('rb-trigger-rule').textContent).toMatch(/5 closes in a row/);
    expect(screen.getByText(/rules v1.1/)).toBeTruthy();
});

test('a v1.0 snapshot without spec or defensive still renders with defaults', async () => {
    const { spec, defensive, ...old } = fixture;
    global.fetch = jest.fn(() => Promise.resolve({ json: () => Promise.resolve(old) }));
    render(<RubberBandRadar />);
    fireEvent.click(await screen.findByTestId('rb-dial-rip'));
    expect(screen.getByTestId('rb-explain-rip').textContent).toMatch(/45 of the last 60/);
    expect(screen.getByText(/state not published yet/)).toBeTruthy();
});
