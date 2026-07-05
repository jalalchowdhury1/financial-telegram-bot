import { render, screen, waitFor } from '@testing-library/react';
import VolMetricsTable from '../VolMetricsTable';

const row = (ticker, proxy, iv, rank, pctile, rv, vrp) => ({
    ticker, proxy, iv, ivRank1y: rank, ivPctile1y: pctile, rv21: rv, vrp, asOf: '2026-07-03',
});

const payload = {
    updated_at: '2026-07-03',
    tickers: [
        row('SPY', 'VIX', 15.8, 12.3, 24.6, 11.2, 4.6),
        row('QQQ', 'VXN', 27.1, 40.0, 55.0, 20.0, 7.1),
        row('TQQQ', '3×VXN', 81.3, 40.0, 55.0, 60.1, 21.2),
        row('SQQQ', '3×VXN', 81.3, 40.0, 55.0, 59.8, 21.5),
        row('UVXY', 'VVIX', null, null, null, 88.0, null),
    ],
    _meta: { source: 'VIX:cboe', hasErrors: false, messages: [] },
};

afterEach(() => {
    jest.restoreAllMocks();
});

it('renders a row per ticker with formatted values', async () => {
    global.fetch = jest.fn().mockResolvedValue({ json: async () => payload });
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('SPY')).toBeInTheDocument());
    for (const t of ['SPY', 'QQQ', 'TQQQ', 'SQQQ', 'UVXY']) {
        expect(screen.getByText(t)).toBeInTheDocument();
    }
    expect(screen.getByText('15.8')).toBeInTheDocument();   // SPY IV
    expect(screen.getAllByText('81.3')).toHaveLength(2);     // TQQQ + SQQQ IV
    expect(screen.getByText('+4.6')).toBeInTheDocument();    // positive VRP gets a plus sign
    expect(screen.getAllByText('—').length).toBeGreaterThan(0); // UVXY nulls render as em-dash
    expect(screen.getByText(/As of 2026-07-03/)).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledWith('/api/vol');
});

it('shows the unavailable message when the fetch fails', async () => {
    global.fetch = jest.fn().mockRejectedValue(new Error('boom'));
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText(/Volatility data unavailable/)).toBeInTheDocument());
});

it('shows the unavailable message on an empty payload', async () => {
    global.fetch = jest.fn().mockResolvedValue({ json: async () => ({ tickers: [] }) });
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText(/Volatility data unavailable/)).toBeInTheDocument());
});
