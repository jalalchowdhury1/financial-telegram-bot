import { render, screen, waitFor, within } from '@testing-library/react';
import VolMetricsTable, { curveSources, fmtDecay } from '../VolMetricsTable';

const row = (ticker, proxy, iv, rank, pctile, rv, vrp) => ({
    ticker, proxy, iv, ivRank1y: rank, ivPctile1y: pctile, rv21: rv, vrp, asOf: '2026-07-03',
});

const point = (tenor, index, value, source = 'cboe') => ({ tenor, index, value, asOf: '2026-09-25', source, live: false });
const calmCurve = {
    points: [point('9D', 'VIX9D', 12.76), point('1M', 'VIX', 14.87), point('3M', 'VIX3M', 17.93), point('6M', 'VIX6M', 20.01)],
    ratio: 0.829, state: 'calm', frontInverted: false, asOf: '2026-09-25', live: false, complete: true, stale: false,
};

const payload = {
    updated_at: '2026-07-03',
    tickers: [
        row('SPY', 'VIX', 15.8, 12.3, 24.6, 11.2, 4.6),
        row('QQQ', 'VXN', 27.1, 40.0, 55.0, 20.0, 7.1),
        row('TQQQ', '3×VXN', 81.3, 40.0, 55.0, 60.1, 21.2),
        row('SQQQ', '3×VXN', 81.3, 40.0, 55.0, 59.8, 21.5),
        row('UVXY', 'VVIX', null, null, null, 88.0, null),
    ],
    regime: {
        curve: calmCurve,
        decay: { leverage: 3, realizedVol: 15.8, impliedVol: 20.9, realized: 7.22, implied: 12.28 },
        moves: { days: 5, SPY: 2.1, QQQ: 2.94 },
    },
    _meta: { source: 'VIX:cboe', hasErrors: false, messages: [] },
};

const ok = (body) => jest.fn().mockResolvedValue({ status: 200, json: async () => body });
const withCurve = (curve) => ({ ...payload, regime: { ...payload.regime, curve } });

afterEach(() => {
    jest.restoreAllMocks();
});

it('shows only the SPY and QQQ rows', async () => {
    global.fetch = ok(payload);
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('SPY')).toBeInTheDocument());
    expect(screen.getByText('QQQ')).toBeInTheDocument();
    for (const t of ['TQQQ', 'SQQQ', 'UVXY']) expect(screen.queryByText(t)).not.toBeInTheDocument();
    expect(screen.getByText('15.8')).toBeInTheDocument();   // SPY IV
    expect(screen.queryByText('81.3')).not.toBeInTheDocument();
    expect(screen.getByText('+4.6')).toBeInTheDocument();    // positive VRP gets a plus sign
    expect(screen.getByText(/As of 2026-07-03/)).toBeInTheDocument();
    expect(screen.getByText(/SPY→VIX · QQQ→VXN\)/)).toBeInTheDocument();
    expect(global.fetch.mock.calls[0][0]).toBe('/api/vol');
});

it('renders the VIX curve, the call, TQQQ decay and the 5-day moves', async () => {
    global.fetch = ok(payload);
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('🟢 Calm')).toBeInTheDocument());
    const bars = within(screen.getByRole('list', { name: 'VIX term structure' }));
    expect(bars.getAllByRole('listitem').map((li) => li.textContent)).toEqual(['9D12.8', '1M14.9', '3M17.9', '6M20.0']);
    expect(screen.getByText('VIX ÷ VIX3M 0.83')).toBeInTheDocument();
    expect(screen.getByText(/≈7.2%\/yr/)).toBeInTheDocument();
    expect(screen.getByText(/≈12% at VXN/)).toBeInTheDocument();
    expect(screen.getByText('SPY ±2.1% · QQQ ±2.9%')).toBeInTheDocument();
    expect(screen.getByText('2026-09-25 close · CBOE')).toBeInTheDocument();
    expect(screen.queryByText(/9-day above 1-month/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Saved copy/)).not.toBeInTheDocument();
});

it('shows stress, the front inversion and a backup copy honestly', async () => {
    global.fetch = ok(withCurve({ ...calmCurve, ratio: 1.056, state: 'stress', frontInverted: true, stale: true, backup: 'KV 2026-09-25T21:00Z' }));
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('🔴 Stress')).toBeInTheDocument());
    expect(screen.getByText(/9-day above 1-month/)).toBeInTheDocument();
    expect(screen.getByText(/Saved copy from 2026-09-25/)).toBeInTheDocument();
    expect(screen.queryByText(/close · CBOE/)).not.toBeInTheDocument();
});

it('names the backup source when a fallback tier served a point', async () => {
    global.fetch = ok(withCurve({ ...calmCurve, live: true, points: calmCurve.points.map((p, i) => (i === 2 ? { ...p, source: 'fred' } : { ...p, source: 'cboe+live' })) }));
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('Intraday · CBOE · FRED')).toBeInTheDocument());
});

it('says the curve is unavailable but keeps the table when there is no call', async () => {
    global.fetch = ok(withCurve({ ...calmCurve, state: null, ratio: null }));
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText(/VIX curve unavailable right now/)).toBeInTheDocument());
    expect(screen.getByText('SPY')).toBeInTheDocument();
    expect(screen.getByText('SPY ±2.1% · QQQ ±2.9%')).toBeInTheDocument();
});

it('still renders the table from an old payload with no regime block', async () => {
    const { regime, ...old } = payload; // eslint-disable-line no-unused-vars
    global.fetch = ok(old);
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('SPY')).toBeInTheDocument());
    expect(screen.queryByText('VIX curve')).not.toBeInTheDocument(); // the section label, not the badge
});

it('shows the unavailable message when the fetch fails (after its one retry)', async () => {
    global.fetch = jest.fn().mockRejectedValue(new Error('boom'));
    render(<VolMetricsTable />);
    // getJson retries once after 1.5 s — past waitFor's 1 s default.
    await waitFor(() => expect(screen.getByText(/Volatility data unavailable/)).toBeInTheDocument(), { timeout: 3000 });
    expect(global.fetch).toHaveBeenCalledTimes(2);
});

it('shows the unavailable message on an empty payload', async () => {
    global.fetch = ok({ tickers: [] });
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText(/Volatility data unavailable/)).toBeInTheDocument());
});

it('skips a refresh inside 60 s, but a manual refresh busts the cache', async () => {
    global.fetch = ok(payload);
    const { rerender } = render(<VolMetricsTable refreshKey={0} />);
    await waitFor(() => expect(screen.getByText('SPY')).toBeInTheDocument());
    rerender(<VolMetricsTable refreshKey={1} />);
    expect(global.fetch).toHaveBeenCalledTimes(1);
    rerender(<VolMetricsTable refreshKey={2} bust />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
    expect(global.fetch.mock.calls[1][0]).toMatch(/^\/api\/vol\?_t=\d+$/);
});

it('keeps what it shows when a refresh fails', async () => {
    global.fetch = ok(payload);
    const { rerender } = render(<VolMetricsTable refreshKey={0} />);
    await waitFor(() => expect(screen.getByText('SPY')).toBeInTheDocument());
    global.fetch = jest.fn().mockRejectedValue(new Error('offline'));
    rerender(<VolMetricsTable refreshKey={1} bust />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2), { timeout: 3000 });
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.getByText('SPY')).toBeInTheDocument();
    expect(screen.queryByText(/Volatility data unavailable/)).not.toBeInTheDocument();
});

it('shows the intraday footnote with an ET timestamp when rows are live', async () => {
    const livePayload = {
        ...payload,
        updated_at: '2026-07-15',
        live_at: '2026-07-15T13:42:31.000-0400',
        tickers: payload.tickers.map((t) => ({ ...t, live: t.ticker !== 'UVXY', asOf: '2026-07-15' })),
    };
    global.fetch = ok(livePayload);
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('SPY')).toBeInTheDocument());
    expect(screen.getByText(/As of 2026-07-15, 1:42 PM ET · intraday/)).toBeInTheDocument();
});

it('falls back to the plain date when live rows have no usable timestamp', async () => {
    const p = { ...payload, updated_at: '2026-07-15', live_at: null, tickers: payload.tickers.map((t) => ({ ...t, live: true })) };
    global.fetch = ok(p);
    render(<VolMetricsTable />);
    await waitFor(() => expect(screen.getByText('SPY')).toBeInTheDocument());
    expect(screen.getByText(/As of 2026-07-15 · intraday/)).toBeInTheDocument();
});

it('formats decay and source labels', () => {
    expect(fmtDecay(7.224)).toBe('7.2%');
    expect(fmtDecay(12.28)).toBe('12%');
    expect(fmtDecay(null)).toBe('—');
    expect(curveSources([{ source: 'cboe' }, { source: 'cboe+live' }, { source: 'cnbc-quote' }])).toBe('CBOE · CNBC quote');
    expect(curveSources(null)).toBe('');
});

describe('instant open (saved copy)', () => {
    const { writeSnap } = require('../../lib/snapshot');
    test('paints the saved copy tagged 🕐, then the live answer drops the tag', async () => {
        writeSnap('vol', payload, Date.now() - 60e3);
        let resolve;
        global.fetch = jest.fn(() => new Promise((r) => { resolve = r; }));
        const { container } = render(<VolMetricsTable />);
        const card = container.querySelector('.card');
        expect(card).toHaveAttribute('data-cached');
        expect(screen.getByText('SPY')).toBeInTheDocument(); // no skeleton
        resolve({ status: 200, json: async () => payload });
        await waitFor(() => expect(card).not.toHaveAttribute('data-cached'));
    });
    test('a failed live fetch keeps the saved copy, still tagged', async () => {
        writeSnap('vol', payload, Date.now() - 60e3);
        global.fetch = jest.fn().mockRejectedValue(new Error('offline'));
        const { container } = render(<VolMetricsTable />);
        await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2), { timeout: 3000 });
        expect(container.querySelector('.card')).toHaveAttribute('data-cached');
        expect(screen.getByText('SPY')).toBeInTheDocument();
    });
});
