/**
 * The SPY card with FactSet's file unreachable for the whole visit (own file: loadFwd keeps its answer per page):
 * SPY's own 1-year+ low still opens the card, and the views FactSet feeds say they are out instead of loading forever.
 * The P/E view draws only beside a P/E number on the page (N/A or CAPE standing in: no chart).
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SpyChart from '../SpyChart';

const iso = (t) => new Date(t).toISOString().slice(0, 10);
const off = (a, b) => Math.round((Date.parse(b) - Date.parse(a)) / 864e5);
const live = (last) => Array.from({ length: 302 }, (_, i) => {
    const p = i === 301 ? last : 600 + i * 0.1;
    return { date: iso(Date.UTC(2026, 8, 25) - (301 - i) * 864e5), price: p, ma50: p, ma200: p };
});
const spyFile = { from: '2023-06-01', t: [...Array(off('2023-06-01', '2025-12-31')).keys()], v: Array(off('2023-06-01', '2025-12-31')).fill(600) };

beforeEach(() => {
    global.fetch = jest.fn((url) => (url === '/history/spy-daily.json' ? Promise.resolve({ ok: true, json: async () => spyFile }) : Promise.resolve({ ok: false, status: 503 })));
});

test('FactSet down: a SPY low still opens the card on Price, with why', async () => {
    render(<SpyChart chartHistory={live(500)} current={500} lenses={{ peNow: null }} />);
    await waitFor(() => expect(screen.getByText('📉 SPY lowest since Jun 2023')).toBeTruthy());
    expect(screen.getByRole('tab', { name: 'Price' }).getAttribute('aria-selected')).toBe('true');
});

test('FactSet down: the Fwd P/E view says so; no P/E number: the P/E view says so', async () => {
    render(<SpyChart chartHistory={live(620)} current={620} lenses={{ peNow: null }} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Fwd P/E' }));
    await waitFor(() => expect(screen.getByText('Forward P/E unavailable right now')).toBeTruthy());
    fireEvent.click(screen.getByRole('tab', { name: 'P/E' }));
    expect(screen.getByText('P/E unavailable right now')).toBeTruthy();
    expect(screen.getByRole('tab', { name: 'P/E' }).textContent).toBe('P/E'); // no number on its button either
});
