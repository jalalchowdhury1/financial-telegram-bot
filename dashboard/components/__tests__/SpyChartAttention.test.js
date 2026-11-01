/**
 * The SPY card opens on the view that needs a look (his 1 Nov 2026 "default view should be the one that should get my
 * immediate attention ... current view should probably be: Fwd P/E since its low"). Own file: loadFwd keeps the first
 * FactSet answer for the whole file.
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SpyChart from '../SpyChart';

const iso = (t) => new Date(t).toISOString().slice(0, 10);
const off = (a, b) => Math.round((Date.parse(b) - Date.parse(a)) / 864e5);
// a close every day to 2026-09-25, climbing 600 → 630, then `last` on the final day
const live = (last) => Array.from({ length: 302 }, (_, i) => {
    const p = i === 301 ? last : 600 + i * 0.1;
    return { date: iso(Date.UTC(2026, 8, 25) - (301 - i) * 864e5), price: p, ma50: p, ma200: p };
});
// SPY flat at 600 from Jun 2023 until the live window
const spyFile = { from: '2023-06-01', t: [...Array(off('2023-06-01', '2025-12-31')).keys()], v: Array(off('2023-06-01', '2025-12-31')).fill(600) };
// FactSet: 22x in Jan 2024, 21x in Jan 2025, 16x in Sep 2026
const dates = ['2024-01-05', '2025-01-03', '2026-09-18'];
const fwdFile = { from: dates[0], to: dates[2], t: dates.map((d) => off(dates[0], d)), v: [22, 21, 16], a5: [19.8, 19.8, 19.8], a10: [19.1, 19.1, 19.1] };

beforeEach(() => {
    global.fetch = jest.fn((url) => {
        const body = { '/history/spy-daily.json': spyFile, '/history/fwdPE.json': fwdFile }[url];
        return Promise.resolve(body ? { ok: true, json: async () => body } : { ok: false });
    });
});

const selected = (name) => screen.getByRole('tab', { name }).getAttribute('aria-selected');

test('opens on Fwd P/E at its lowest in years, with a dot on its button; a tap wins after that', async () => {
    const { rerender } = render(<SpyChart chartHistory={live(620)} current={620} lenses={{ peNow: 27.9 }} />);
    expect(selected('Price')).toBe('true'); // the price chart paints first
    await waitFor(() => expect(selected('Fwd P/E')).toBe('true'));
    const tab = screen.getByRole('tab', { name: 'Fwd P/E' });
    expect(tab.title).toBe('Lowest since Jan 2024: worth a look');
    expect(tab.textContent).toBe('Fwd P/E15.8x'); // its number now, beside a sparkline of its year
    expect(screen.getByRole('tab', { name: 'Price' }).textContent).toBe('Price$620');
    expect(screen.getByRole('tab', { name: 'P/E' }).textContent).toBe('P/E27.9x');
    expect(screen.getByRole('tab', { name: 'Earnings' }).textContent).toMatch(/^Earnings\+\d+%$/);
    fireEvent.click(screen.getByRole('tab', { name: 'Price' }));
    rerender(<SpyChart chartHistory={live(620)} current={620} lenses={{ peNow: 27.9 }} />); // fresh data: the pick runs again
    await waitFor(() => expect(selected('Price')).toBe('true'));
    expect(selected('Fwd P/E')).toBe('false');
});

test('SPY at its lowest in years beats a younger P/E low, and says why on the price chart', async () => {
    render(<SpyChart chartHistory={live(500)} current={500} lenses={{ peNow: null }} />);
    await waitFor(() => expect(screen.getByText('📉 SPY lowest since Jun 2023')).toBeTruthy());
    expect(selected('Price')).toBe('true');
    expect(screen.getByRole('tab', { name: 'Price' }).title).toBe('Lowest since Jun 2023: worth a look');
});

test('nothing unusual: Price, no dot', async () => {
    render(<SpyChart chartHistory={live(630.1)} current={630.1} lenses={{ peNow: null }} />);
    await waitFor(() => expect(screen.getByRole('tab', { name: 'Fwd P/E' }).textContent).toMatch(/x$/)); // FactSet loaded
    expect(selected('Price')).toBe('true');
    expect(screen.getAllByRole('tab').map((t) => t.title)).toEqual(['', '', '', '']);
});
