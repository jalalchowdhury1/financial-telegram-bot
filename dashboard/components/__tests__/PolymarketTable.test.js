import { render, screen, act } from '@testing-library/react';
import PolymarketTable from '../PolymarketTable';

const board = { macro: [{ title: 'US recession by end of 2026?', odds: 0.23 }] };

afterEach(() => { delete global.fetch; jest.useRealTimers(); });

it('a failed refresh keeps the last odds, tagged 🕐 with when they landed (audit D7)', async () => {
    global.fetch = jest.fn(async () => ({ status: 200, json: async () => board }));
    const { rerender } = render(<PolymarketTable refreshKey={0} />);
    expect(await screen.findByText(/US recession/)).toBeInTheDocument();
    expect(document.querySelector('.pm-card')).not.toHaveAttribute('data-cached');

    global.fetch = jest.fn(() => Promise.reject(new TypeError('Failed to fetch')));
    jest.useFakeTimers();
    rerender(<PolymarketTable refreshKey={1} bust />);
    await act(async () => { await jest.advanceTimersByTimeAsync(2000); });
    expect(screen.getByText(/US recession/)).toBeInTheDocument();
    expect(document.querySelector('.pm-card').getAttribute('data-cached')).toMatch(/^\d\d:\d\d$/);
});
