import { render, screen, act } from '@testing-library/react';
import MarketClock, { CLOCK_TICK_MS } from '../MarketClock';

afterEach(() => jest.useRealTimers());

test('shows the NYSE state and keeps ticking', () => {
    jest.useFakeTimers().setSystemTime(new Date('2026-09-28T17:50:00Z')); // Mon 13:50 ET
    render(<MarketClock />);
    expect(screen.getByText('Open · closes in 2h 10m')).toBeInTheDocument();
    act(() => { jest.setSystemTime(new Date('2026-09-28T19:59:00Z')); jest.advanceTimersByTime(CLOCK_TICK_MS); });
    expect(screen.getByText('Open · closes in 1m')).toBeInTheDocument();
    act(() => { jest.setSystemTime(new Date('2026-09-28T20:01:00Z')); jest.advanceTimersByTime(CLOCK_TICK_MS); });
    expect(screen.getByText(/^Closed · opens in 17h 29m$/)).toBeInTheDocument();
});
