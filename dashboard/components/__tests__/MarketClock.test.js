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

describe('next releases line', () => {
    const line = (container) => container.querySelector('.econ-line');

    test('a quiet "Next · …" line under the pill, ticking with the clock into amber on the day', () => {
        jest.useFakeTimers().setSystemTime(new Date('2026-10-13T15:00:00Z')); // Tue 11:00 ET, CPI tomorrow
        const { container } = render(<MarketClock />);
        expect(line(container).textContent).toBe('Next · CPI Wed 8:30 ET');
        expect(line(container).classList.contains('is-today')).toBe(false);
        act(() => { jest.setSystemTime(new Date('2026-10-14T11:20:00Z')); jest.advanceTimersByTime(CLOCK_TICK_MS); });
        expect(line(container).textContent).toBe('🔔 CPI today 8:30 ET · in 1h 10m · Next · FOMC Oct 28 2:00 ET');
        expect(line(container).classList.contains('is-today')).toBe(true);
        // only the release-day countdown is amber; the rest of the line stays quiet
        expect(line(container).querySelector('.econ-hot').textContent).toBe('🔔 CPI today 8:30 ET · in 1h 10m');
        act(() => { jest.setSystemTime(new Date('2026-10-14T12:31:00Z')); jest.advanceTimersByTime(CLOCK_TICK_MS); });
        expect(line(container).textContent).toBe('CPI out 8:30 · Next · FOMC Oct 28 2:00 ET');
        expect(line(container).classList.contains('is-today')).toBe(false);
    });

    test('one unbreakable piece per release; the " ·" joins it to the next, so a wrap only falls between releases', () => {
        jest.useFakeTimers().setSystemTime(new Date('2026-12-01T15:00:00Z')); // Tue 10:00 ET
        const { container } = render(<MarketClock />);
        const segs = [...line(container).querySelectorAll('.econ-seg')].map((n) => n.textContent);
        expect(segs).toEqual(['Next · Jobs Fri 8:30', 'FOMC Dec 9 2:00', 'CPI Dec 10 8:30 ET']);
        expect(line(container).querySelectorAll('.econ-sep')).toHaveLength(2);
        // the only break opportunities are the plain spaces between pieces, outside every nowrap span
        const between = [...line(container).childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent);
        expect(between).toEqual([' ', ' ']);
        expect(line(container).textContent).toBe('Next · Jobs Fri 8:30 · FOMC Dec 9 2:00 · CPI Dec 10 8:30 ET');
    });

    test('hidden when nothing is due within 14 days; the pill stays', () => {
        jest.useFakeTimers().setSystemTime(new Date('2026-12-11T15:00:00Z'));
        const { container } = render(<MarketClock />);
        expect(line(container)).toBeNull();
        expect(container.querySelector('.mkt-clock')).not.toBeNull();
    });

    test('the prerendered page carries no time-baked line (nothing until mounted)', () => {
        // react-dom/server needs TextEncoder, which jsdom lacks
        if (typeof global.TextEncoder === 'undefined') global.TextEncoder = require('util').TextEncoder;
        const { renderToString } = require('react-dom/server');
        jest.useFakeTimers().setSystemTime(new Date('2026-10-14T11:20:00Z'));
        expect(renderToString(<MarketClock />)).toBe('');
    });
});
