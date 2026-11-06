/**
 * The v2 Rubber band card (components/RiskCards.js): its badge says what its headline says. In a defensive
 * mode the tripwires can all be green again while the plan says "Half in cash. Wait."; the badge used to say
 * "All clear" right above that.
 */
import { render, screen } from '@testing-library/react';
import { RubberBandCard, RecessionCard } from '../RiskCards';

const RB = require('../../lib/__tests__/fixtures/pulse-rubber-band-2026-10-04.json');
const serve = (body) => { global.fetch = jest.fn(() => Promise.resolve({ json: async () => body })); };
const badge = () => document.querySelector('.rk .card-header .badge:last-child');

it.each([
    ['INVESTED', 'All clear', 'badge-green'],
    ['PENDING_DEFENSIVE', 'Go defensive', 'badge-red'],
    ['DEFENSIVE', 'Defensive', 'badge-yellow'],
    ['PENDING_REENTRY', 'Go back in', 'badge-blue'],
])('mode %s with every tripwire green: the badge says %s', async (mode, word, cls) => {
    serve({ ...RB, defensive: { ...RB.defensive, mode } });
    render(<RubberBandCard open={false} />);
    expect(await screen.findAllByText(word)).not.toHaveLength(0);
    expect(badge().textContent).toBe(word);
    expect(badge().className).toContain(cls);
});

describe('the page refresh reaches the Rubber band (audit D5)', () => {
    const { act } = require('@testing-library/react');
    const calls = () => global.fetch.mock.calls.filter(([u]) => String(u).startsWith('/api/rubber-band')).length;
    afterEach(() => jest.useRealTimers());

    it('a refresh tick asks /api/rubber-band again', async () => {
        serve(RB);
        const { rerender } = render(<RubberBandCard open={false} refreshKey={0} />);
        expect(await screen.findAllByText('All clear')).not.toHaveLength(0);
        rerender(<RubberBandCard open={false} refreshKey={1} bust />);
        await act(async () => {});
        expect(calls()).toBe(2);
    });

    it('a hung route ends in the card\'s error line, not an endless "…"', async () => {
        global.fetch = jest.fn((u, { signal } = {}) => new Promise((_, no) => signal?.addEventListener('abort', () => no(new Error('aborted')))));
        jest.useFakeTimers();
        render(<RubberBandCard open={false} refreshKey={0} />);
        await act(async () => { await jest.advanceTimersByTimeAsync(61000); });
        expect(badge().textContent).not.toBe('…');
        expect(document.querySelector('.rk-na')).not.toBeNull();
    });
});

it('Recession watch with a failed feed (no fred, done loading) says unavailable, not a skeleton forever', () => {
    render(<RecessionCard fred={null} loading={false} open={false} />);
    expect(screen.getByText(/Unavailable — source busy/)).toBeInTheDocument();
});

it('a missing year-ago print (UNRATE skipped Oct 2025) names the gap and keeps the red zone, not a bare line', () => {
    const months = Array.from({ length: 322 }, (_, i) => `${2000 + Math.floor(i / 12)}-${String((i % 12) + 1).padStart(2, '0')}-01`)
        .filter((d) => d !== '2025-10-01').map((date) => ({ date, value: date >= '2008-01-01' && date < '2009-07-01' ? 6 : 4.2 }));
    const fred = { horsemen: { unemployment: { current: 4.2, history: months } }, recessions: [{ start: '2008-01-01', end: '2009-07-01' }] };
    const { container } = render(<RecessionCard fred={fred} loading={false} open={false} />);
    expect(screen.getByText('no Oct 2025 print')).toBeInTheDocument();
    const meter = container.querySelectorAll('.rw-meter')[1];
    expect(meter.querySelector('b')).not.toBeNull();
    expect(meter.querySelector('i')).toBeNull();
});
