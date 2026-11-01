/**
 * The v2 Rubber band card (components/RiskCards.js): its badge says what its headline says. In a defensive
 * mode the tripwires can all be green again while the plan says "Half in cash. Wait."; the badge used to say
 * "All clear" right above that.
 */
import { render, screen } from '@testing-library/react';
import { RubberBandCard } from '../RiskCards';

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
