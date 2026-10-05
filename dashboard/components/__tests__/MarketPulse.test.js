/**
 * 📡 Market Pulse = a line of tappable verdict chips about the cards below the fold
 * (lib/pulseVerdicts.js). It no longer repeats SPY / F&G / RSI, which sit on screen already.
 */
import { render, screen, fireEvent, act } from '@testing-library/react';
import MarketPulse from '../MarketPulse';

const FRED = require('../../lib/__tests__/fixtures/pulse-fred-2026-10-04.json');
const VOL = require('../../lib/__tests__/fixtures/pulse-vol-2026-10-04.json');
const RB = require('../../lib/__tests__/fixtures/pulse-rubber-band-2026-10-04.json');

beforeEach(() => {
    jest.spyOn(Element.prototype, 'getClientRects').mockImplementation(() => [{ top: 0 }]);
    Element.prototype.scrollIntoView = jest.fn();
});
afterEach(() => { jest.useRealTimers(); jest.restoreAllMocks(); });

const chips = () => screen.getAllByRole('button');

test('one chip per verdict, coloured by meaning; no repeated SPY / F&G / RSI numbers', () => {
    const { container } = render(<MarketPulse fred={FRED} vol={VOL} rubberBand={RB} />);
    expect(chips().map((c) => c.textContent)).toEqual(['Dips pay ✓', 'Vol calm', 'Horsemen 1/4', 'Curve +0.45%', 'Bull 7/8']);
    expect(chips().map((c) => c.className)).toEqual([
        'pulse-chip tone-good', 'pulse-chip tone-good', 'pulse-chip tone-watch', 'pulse-chip tone-good', 'pulse-chip tone-good',
    ]);
    const line = container.querySelector('.market-pulse');
    expect(line).not.toBeNull();                          // the glance bar anchors on this class
    expect(line.textContent).not.toMatch(/SPY|F&G|RSI/);
    expect(chips()[2]).toHaveAttribute('aria-label', 'Horsemen 1/4 — Recession watch: 1 of 4 riding');
    expect(chips()[2].title).toBe('Recession watch: 1 of 4 riding · tap for the card');
});

test('tapping a chip scrolls to its card and flashes it', () => {
    jest.useFakeTimers();
    const card = document.createElement('div');
    card.setAttribute('data-jump', 'Volatility');
    document.body.appendChild(card);
    render(<MarketPulse fred={FRED} vol={VOL} rubberBand={RB} />);
    fireEvent.click(screen.getByRole('button', { name: /^Vol calm/ }));
    expect(card.scrollIntoView).toHaveBeenCalled();
    expect(card).toHaveClass('jump-flash');
    act(() => { jest.advanceTimersByTime(2000); });
    expect(card).not.toHaveClass('jump-flash');
    card.remove();
});

test('a saved copy or a stale source marks the chip, and says so', () => {
    const staleRb = { ...RB, _meta: { ...RB._meta, stale: true, ageDays: 6 } };
    render(<MarketPulse fred={FRED} vol={VOL} rubberBand={staleRb} saved={{ fred: '19:43' }} />);
    const [dips, vol, horse] = chips();
    expect(dips).toHaveClass('is-old');
    expect(dips.textContent).toBe('🕐Dips pay ✓');                 // stale at the source: 🕐 on the chip
    expect(dips.getAttribute('aria-label')).toMatch(/stale · 6 days old$/);
    expect(vol).not.toHaveClass('is-old');                          // vol is live here
    expect(horse).toHaveClass('is-old');                            // fred from the saved copy
    expect(horse.textContent).toBe('Horsemen 1/4');                 // the line's own 🕐 tag covers a saved copy
    expect(horse.getAttribute('aria-label')).toMatch(/saved copy 19:43$/);
});

test('holds its place while its feeds load; renders nothing once they fail', () => {
    const { container, rerender } = render(<MarketPulse fred={null} vol={null} rubberBand={undefined} waiting />);
    const line = container.querySelector('.market-pulse');
    expect(line).toHaveClass('is-waiting');
    expect(line).toHaveAttribute('aria-busy', 'true');
    expect(screen.queryAllByRole('button')).toHaveLength(0);
    rerender(<MarketPulse fred={{}} vol={{ tickers: [] }} rubberBand={null} waiting={false} />);
    expect(container.firstChild).toBeNull();
    rerender(<MarketPulse fred={{}} vol={VOL} rubberBand={null} waiting={false} />);
    expect(chips().map((c) => c.textContent)).toEqual(['Vol calm']);
});
