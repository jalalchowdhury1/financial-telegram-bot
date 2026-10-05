import { render, screen, fireEvent } from '@testing-library/react';
import WhatMoved, { jumpToCard } from '../WhatMoved';

const alt = (n, a, b) => Array.from({ length: n }, (_, i) => ({ date: `2026-08-${String(i + 1).padStart(2, '0')}`, price: i % 2 ? b : a }));
const props = () => ({
    spy: { current: 102, dailyChange: { value: 2, pct: 2 }, chartHistory: [...alt(24, 100, 101), { date: '2026-09-25', price: 102 }] },
    fg: { score: 40, previousClose: 30, _meta: { source: 'CNN' } },
    vol: { vixDay: { value: 20, prev: 16, asOf: '2026-09-25', sigma: 5 } },
    extra: null,
    history: null,
});

beforeEach(() => {
    jest.useFakeTimers({ now: Date.parse('2026-09-26T16:00:00Z') });
    jest.spyOn(Element.prototype, 'getClientRects').mockImplementation(() => [{ top: 0 }]);
    Element.prototype.scrollIntoView = jest.fn();
});
afterEach(() => { jest.useRealTimers(); jest.restoreAllMocks(); });

test('shows the biggest moves first, ⚡ on the unusual ones', () => {
    render(<WhatMoved {...props()} />);
    const chips = screen.getAllByRole('button');
    expect(chips.map((c) => c.textContent)).toEqual(['⚡VIX+25.0%', '⚡F&G+10', '⚡SPY+2.0%']);
    expect(chips[0]).toHaveClass('is-big');
    expect(chips[0]).toHaveAttribute('aria-label', 'VIX 16 → 20 · 5.0× a normal day');
});

test('renders nothing when nothing can be ranked', () => {
    const { container } = render(<WhatMoved spy={null} fg={null} vol={null} extra={null} history={null} />);
    expect(container.firstChild).toBeNull();
});

test('a saved copy is labelled inline and on the strip', () => {
    const { container } = render(<WhatMoved {...props()} saved="10:42" />);
    expect(screen.getByText('🕐 10:42')).toBeInTheDocument();
    expect(container.firstChild).toHaveAttribute('data-cached', '10:42');
});

test('tapping a chip scrolls to its card and flashes it', () => {
    render(
        <>
            <div data-jump="Volatility" style={{ display: 'contents' }}><div className="card" data-testid="vol" /></div>
            <WhatMoved {...props()} />
        </>,
    );
    fireEvent.click(screen.getByRole('button', { name: /^VIX/ }));
    const card = screen.getByTestId('vol');
    expect(card.scrollIntoView).toHaveBeenCalled();
    expect(card).toHaveClass('jump-flash');
    jest.advanceTimersByTime(2000);
    expect(card).not.toHaveClass('jump-flash');
    expect(jumpToCard('Nope')).toBe(false);
});

test('↩ a chip jump remembers where he was (for the back pill); a missing card remembers nothing', () => {
    const { currentJump, clearJump } = require('../../lib/jumpBack');
    clearJump();
    window.scrollY = 3100;
    jest.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function box() {
        const top = Number(this.dataset?.top || 0) - window.scrollY;
        return { top, bottom: top + 100, left: 0, right: 100, width: 100, height: 100 };
    });
    render(
        <>
            <section data-jump="Economy" data-top="3000">e</section>
            <div data-jump="Volatility" style={{ display: 'contents' }}><div className="card" data-top="5000" /></div>
            <WhatMoved {...props()} />
        </>,
    );
    fireEvent.click(screen.getByRole('button', { name: /^VIX/ }));
    expect(currentJump()).toMatchObject({ label: 'Economy', offset: 100, y: 3100 });
    clearJump();
    expect(jumpToCard('Nope')).toBe(false);
    expect(currentJump()).toBeNull();
    window.scrollY = 0;
});

test('holds its line while feeds load, then collapses if nothing moved', () => {
    const empty = { spy: null, fg: null, vol: null, extra: null, history: null };
    const { container, rerender } = render(<WhatMoved {...empty} waiting />);
    expect(container.firstChild).toHaveClass('is-waiting');
    expect(screen.queryAllByRole('button')).toHaveLength(0);
    rerender(<WhatMoved {...empty} waiting={false} />);
    expect(container.firstChild).toBeNull();
});

test('says which session the moves are from: on a Saturday, "· Fri"', () => {
    const { container } = render(<WhatMoved {...props()} />);
    expect(container.querySelector('.moved-when').textContent).toBe(' · Fri');
    expect(container.firstChild).toHaveAttribute('aria-label', 'What moved on Fri');
});
