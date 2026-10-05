import { render, screen, fireEvent, act } from '@testing-library/react';
import JumpNav, { collectSections, SHOW_AFTER_PX } from '../JumpNav';

// jsdom has no layout: give every element a box unless it is marked "hidden-box".
beforeEach(() => {
    jest.spyOn(Element.prototype, 'getClientRects').mockImplementation(function rects() {
        return this.dataset?.noBox ? [] : [{ top: 0 }];
    });
    Element.prototype.scrollIntoView = jest.fn();
    window.scrollTo = jest.fn();
    window.scrollY = 0;
});
afterEach(() => jest.restoreAllMocks());

function Page() {
    return (
        <>
            <section data-jump="Factors">f</section>
            <div data-jump="Economy" style={{ display: 'contents' }}><div className="card">econ</div></div>
            <div data-jump="Volatility" style={{ display: 'contents' }} />
            <div data-jump="Polymarket" data-no-box="1">hidden</div>
            <JumpNav />
        </>
    );
}

const scrollTo = (y) => act(() => { window.scrollY = y; window.dispatchEvent(new Event('scroll')); });

it('stays out of the way until you scroll past the first screen', () => {
    render(<Page />);
    expect(screen.queryByRole('button', { name: 'Jump to a section' })).toBeNull();
    scrollTo(SHOW_AFTER_PX + 1);
    expect(screen.getByRole('button', { name: 'Jump to a section' })).toBeInTheDocument();
});

it('lists only sections that rendered, and jumps to the right element', () => {
    render(<Page />);
    scrollTo(900);
    fireEvent.click(screen.getByRole('button', { name: 'Jump to a section' }));
    const nav = screen.getByRole('navigation', { name: 'Jump to section' });
    const labels = [...nav.querySelectorAll('button')].map((b) => b.textContent);
    // Volatility rendered nothing (empty contents wrapper); Polymarket has no box.
    expect(labels).toEqual(['↑ Top', 'Factors', 'Economy']);
    fireEvent.click(screen.getByRole('button', { name: 'Economy' }));
    const target = Element.prototype.scrollIntoView.mock.contexts[0];
    expect(target.className).toBe('card'); // the wrapper's child, not the contents wrapper
    expect(screen.queryByRole('navigation')).toBeNull();
});

it('↑ Top scrolls to the top; Esc and the backdrop close the menu', () => {
    const { container } = render(<Page />);
    scrollTo(900);
    fireEvent.click(screen.getByRole('button', { name: 'Jump to a section' }));
    fireEvent.click(screen.getByRole('button', { name: '↑ Top' }));
    expect(window.scrollTo).toHaveBeenCalledWith(expect.objectContaining({ top: 0 }));

    fireEvent.click(screen.getByRole('button', { name: 'Jump to a section' }));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('navigation')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Jump to a section' }));
    fireEvent.click(container.querySelector('.jump-backdrop'));
    expect(screen.queryByRole('navigation')).toBeNull();
});

it('↩ a menu jump and ↑ Top both remember where he was (for the back pill)', () => {
    const { currentJump, clearJump } = require('../../lib/jumpBack');
    clearJump();
    jest.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function box() {
        const top = Number(this.dataset?.top || 0) - window.scrollY;
        return { top, bottom: top + 100, left: 0, right: 100, width: 100, height: 100 };
    });
    render(
        <>
            <section data-jump="Factors" data-top="900">f</section>
            <div data-jump="Economy" style={{ display: 'contents' }}><div className="card" data-top="3000">econ</div></div>
            <JumpNav />
        </>,
    );
    scrollTo(3100);
    fireEvent.click(screen.getByRole('button', { name: 'Jump to a section' }));
    fireEvent.click(screen.getByRole('button', { name: 'Factors' }));
    expect(currentJump()).toMatchObject({ label: 'Economy', offset: 100, y: 3100 });

    scrollTo(1000);
    fireEvent.click(screen.getByRole('button', { name: 'Jump to a section' }));
    fireEvent.click(screen.getByRole('button', { name: '↑ Top' }));
    expect(currentJump()).toMatchObject({ label: 'Factors', offset: 100, y: 1000 });
    clearJump();
});

it('collectSections is safe with no document root', () => {
    expect(collectSections(null)).toEqual([]);
});
