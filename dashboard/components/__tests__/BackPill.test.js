/**
 * ↩ Back pill (components/BackPill.js): after a jump, one tap returns to where he was.
 * It fades by itself after 8 s, or once he scrolls more than a screen by hand.
 */
import { render, screen, fireEvent, act } from '@testing-library/react';
import BackPill, { BACK_PILL_MS } from '../BackPill';
import { rememberJump, clearJump, currentJump } from '../../lib/jumpBack';

const VH = 844;
const SECTIONS = [{ label: 'Economy', top: 3000 }, { label: 'Volatility', top: 5000 }];

// jsdom has no layout: an element's document top comes from data-top.
beforeEach(() => {
    jest.useFakeTimers();
    jest.spyOn(Element.prototype, 'getClientRects').mockImplementation(() => [{ top: 0 }]);
    jest.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function box() {
        const top = Number(this.dataset?.top || 0) - window.scrollY;
        return { top, bottom: top + 100, left: 0, right: 100, width: 100, height: 100 };
    });
    window.scrollTo = jest.fn();
    window.scrollY = 0;
    window.innerHeight = VH;
    delete window.matchMedia;
});
afterEach(() => { act(() => clearJump()); jest.useRealTimers(); jest.restoreAllMocks(); });

function Page() {
    return (
        <>
            <section data-jump="Economy" data-top="3000" data-testid="econ">e</section>
            <section data-jump="Volatility" data-top="5000">v</section>
            <BackPill />
        </>
    );
}
const pill = () => document.querySelector('.back-pill');
const scroll = (y) => act(() => { window.scrollY = y; window.dispatchEvent(new Event('scroll')); });
const jumpFrom = (y) => act(() => { window.scrollY = y; rememberJump({ y, vh: VH, sections: SECTIONS }); });
const land = (y) => { scroll(y); act(() => { jest.advanceTimersByTime(300); }); };

test('nothing shows before any jump', () => {
    render(<Page />);
    expect(pill()).toBeNull();
});

test('names the section he left; one tap goes back there even after cards landed above it', () => {
    render(<Page />);
    jumpFrom(3100);
    expect(pill()).toHaveClass('is-on');
    expect(screen.getByRole('button', { name: 'Back to Economy' })).toHaveTextContent('↩ Back to Economy');
    land(5000);
    // the Jev card painted above while he peeked: Economy moved down 392 px
    screen.getByTestId('econ').dataset.top = '3392';
    fireEvent.click(screen.getByRole('button', { name: 'Back to Economy' }));
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 3492, behavior: 'smooth' });
    expect(pill()).not.toHaveClass('is-on');
    expect(pill()).toHaveAttribute('aria-hidden', 'true');
    expect(pill()).toHaveAttribute('tabindex', '-1');
    expect(currentJump()).toBeNull();
});

test('two cards on one desk row with the centre in the gutter: the pill names the row, and goes back to it', () => {
    render(<Page />);
    act(() => {
        window.scrollY = 3040;
        rememberJump({ y: 3040, vh: 900, cx: 720, sections: [
            { label: 'Economy', top: 3000, left: 24, right: 708 },
            { label: 'Volatility', top: 3001, left: 732, right: 1416 },
        ] });
    });
    expect(screen.getByRole('button', { name: 'Back to Economy · Volatility' })).toHaveTextContent('↩ Back to Economy · Volatility');
    fireEvent.click(pill());
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 3040, behavior: 'smooth' });
});

test('fades by itself after 8 s; a second jump starts the clock again', () => {
    render(<Page />);
    jumpFrom(3100);
    act(() => { jest.advanceTimersByTime(BACK_PILL_MS - 2000); });
    jumpFrom(5100);                                            // another jump from the Volatility card
    expect(screen.getByRole('button', { name: 'Back to Volatility' })).toBeInTheDocument();
    act(() => { jest.advanceTimersByTime(BACK_PILL_MS - 100); });
    expect(pill()).toHaveClass('is-on');
    act(() => { jest.advanceTimersByTime(200); });
    expect(pill()).not.toHaveClass('is-on');
    expect(BACK_PILL_MS).toBe(8000);
});

test('the flight to the card never dismisses it; a manual scroll of more than one screen does', () => {
    render(<Page />);
    jumpFrom(3100);
    scroll(3600); scroll(4400); scroll(5000);                 // the smooth scroll itself
    act(() => { jest.advanceTimersByTime(300); });           // landed at 5000
    expect(pill()).toHaveClass('is-on');
    scroll(5000 + VH - 10);                                    // reading on: under a screen
    expect(pill()).toHaveClass('is-on');
    scroll(5000 + VH + 10);
    expect(pill()).not.toHaveClass('is-on');
});

test('a smooth scroll that starts late (a busy phone) still counts as the flight, not a manual scroll', () => {
    render(<Page />);
    jumpFrom(0);
    act(() => { jest.advanceTimersByTime(600); });            // nothing moved yet
    scroll(2000); scroll(4000); scroll(5000);
    act(() => { jest.advanceTimersByTime(300); });
    expect(pill()).toHaveClass('is-on');
    scroll(5000 + VH + 10);                                    // now a real manual scroll
    expect(pill()).not.toHaveClass('is-on');
});

test('from the top it says "Back to top"; reduced motion jumps back instantly', () => {
    window.matchMedia = jest.fn(() => ({ matches: true }));
    render(<Page />);
    jumpFrom(0);
    land(5000);
    fireEvent.click(screen.getByRole('button', { name: 'Back to top' }));
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'auto' });
});
