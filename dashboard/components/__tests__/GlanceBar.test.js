import { render, screen, fireEvent, act } from '@testing-library/react';
import GlanceBar from '../GlanceBar';

const spy = { current: 769.64, dailyChange: { value: 5.65, pct: 0.7395 } };
const fg = { score: 31.2, rating: 'fear' };
const fgColor = () => 'rgb(249, 115, 22)';

// jsdom has no layout: the Market Pulse box sits wherever `pulseBottom` says.
let pulseBottom = 400;
beforeEach(() => {
    pulseBottom = 400;
    jest.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function rect() {
        const b = this.classList?.contains('market-pulse') ? pulseBottom : 0;
        return { top: b - 50, bottom: b, left: 0, right: 0, width: 0, height: 50 };
    });
    window.scrollTo = jest.fn();
    jest.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => { cb(); return 1; });
});
afterEach(() => jest.restoreAllMocks());

const Page = (props) => (
    <>
        <div className="market-pulse">pulse</div>
        <GlanceBar spy={spy} spyDailyMove={{ value: '0.74%' }} fg={fg} fgColor={fgColor} updatedAt={Date.now() - 3 * 60e3} onRefresh={() => {}} {...props} />
    </>
);
const scrollPast = (bottom) => act(() => { pulseBottom = bottom; window.dispatchEvent(new Event('scroll')); });
const bar = () => document.querySelector('.glance-bar');

it('stays hidden while Market Pulse is on screen, slides in once it scrolls off, hides again near the top', () => {
    render(<Page />);
    expect(bar()).not.toHaveClass('is-on');
    scrollPast(-5);
    expect(bar()).toHaveClass('is-on');
    expect(bar().textContent).toMatch(/^SPY 769\.64 ▲0\.74% · F&G 31 · 3 min ago$/);
    scrollPast(200);
    expect(bar()).not.toHaveClass('is-on');
});

it('↻ runs the page refresh; it is disabled and spins while busy', () => {
    const onRefresh = jest.fn();
    const { rerender } = render(<Page onRefresh={onRefresh} />);
    scrollPast(-5);
    fireEvent.click(screen.getByRole('button', { name: 'Refresh now' }));
    expect(onRefresh).toHaveBeenCalledTimes(1);
    rerender(<Page onRefresh={onRefresh} busy />);
    const btn = screen.getByRole('button', { name: 'Refresh now' });
    expect(btn).toBeDisabled();
    expect(btn.querySelector('svg')).toHaveClass('spinning');
});

it('tapping the numbers scrolls back to the top', () => {
    render(<Page />);
    scrollPast(-5);
    fireEvent.click(screen.getByRole('button', { name: /back to top/ }));
    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'smooth' });
});

it('a saved copy says so: "🕐 Saved 10:42" instead of an age', () => {
    render(<Page updatedAt={null} saved="10:42" loading />);
    scrollPast(-5);
    expect(bar().textContent).toMatch(/🕐 Saved 10:42/);
    expect(bar().textContent).not.toMatch(/ago/);
});

it('renders nothing without SPY or F&G, or with an errored payload', () => {
    const { rerender } = render(<Page spy={null} />);
    expect(bar()).toBeNull();
    rerender(<Page fg={{ error: 'down' }} />);
    expect(bar()).toBeNull();
    rerender(<Page spy={{}} />);
    expect(bar()).toBeNull();
});

it('F&G that is not a number is simply left out', () => {
    render(<Page fg={{ score: 'N/A' }} />);
    scrollPast(-5);
    expect(bar().textContent).not.toMatch(/F&G|NaN/);
    expect(bar().textContent).toMatch(/SPY 769\.64/);
});
