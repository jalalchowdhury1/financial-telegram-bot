import { render, screen, fireEvent, act } from '@testing-library/react';
import GlanceBar from '../GlanceBar';
import { onOpenMarks } from '../MarkChip';

let counts = { print: 0, move: 0, total: 0 };
jest.mock('../MarkProvider', () => ({ useMarkCounts: () => counts }));

const spy = { current: 769.64, dailyChange: { value: 5.65, pct: 0.7395 } };
const fg = { score: 31.2, rating: 'fear' };

beforeEach(() => {
    jest.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function rect() {
        const b = this.classList?.contains('market-pulse') ? -5 : 0;
        return { top: b - 50, bottom: b, left: 0, right: 0, width: 0, height: 50 };
    });
    jest.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => { cb(); return 1; });
    Element.prototype.scrollIntoView = jest.fn();
});
afterEach(() => jest.restoreAllMocks());

const Page = () => (
    <>
        <div className="market-pulse">pulse</div>
        <div data-mark="print" id="m1">a</div>
        <div data-mark="move" id="m2">b</div>
        <GlanceBar spy={spy} spyDailyMove={{ value: '0.74%' }} fg={fg} fgColor={() => '#f90'} updatedAt={Date.now()} onRefresh={() => {}} markValues={{}} />
    </>
);

it('no dot when nothing is lit', () => {
    counts = { print: 0, move: 0, total: 0 };
    render(<Page />);
    expect(document.querySelector('.glance-mark')).toBeNull();
});

it('shows ● N and a tap opens the ● sheet (the same one the header chip opens)', () => {
    counts = { print: 2, move: 0, total: 2 };
    render(<Page />);
    const btn = screen.getByRole('button', { name: /See what changed \(2 new prints\)/ });
    expect(btn.textContent).toBe('2');
    const opened = jest.fn();
    const off = onOpenMarks(opened);
    act(() => { fireEvent.click(btn); });
    off();
    expect(opened).toHaveBeenCalledTimes(1);
    expect(Element.prototype.scrollIntoView).not.toHaveBeenCalled();
});

it('the bar is tagged has-mark only while ● N shows, so a phone can drop "Fri" and keep F&G whole', () => {
    counts = { print: 0, move: 0, total: 0 };
    const { unmount } = render(<Page />);
    expect(document.querySelector('.glance-bar')).not.toHaveClass('has-mark');
    unmount();
    counts = { print: 1, move: 0, total: 1 };
    render(<Page />);
    expect(document.querySelector('.glance-bar')).toHaveClass('has-mark');
});
