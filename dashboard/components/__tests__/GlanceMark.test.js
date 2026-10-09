import { render, screen, fireEvent, act } from '@testing-library/react';
import GlanceBar from '../GlanceBar';
import { resetMarkCursor } from '../MarkChip';

let counts = { print: 0, move: 0, total: 0 };
jest.mock('../MarkProvider', () => ({ useMarkCounts: () => counts }));

const spy = { current: 769.64, dailyChange: { value: 5.65, pct: 0.7395 } };
const fg = { score: 31.2, rating: 'fear' };

beforeEach(() => {
    resetMarkCursor();
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

it('shows ● N and each tap walks to the next marked number, wrapping round', () => {
    counts = { print: 2, move: 0, total: 2 };
    render(<Page />);
    const btn = screen.getByRole('button', { name: /Jump to the next changed number \(2 new prints\)/ });
    expect(btn.textContent).toBe('2');
    const seen = [];
    Element.prototype.scrollIntoView = jest.fn(function s() { seen.push(this.id); });
    act(() => { fireEvent.click(btn); fireEvent.click(btn); fireEvent.click(btn); });
    expect(seen).toEqual(['m1', 'm2', 'm1']);
});
