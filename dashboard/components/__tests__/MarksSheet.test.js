import { render, screen, fireEvent, act } from '@testing-library/react';
import { MarkProvider } from '../MarkProvider';
import MarkChip from '../MarkChip';
import MarksSheet from '../MarksSheet';
import { currentJump, clearJump } from '../../lib/jumpBack';

// Sahm −0.03 → 0.00 is a far bigger change for its size than the mortgage rate's 6.30 → 6.27,
// so it must be listed first.
const history = {
    today: '2026-10-25',
    metrics: {
        mortgageRate: { kind: 'print', baseline: 6.30, dp: 2, heldFrom: '2026-10-16', runs: [6.3] },
        sahmRule: { kind: 'print', baseline: -0.03, dp: 2, heldFrom: '2026-10-03', runs: [-0.03] },
        claims: { kind: 'print', baseline: 225, dp: 0, heldFrom: '2026-10-18', runs: [225] },
    },
};
const values = { sahmRule: 0, mortgageRate: 6.27, claims: 225 }; // claims unchanged: no row

beforeEach(() => {
    Element.prototype.scrollIntoView = jest.fn();
    jest.useFakeTimers();
});
afterEach(() => { act(() => { jest.runOnlyPendingTimers(); }); jest.useRealTimers(); clearJump(); });

const Page = () => (
    <MarkProvider history={history}>
        <MarkChip values={values} />
        <section data-jump="Economy"><span data-mark="print" data-mark-key="sahmRule">0.00</span></section>
        <section data-jump="Markets"><span data-mark="print" data-mark-key="mortgageRate">6.27%</span></section>
        <MarksSheet values={values} />
    </MarkProvider>
);

const openSheet = () => fireEvent.click(screen.getByRole('button', { name: /2 new prints/ }));

it('the chip opens one sheet row per lit number, biggest change first: label · prev → now · card', () => {
    render(<Page />);
    expect(screen.queryByRole('dialog')).toBeNull();
    openSheet();
    const rows = screen.getAllByRole('button', { name: /→/ });
    expect(rows.map((r) => r.textContent)).toEqual([
        'Sahm Rule−0.03 → 0.00 · Economy',
        '30-Yr Mortgage Rate6.30 → 6.27 · Markets',
    ]);
    expect(document.documentElement).toHaveClass('sheet-open');
});

it('Esc, the × and a tap outside all close it', () => {
    render(<Page />);
    openSheet();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(document.documentElement).not.toHaveClass('sheet-open');
    openSheet();
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.queryByRole('dialog')).toBeNull();
    openSheet();
    fireEvent.click(document.querySelector('.marks-sheet-backdrop'));
    expect(screen.queryByRole('dialog')).toBeNull();
});

it('a row closes the sheet, then scrolls to that number, flashes it and arms the back pill', () => {
    render(<Page />);
    openSheet();
    const seen = [];
    Element.prototype.scrollIntoView = jest.fn(function s() { seen.push(this.getAttribute('data-mark-key')); });
    fireEvent.click(screen.getByRole('button', { name: /30-Yr Mortgage Rate/ }));
    expect(screen.queryByRole('dialog')).toBeNull();
    expect(document.documentElement).not.toHaveClass('sheet-open');
    expect(seen).toEqual(['mortgageRate']);
    expect(document.querySelector('[data-mark-key="mortgageRate"]')).toHaveClass('jump-flash');
    expect(currentJump()).not.toBeNull();
    act(() => { jest.advanceTimersByTime(2000); });
    expect(document.querySelector('[data-mark-key="mortgageRate"]')).not.toHaveClass('jump-flash');
});

it('one lit number is still a list of one, never a direct jump', () => {
    const one = { sahmRule: 0 };
    render(
        <MarkProvider history={history}>
            <MarkChip values={one} />
            <section data-jump="Economy"><span data-mark="print" data-mark-key="sahmRule">0.00</span></section>
            <MarksSheet values={one} />
        </MarkProvider>,
    );
    fireEvent.click(screen.getByRole('button', { name: /1 new print/ }));
    expect(Element.prototype.scrollIntoView).not.toHaveBeenCalled();
    expect(screen.getAllByRole('button', { name: /→/ })).toHaveLength(1);
});
