import { render, screen, fireEvent } from '@testing-library/react';
import Delta from '../Delta';

const printMark = {
    kind: 'print', value: 5.0, prev: 8.19, dir: -1,
    heldFrom: '2026-06-03', heldDays: 22, runs: [4.47, -2.71, 8.12, 8.19, 5.0],
};
const moveMark = {
    kind: 'move', value: 1.50, prev: 1.56, dir: -1, sigma: 0.02, move: -0.06,
    runs: [1.61, 1.58, 1.60, 1.55, 1.57, 1.56, 1.50],
};

beforeAll(() => {
    // jsdom has no IntersectionObserver; Delta falls back to marking `seen` immediately.
    global.IntersectionObserver = undefined;
});

describe('rendering', () => {
    test('an unmarked number renders as plain text with no glyph and no button role', () => {
        const { container } = render(<Delta mark={null}>5.0%</Delta>);
        expect(screen.getByText('5.0%')).toBeInTheDocument();
        expect(container.querySelector('.mark-glyph')).toBeNull();
        expect(screen.queryByRole('button')).toBeNull();
    });

    test('a print mark renders a dot', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        expect(container.querySelector('.mark-dot')).toBeInTheDocument();
        expect(container.querySelector('[data-mark="print"]')).toBeInTheDocument();
    });

    test('a move mark renders a directional chevron, not a dot', () => {
        const { container } = render(<Delta mark={moveMark}>1.50</Delta>);
        expect(container.querySelector('.mark-dot')).toBeNull();
        expect(container.querySelector('.mark-glyph').textContent).toBe('⌄');
        const up = render(<Delta mark={{ ...moveMark, dir: 1 }}>1.62</Delta>);
        expect(up.container.querySelector('.mark-glyph').textContent).toBe('⌃');
    });

    test('the current value is always rendered, marked or not', () => {
        render(<Delta mark={printMark}>5.0%</Delta>);
        expect(screen.getByText('5.0%')).toBeInTheDocument();
    });
});

describe('the reveal', () => {
    test('double-click opens exactly one popover, and it mounts on document.body', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(0);

        fireEvent.doubleClick(container.querySelector('[data-mark]'));
        const pops = document.querySelectorAll('.mark-pop');
        expect(pops).toHaveLength(1);

        // it must escape the card's stacking context, so it cannot be inside the trigger
        expect(container.contains(pops[0])).toBe(false);
        expect(document.body.contains(pops[0])).toBe(true);
    });

    test('shows the previous value, the delta, and how long it held', () => {
        const { container } = render(<Delta mark={printMark} format={(v) => `${v.toFixed(2)}%`}>5.0%</Delta>);
        fireEvent.doubleClick(container.querySelector('[data-mark]'));
        expect(screen.getByText('8.19%')).toBeInTheDocument();
        expect(screen.getByText(/held 22 days/)).toBeInTheDocument();
        expect(screen.getByText(/Jun 3/)).toBeInTheDocument();
        expect(screen.getByText(/▼/)).toBeInTheDocument();
    });

    test('a move mark explains itself as a 2 sigma move, not a print', () => {
        const { container } = render(<Delta mark={moveMark}>1.50</Delta>);
        fireEvent.doubleClick(container.querySelector('[data-mark]'));
        expect(screen.getByText('Yesterday')).toBeInTheDocument();
        expect(screen.getByText(/2σ of its own daily range/)).toBeInTheDocument();
    });

    test('clicking the dot opens it too', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        fireEvent.click(container.querySelector('.mark-glyph'));
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(1);
    });

    test('double-clicking again closes it', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        const trigger = container.querySelector('[data-mark]');
        fireEvent.doubleClick(trigger);
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(1);
        fireEvent.doubleClick(trigger);
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(0);
    });

    test('Escape closes it', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        fireEvent.doubleClick(container.querySelector('[data-mark]'));
        fireEvent.keyDown(document, { key: 'Escape' });
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(0);
    });

    test('a click outside closes it, a click inside does not', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        fireEvent.doubleClick(container.querySelector('[data-mark]'));
        fireEvent.mouseDown(document.querySelector('.mark-pop'));
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(1);
        fireEvent.mouseDown(document.body);
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(0);
    });

    test('keyboard: Enter opens it', () => {
        const { container } = render(<Delta mark={printMark}>5.0%</Delta>);
        fireEvent.keyDown(container.querySelector('[data-mark]'), { key: 'Enter' });
        expect(document.querySelectorAll('.mark-pop')).toHaveLength(1);
    });
});

describe('resilience', () => {
    test('omits the sparkline rather than breaking when there are too few points', () => {
        const { container } = render(<Delta mark={{ ...printMark, runs: [5.0] }}>5.0%</Delta>);
        fireEvent.doubleClick(container.querySelector('[data-mark]'));
        expect(document.querySelector('.mark-pop')).toBeInTheDocument();
        expect(document.querySelector('.mark-spark')).toBeNull();
    });

    test('renders with no runs array at all', () => {
        const { container } = render(<Delta mark={{ ...printMark, runs: undefined }}>5.0%</Delta>);
        expect(() => fireEvent.doubleClick(container.querySelector('[data-mark]'))).not.toThrow();
        expect(document.querySelector('.mark-pop')).toBeInTheDocument();
    });
});
