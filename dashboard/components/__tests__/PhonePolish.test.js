import { render, screen, act } from '@testing-library/react';
import { agoLabel, UpdatedAgo, OfflineBanner, PullToRefresh, PULL_TRIGGER_PX, TICK_MS } from '../PhonePolish';

const T0 = Date.parse('2026-09-26T16:00:00Z');

test('agoLabel', () => {
    expect(agoLabel(T0, T0 + 20e3)).toBe('just now');
    expect(agoLabel(T0, T0 + 3 * 60e3)).toBe('3 min ago');
    expect(agoLabel(T0, T0 + 125 * 60e3)).toBe('2 h ago');
    expect(agoLabel(T0, T0 + 50 * 3600e3)).toBe('2 d ago');
    expect(agoLabel(T0 + 5000, T0)).toBe('just now'); // clock skew never goes negative
});

describe('UpdatedAgo', () => {
    afterEach(() => jest.useRealTimers());
    test('ticks by itself and turns amber past 10 min', () => {
        jest.useFakeTimers({ now: T0 });
        render(<UpdatedAgo at={T0} />);
        expect(screen.getByText('just now')).not.toHaveClass('is-old');
        act(() => { jest.advanceTimersByTime(4 * 60e3 + TICK_MS); });
        expect(screen.getByText('4 min ago')).toBeInTheDocument();
        act(() => { jest.advanceTimersByTime(8 * 60e3); });
        expect(screen.getByText(/min ago/)).toHaveClass('is-old');
    });
    test('nothing before the first load', () => {
        const { container } = render(<UpdatedAgo at={null} />);
        expect(container.firstChild).toBeNull();
    });
});

describe('OfflineBanner', () => {
    const setOnline = (v) => Object.defineProperty(window.navigator, 'onLine', { configurable: true, get: () => v });
    afterEach(() => setOnline(true));
    test('shows while offline, refreshes on reconnect', () => {
        const onBack = jest.fn();
        render(<OfflineBanner onBack={onBack} since="10:42" />);
        expect(screen.queryByRole('status')).toBeNull();
        act(() => { setOnline(false); window.dispatchEvent(new Event('offline')); });
        expect(screen.getByRole('status')).toHaveTextContent('Offline — these numbers are from 10:42');
        act(() => { setOnline(true); window.dispatchEvent(new Event('online')); });
        expect(screen.queryByRole('status')).toBeNull();
        expect(onBack).toHaveBeenCalledTimes(1);
    });
    test('already offline at open', () => {
        setOnline(false);
        render(<OfflineBanner onBack={() => {}} />);
        expect(screen.getByRole('status')).toHaveTextContent('from your last visit');
    });
});

describe('PullToRefresh', () => {
    const touch = (type, x, y, target = document.body) => {
        const ev = new Event(type, { bubbles: true });
        if (x != null) Object.defineProperty(ev, 'touches', { value: [{ clientX: x, clientY: y }] });
        else Object.defineProperty(ev, 'touches', { value: [] });
        act(() => { target.dispatchEvent(ev); });
    };
    const pull = (dy, dx = 0, target) => {
        touch('touchstart', 100, 100, target);
        touch('touchmove', 100 + dx / 2, 100 + dy / 2, target);
        touch('touchmove', 100 + dx, 100 + dy, target);
    };
    beforeEach(() => { window.scrollY = 0; });

    test('a long pull at the top refreshes; the label follows the finger', () => {
        const onRefresh = jest.fn();
        const { rerender } = render(<PullToRefresh onRefresh={onRefresh} busy={false} />);
        touch('touchstart', 100, 100);
        touch('touchmove', 100, 140);
        expect(screen.getByRole('status')).toHaveTextContent('Pull to refresh');
        touch('touchmove', 100, 100 + 2 * PULL_TRIGGER_PX + 30);
        expect(screen.getByRole('status')).toHaveTextContent('Release to refresh');
        touch('touchend');
        expect(onRefresh).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('status')).toHaveTextContent('Refreshing');
        rerender(<PullToRefresh onRefresh={onRefresh} busy />);
        rerender(<PullToRefresh onRefresh={onRefresh} busy={false} />);
        expect(screen.queryByRole('status')).toBeNull();
    });

    test('short pulls, sideways swipes and scrolled pages do nothing', () => {
        const onRefresh = jest.fn();
        render(<PullToRefresh onRefresh={onRefresh} busy={false} />);
        pull(60); touch('touchend');
        pull(300, 400); touch('touchend');
        window.scrollY = 400;
        pull(300); touch('touchend');
        expect(onRefresh).not.toHaveBeenCalled();
        expect(screen.queryByRole('status')).toBeNull();
    });

    test('a pinch (second finger mid-gesture) or a zoomed-in page is not a pull', () => {
        const onRefresh = jest.fn();
        render(<PullToRefresh onRefresh={onRefresh} busy={false} />);
        touch('touchstart', 100, 100);
        touch('touchmove', 100, 140);
        const two = new Event('touchmove', { bubbles: true });
        Object.defineProperty(two, 'touches', { value: [{ clientX: 100, clientY: 400 }, { clientX: 200, clientY: 50 }] });
        act(() => { document.body.dispatchEvent(two); });
        touch('touchmove', 100, 400);
        touch('touchend');
        expect(onRefresh).not.toHaveBeenCalled();
        window.visualViewport = { scale: 2 };
        pull(300); touch('touchend');
        delete window.visualViewport;
        expect(onRefresh).not.toHaveBeenCalled();
    });

    test('a pull that starts inside a fixed overlay (modal, menu) is ignored', () => {
        const onRefresh = jest.fn();
        const modal = document.createElement('div');
        modal.style.position = 'fixed';
        const inner = document.createElement('p');
        modal.appendChild(inner);
        document.body.appendChild(modal);
        render(<PullToRefresh onRefresh={onRefresh} busy={false} />);
        pull(300, 0, inner); touch('touchend', null, null, inner);
        expect(onRefresh).not.toHaveBeenCalled();
        modal.remove();
    });
});
