import { render, screen, fireEvent } from '@testing-library/react';
import SinceLastVisit from '../SinceLastVisit';
import { mergeSeen, pickSeen } from '../../lib/lastVisit';

// Local noon: "5 hours ago" stays the same day, so the line reads "Since 07:00" in every
// time zone. On the real clock it read "Since Sat 22:30" (and failed) whenever CI ran
// between local midnight and 05:00. Only Date is faked.
jest.useFakeTimers({ now: new Date(2026, 9, 9, 12, 0), doNotFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'setImmediate', 'clearImmediate', 'queueMicrotask', 'nextTick', 'hrtime', 'performance', 'requestAnimationFrame', 'cancelAnimationFrame', 'requestIdleCallback', 'cancelIdleCallback'] });
afterAll(() => jest.useRealTimers());

const base = mergeSeen(null, pickSeen({ spy: { current: 770 }, vol: { vixDay: { value: 15 } } }, { claims: 200 }), Date.now() - 5 * 3600e3);
const live = pickSeen({ spy: { current: 780 }, vol: { vixDay: { value: 15 } } }, { claims: 205 });

test('shows what changed since the last visit, and ✕ hides it', () => {
    render(<SinceLastVisit base={base} live={live} />);
    const line = screen.getByRole('note');
    expect(line.textContent).toMatch(/👋 Since \d\d:\d\d/);
    expect(line.textContent).toMatch(/SPY \+1\.3%/);
    expect(line.textContent).toMatch(/VIX flat/);
    expect(line.textContent).toMatch(/🆕 Initial Claims \(4wk\)/);
    fireEvent.click(screen.getByRole('button', { name: /Hide/ }));
    expect(screen.queryByRole('note')).toBeNull();
});

test('renders nothing with no last visit or no live numbers yet', () => {
    const { container } = render(<SinceLastVisit base={null} live={live} />);
    expect(container.innerHTML).toBe('');
    const r2 = render(<SinceLastVisit base={base} live={pickSeen({})} />);
    expect(r2.container.innerHTML).toBe('');
});
