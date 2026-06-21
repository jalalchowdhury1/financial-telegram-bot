import { render, screen } from '@testing-library/react';
import BullChecklist from '../BullChecklist';

const mk = (over) => ({
    value: 1, asOf: '2026-04-01', stale: false, unavailable: false, staleDays: 0,
    bullish: true, status: 'good', label: 'Item', ...over,
});

test('stale item shows a clock icon + value (not N/A) and still counts in the score', () => {
    const fred = { checklist: {
        a: mk({ label: 'Fresh A', value: 5, bullish: true }),
        b: mk({ label: 'Stale B', value: 3, stale: true, staleDays: 2, bullish: true }),
    }};
    render(<BullChecklist fred={fred} loading={false} />);
    expect(screen.getByText('+3.0%')).toBeInTheDocument(); // stale value shown, not N/A
    expect(screen.getByText('🕐')).toBeInTheDocument();      // clock on the stale row
    expect(screen.getByText('✅')).toBeInTheDocument();      // fresh bullish row keeps its check
    // score counts the stale-bullish item (appears in both the header badge and score line)
    expect(screen.getAllByText(/2\/2/).length).toBeGreaterThanOrEqual(1);
});

test('unavailable item shows N/A with the ⚪ icon', () => {
    const fred = { checklist: {
        a: mk({ label: 'Gone', value: null, unavailable: true, stale: false, bullish: false }),
    }};
    render(<BullChecklist fred={fred} loading={false} />);
    expect(screen.getByText('N/A')).toBeInTheDocument();
    expect(screen.getByText('⚪')).toBeInTheDocument();
});
