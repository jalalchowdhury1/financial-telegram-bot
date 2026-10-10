import React from 'react';
import { render, screen } from '@testing-library/react';
import JevPills from '../JevPills';
import { assemblePills } from '../../lib/jevPills';

// Every feed dead → every pill must say "No data" in neutral grey, never a calm
// green "Low" / "Fair" / "Aligned" (the pre-2026-10-09 behavior: missing data read as calm).
describe('JevPills — unknown verdicts render as an explicit no-data state', () => {
    const data = assemblePills({ raw: {}, jevAnswers: null, yesterday: null, mode: 'rules' });

    test('each pill shows "No data" with the neutral grey badge', () => {
        render(<JevPills data={data} loading={false} />);
        const badges = screen.getAllByText('No data');
        expect(badges).toHaveLength(5);
        for (const b of badges) {
            expect(b.className).toContain('badge-gray');
            expect(b.className).not.toContain('badge-green');
        }
        expect(screen.queryByText('Low')).toBeNull();
        expect(screen.queryByText('Fair')).toBeNull();
        expect(screen.queryByText('Aligned')).toBeNull();
    });

    test('the why line names the missing inputs instead of a calm reason', () => {
        render(<JevPills data={data} loading={false} />);
        const whys = screen.getAllByText(/not enough data/i);
        expect(whys.length).toBe(5);
        expect(screen.queryByText(/no recession signals/i)).toBeNull();
        expect(screen.queryByText(/fair by default/i)).toBeNull();
    });
});
