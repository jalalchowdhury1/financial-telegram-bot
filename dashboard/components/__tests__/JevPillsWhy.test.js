import React from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import JevPills from '../JevPills';
import live from '../../lib/__tests__/fixtures/jev-pills-2026-10-04.json';

// The reason used to sit only in a hover `title` (never shown on an iPhone). Each pill now
// carries one quiet line built from its fired factor rows. Live /api/jev-pills, 2026-10-04.
describe('JevPills — one plain "why" line per pill', () => {
    const pill = (label) => screen.getByText(label).closest('button');

    test('each verdict has its line, from the live payload', () => {
        render(<JevPills data={live} loading={false} />);
        const why = (label) => pill(label).querySelector('.jev-pill-why')?.textContent;
        expect(why('Regime')).toBe('2 of 3 votes: uptrend + junk bonds firm');
        expect(why('Recession')).toBe('0 of 4 warnings tripped');
        expect(why('Breadth')).toBe('Average stock −4.3% vs SPY in 20 days · below its 50-day trend');
        expect(why('Hedges')).toBe('Options cheap: bottom 15% of the year');
        expect(why('Conflict')).toBe('Fearful crowd in an uptrend · near the high, average stock slipping');
    });

    test('the line sits inside the pill button, so a tap still opens the full sheet', () => {
        render(<JevPills data={live} loading={false} />);
        fireEvent.click(pill('Breadth').querySelector('.jev-pill-why'));
        expect(screen.getByRole('dialog')).toBeInTheDocument();
    });

    test('a pill whose rows are missing shows no line (and nothing throws)', () => {
        const data = { ...live, factors: { ...live.factors, breadth: undefined, hedging: { summary: 'x', rows: [] } } };
        render(<JevPills data={data} loading={false} />);
        expect(pill('Breadth').querySelector('.jev-pill-why')).toBeNull();
        expect(pill('Hedges').querySelector('.jev-pill-why')).toBeNull();
        expect(within(pill('Regime')).getByText('2 of 3 votes: uptrend + junk bonds firm')).toBeInTheDocument();
    });

    test('Jev overrides the rule with another verdict: the line is labelled as the rule', () => {
        const data = { ...live, pills: { ...live.pills, recession: { ...live.pills.recession, verdict: 'high', by: 'jev', p: 0.8 } } };
        render(<JevPills data={data} loading={false} />);
        expect(within(pill('Recession')).getByText('High')).toBeInTheDocument();
        expect(pill('Recession').querySelector('.jev-pill-why').textContent).toBe('Rule says low · 0 of 4 warnings tripped');
    });

    test('every input n/a (feed outage): no line reads as an all-clear', () => {
        const na = (f) => ({ ...f, rows: f.rows.map((r) => ({ ...r, value: 'n/a', hit: false, effect: '' })) });
        const factors = Object.fromEntries(Object.entries(live.factors).map(([k, f]) => [k, na(f)]));
        const { container } = render(<JevPills data={{ ...live, factors }} loading={false} />);
        expect(container.querySelectorAll('.jev-pill')).toHaveLength(5);
        expect(container.querySelector('.jev-pill-why')).toBeNull();
    });

    test('no factors at all (older payload): pills render exactly as before', () => {
        const { factors, ...noFactors } = live;
        const { container } = render(<JevPills data={noFactors} loading={false} />);
        expect(container.querySelectorAll('.jev-pill')).toHaveLength(5);
        expect(container.querySelector('.jev-pill-why')).toBeNull();
    });
});
