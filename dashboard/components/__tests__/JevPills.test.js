import React from 'react';
import { render, screen } from '@testing-library/react';
import JevPills from '../JevPills';

const sampleData = {
    enabled: true,
    mode: 'on',
    asOf: '2026-09-18 15:30 UTC',
    state: '1. SPY 761.69...',
    pills: {
        regime:    { verdict: 'risk-on',  p: 0.87, by: 'jev',  reason: 'SPY above 200d MA, F&G above 50, HYG/LQD positive → score 2' },
        recession: { verdict: 'low',      p: 0.62, by: 'jev',  reason: 'Sahm rule 0.16, yield curve 0.36, claims 221K' },
        breadth:   { verdict: 'broad',    p: 0.74, by: 'jev',  reason: 'RSP/SPY 20d positive, IWM/SPY 20d positive' },
        hedging:   { verdict: 'fair',     p: 0.55, by: 'rule', reason: 'IV percentile 34, VRP 4.8' },
        conflict:  { verdict: 'aligned',  p: 0.91, by: 'jev',  reason: 'No divergences firing' },
    },
    conflictPairs: [],
    since: {
        date: '2026-09-17',
        direction: 'softening',
        changed: [
            { pill: 'breadth', from: 'narrow', to: 'broad' },
        ],
        noBaseline: false,
    },
};

describe('JevPills', () => {
    test('renders nothing when data is null', () => {
        const { container } = render(<JevPills data={null} loading={false} />);
        expect(container.firstChild).toBeNull();
    });

    test('renders nothing when enabled is false', () => {
        const { container } = render(<JevPills data={{ enabled: false }} loading={false} />);
        expect(container.firstChild).toBeNull();
    });

    test('renders 5 pills from a sample payload', () => {
        render(<JevPills data={sampleData} loading={false} />);
        expect(screen.getByText('Regime')).toBeInTheDocument();
        expect(screen.getByText('Recession')).toBeInTheDocument();
        expect(screen.getByText('Breadth')).toBeInTheDocument();
        expect(screen.getByText('Hedges')).toBeInTheDocument();
        expect(screen.getByText('Conflict')).toBeInTheDocument();

        // Verdicts rendered
        expect(screen.getByText('risk-on')).toBeInTheDocument();
        expect(screen.getByText('low')).toBeInTheDocument();
        expect(screen.getByText('broad')).toBeInTheDocument();
        expect(screen.getByText('fair')).toBeInTheDocument();
        expect(screen.getByText('aligned')).toBeInTheDocument();
    });

    test('shows rule tag for rule-based verdict', () => {
        render(<JevPills data={sampleData} loading={false} />);
        expect(screen.getByText('rule')).toBeInTheDocument();
    });

    test('shows Jev tag for Jev-based verdict', () => {
        render(<JevPills data={sampleData} loading={false} />);
        expect(screen.getByText('Jev 0.87')).toBeInTheDocument();
    });

    test('shows since yesterday softening chip', () => {
        render(<JevPills data={sampleData} loading={false} />);
        expect(screen.getByText(/Since yesterday: softening/)).toBeInTheDocument();
        expect(screen.getByText(/breadth narrow → broad/)).toBeInTheDocument();
    });

    test('shows no baseline chip', () => {
        const noBaselineData = {
            ...sampleData,
            since: { date: null, direction: 'none', changed: [], noBaseline: true },
        };
        render(<JevPills data={noBaselineData} loading={false} />);
        expect(screen.getByText(/no baseline yet/)).toBeInTheDocument();
    });

    test('shows none chip when nothing changed', () => {
        const noChangeData = {
            ...sampleData,
            since: { date: '2026-09-17', direction: 'none', changed: [], noBaseline: false },
        };
        render(<JevPills data={noChangeData} loading={false} />);
        expect(screen.getByText(/Since yesterday: none/)).toBeInTheDocument();
    });

    test('renders asOf and mode:rules footer', () => {
        const rulesData = {
            ...sampleData,
            mode: 'rules',
        };
        render(<JevPills data={rulesData} loading={false} />);
        expect(screen.getByText(/as of 2026-09-18/)).toBeInTheDocument();
        expect(screen.getByText(/mode: rules/)).toBeInTheDocument();
    });

    test('loading shows skeleton', () => {
        const { container } = render(<JevPills data={null} loading={true} />);
        // Should render card with Jev Regime Pills heading
        expect(screen.getByText('🧪 Jev Regime Pills')).toBeInTheDocument();
        // Should have skeleton elements
        expect(container.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
    });

    test('conflict pill title lists pairs', () => {
        const conflictData = {
            ...sampleData,
            conflictPairs: [
                { pair: 'sentiment vs price', detail: 'fg 32 < 35, spy > 200MA 3%' },
                { pair: '2s10s vs 3m10y', detail: 'yield curve 0.36, t10y3m -0.15' },
            ],
            pills: {
                ...sampleData.pills,
                conflict: { verdict: 'mild-divergence', p: 0.68, by: 'jev', reason: '1 pair firing' },
            },
        };
        render(<JevPills data={conflictData} loading={false} />);
        const conflictPill = screen.getByText('Conflict').closest('.indicator-pill');
        expect(conflictPill).toHaveAttribute('title', 'sentiment vs price, 2s10s vs 3m10y');
    });

    test('renders rules-only fallback with pills null and no since/conflictPairs without throwing', () => {
        const rulesOnly = {
            enabled: true,
            mode: 'rules',
            pills: null,
            _meta: { jev: 'error: no data' },
        };
        const { container } = render(<JevPills data={rulesOnly} loading={false} />);
        expect(screen.getByText(/mode: rules/)).toBeInTheDocument();
    });

    test('renders Jev source tag when p is missing', () => {
        const noPData = {
            ...sampleData,
            pills: {
                ...sampleData.pills,
                regime: { verdict: 'risk-on', by: 'jev', reason: 'x' },
            },
        };
        render(<JevPills data={noPData} loading={false} />);
        expect(screen.getByText('Jev')).toBeInTheDocument();
    });
});