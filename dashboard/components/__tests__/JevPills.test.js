import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import JevPills from '../JevPills';

const factors = {
    regime: {
        summary: 'Score 2 → risk-on (risk-on needs ≥ 2, risk-off ≤ −1)',
        rows: [
            { label: 'SPY vs 200-day avg', value: '+6.3%', test: '> 0 → +1, else −1', hit: true, effect: '+1' },
            { label: 'Fear & Greed',       value: '65',    test: '≥ 50 → +1 · < 30 → −1 · else 0', hit: true, effect: '+1' },
            { label: 'HYG/LQD 20d',        value: '+2.1%', test: '> 0 → +1 · < −1 → −1 · else 0', hit: true, effect: '+1' },
        ],
    },
    recession: {
        summary: 'First rule that fires wins; none fired → low',
        rows: [
            { label: 'Sahm rule',               value: '0.16',   test: '≥ 0.5 → high',                            hit: false, effect: '' },
            { label: 'Yield curve (2s10s) + claims', value: '0.36 · 221k', test: '< 0 and claims ≥ 260k → high', hit: false, effect: '' },
            { label: 'Sahm rule',               value: '0.16',   test: '≥ 0.2 → rising',                         hit: false, effect: '' },
            { label: 'Yield curve (2s10s)',      value: '0.36',   test: '< 0 → rising',                           hit: false, effect: '' },
            { label: 'Jobless claims',           value: '221k',   test: '≥ 260k → rising',                        hit: false, effect: '' },
            { label: 'NFCI',                     value: '-0.02',  test: '> 0 → rising',                            hit: false, effect: '' },
        ],
    },
    breadth: {
        summary: 'Broad participation: RSP and IWM both positive over 20d',
        rows: [
            { label: 'RSP/SPY 20d',        value: '+1.2%',  test: '< −1.5% with RSP vs 50d < 0 → rolling-over · > 0 with IWM > 0 → broad', hit: true, effect: 'broad' },
            { label: 'RSP/SPY vs 50d avg', value: '+0.8%',  test: '< 0 (with 20d < −1.5%) → rolling-over',                                 hit: false, effect: '' },
            { label: 'IWM/SPY 20d',        value: '+0.5%',  test: '> 0 (with RSP 20d > 0) → broad',                                         hit: true, effect: 'broad' },
        ],
    },
    hedging: {
        summary: 'Hedging is fair',
        rows: [
            { label: 'IV percentile (1y)', value: '34', test: '> 70 → expensive · < 20 (with VRP < 6) → cheap', hit: false, effect: '' },
            { label: 'VRP',                value: '4.8', test: '> 10 → expensive · < 6 (with IV pct < 20) → cheap',  hit: false, effect: '' },
        ],
    },
    conflict: {
        summary: '0 of 4 pairs diverge → aligned',
        rows: [
            { label: 'sentiment vs price', value: 'F&G 65 · SPY +6.3% vs 200d',     test: 'F&G < 35 with SPY > +3% · or F&G > 65 with SPY < −3%', hit: false, effect: '' },
            { label: '2s10s vs 3m10y',    value: '2s10s 0.36 · 3m10y -0.15',       test: 'Opposite signs between 2s10s and 3m10y',               hit: false, effect: '' },
            { label: 'credit vs equities', value: 'SPY +6.3% vs 200d · HYG/LQD +2.1%', test: 'SPY > 0% vs 200d with HYG/LQD 20d < −1.5%',       hit: false, effect: '' },
            { label: 'breadth vs index',   value: 'SPY +3.2% from 52w high · RSP/SPY +1.2%', test: 'SPY within 2% of high with RSP/SPY 20d < −1.5%', hit: false, effect: '' },
        ],
    },
};

const sampleData = {
    enabled: true,
    mode: 'on',
    asOf: '2026-09-18 15:30 UTC',
    state: '1. SPY 761.69...',
    pills: {
        regime:    { verdict: 'risk-on',  p: 0.87, by: 'jev',  reason: 'SPY above 200d MA, F&G above 50, HYG/LQD positive → score 2', jev: { verdict: 'risk-on', p: 0.87 } },
        recession: { verdict: 'low',      p: 0.62, by: 'jev',  reason: 'Sahm rule 0.16, yield curve 0.36, claims 221K',                jev: { verdict: 'low', p: 0.62 } },
        breadth:   { verdict: 'broad',    p: 0.74, by: 'jev',  reason: 'RSP/SPY 20d positive, IWM/SPY 20d positive',                   jev: { verdict: 'broad', p: 0.74 } },
        hedging:   { verdict: 'fair',     p: 0.55, by: 'rule', reason: 'IV percentile 34, VRP 4.8',                                    jev: { verdict: 'fair', p: 0.55 } },
        conflict:  { verdict: 'aligned',  p: 0.91, by: 'jev',  reason: 'No divergences firing',                                        jev: { verdict: 'aligned', p: 0.91 } },
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
    factors,
    _meta: {
        jev: 'ok',
        sources: {
            spy: 'yfinance',
            fg: 'CNN Fear & Greed',
            vol: 'CBOE + live',
            fred: 'St. Louis Fed',
            breadth: 'Polygon',
        },
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

    test('renders 5 pills with friendly verdict labels', () => {
        render(<JevPills data={sampleData} loading={false} />);
        expect(screen.getByText('Regime')).toBeInTheDocument();
        expect(screen.getByText('Recession')).toBeInTheDocument();
        expect(screen.getByText('Breadth')).toBeInTheDocument();
        expect(screen.getByText('Hedges')).toBeInTheDocument();
        expect(screen.getByText('Conflict')).toBeInTheDocument();

        // Friendly verdict labels (capitalized)
        expect(screen.getByText('Risk-on')).toBeInTheDocument();
        expect(screen.getByText('Low')).toBeInTheDocument();
        expect(screen.getByText('Broad')).toBeInTheDocument();
        expect(screen.getByText('Fair')).toBeInTheDocument();
        expect(screen.getByText('Aligned')).toBeInTheDocument();
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
        expect(screen.getByText('Softening ↓')).toBeInTheDocument();
        const chip = screen.getByTitle(/Since yesterday: breadth narrow → broad/);
        expect(chip).toBeInTheDocument();
    });

    test('shows no baseline chip', () => {
        const noBaselineData = {
            ...sampleData,
            since: { date: null, direction: 'none', changed: [], noBaseline: true },
        };
        render(<JevPills data={noBaselineData} loading={false} />);
        expect(screen.getByText('No baseline yet')).toBeInTheDocument();
    });

    test('shows unchanged chip when nothing changed', () => {
        const noChangeData = {
            ...sampleData,
            since: { date: '2026-09-17', direction: 'none', changed: [], noBaseline: false },
        };
        render(<JevPills data={noChangeData} loading={false} />);
        expect(screen.getByText('Unchanged since yesterday')).toBeInTheDocument();
    });

    test('renders asOf and rules-only footer', () => {
        const rulesData = {
            ...sampleData,
            mode: 'rules',
        };
        render(<JevPills data={rulesData} loading={false} />);
        // asOf rendered (format depends on locale, just check it's in the document)
        expect(screen.getByText(/as of/)).toBeInTheDocument();
        expect(screen.getByText(/rules only/)).toBeInTheDocument();
    });

    test('loading shows skeleton', () => {
        const { container } = render(<JevPills data={null} loading={true} />);
        expect(screen.getByText('🧪 Jev Regime Pills')).toBeInTheDocument();
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
                conflict: { verdict: 'mild-divergence', p: 0.68, by: 'jev', reason: '1 pair firing', jev: { verdict: 'mild-divergence', p: 0.68 } },
            },
        };
        render(<JevPills data={conflictData} loading={false} />);
        const conflictPill = screen.getByText('Conflict').closest('.jev-pill');
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
        expect(screen.getByText(/rules only/)).toBeInTheDocument();
    });

    test('renders Jev source tag when p is missing', () => {
        const noPData = {
            ...sampleData,
            pills: {
                ...sampleData.pills,
                regime: { verdict: 'risk-on', by: 'jev', reason: 'x', jev: { verdict: 'risk-on', p: 0.87 } },
            },
        };
        render(<JevPills data={noPData} loading={false} />);
        expect(screen.getByText('Jev')).toBeInTheDocument();
    });

    // ── Modal tests ──

    test('clicking Regime pill opens dialog with factor rows and decided-by-rule line', () => {
        render(<JevPills data={sampleData} loading={false} />);
        fireEvent.click(screen.getByText('Regime'));
        const dialog = screen.getByRole('dialog');
        expect(dialog).toBeInTheDocument();
        // Decided by Jev (since by: 'jev' and p: 0.87)
        expect(screen.getByText(/Decided by Jev/)).toBeInTheDocument();
        // Factor row labels present (both desktop table + mobile cards render in jsdom)
        expect(screen.getAllByText('SPY vs 200-day avg').length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText('Fear & Greed').length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText('HYG/LQD 20d').length).toBeGreaterThanOrEqual(1);
        // Factor row values present
        expect(screen.getAllByText('+6.3%').length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText('65').length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText('+2.1%').length).toBeGreaterThanOrEqual(1);
        // Jev's view section present
        expect(screen.getByText(/Jev['\u2019]s view/)).toBeInTheDocument();
    });

    test('Escape key closes the dialog', () => {
        render(<JevPills data={sampleData} loading={false} />);
        fireEvent.click(screen.getByText('Regime'));
        expect(screen.getByRole('dialog')).toBeInTheDocument();
        fireEvent.keyDown(document, { key: 'Escape' });
        expect(screen.queryByRole('dialog')).toBeNull();
    });

    test('backdrop click closes the dialog', () => {
        render(<JevPills data={sampleData} loading={false} />);
        fireEvent.click(screen.getByText('Regime'));
        expect(screen.getByRole('dialog')).toBeInTheDocument();
        const backdrop = document.querySelector('.jev-modal-backdrop');
        fireEvent.click(backdrop);
        expect(screen.queryByRole('dialog')).toBeNull();
    });

    test('payload without factors shows "Inputs unavailable" in modal', () => {
        const noFactorsData = {
            ...sampleData,
            factors: undefined,
            pills: {
                ...sampleData.pills,
                regime: { verdict: 'risk-on', by: 'rule', reason: 'Score 2 → risk-on', jev: { verdict: 'risk-on', p: 0.87 } },
            },
        };
        render(<JevPills data={noFactorsData} loading={false} />);
        fireEvent.click(screen.getByText('Regime'));
        expect(screen.getByRole('dialog')).toBeInTheDocument();
        expect(screen.getByText('Inputs unavailable')).toBeInTheDocument();
    });

    test('Jev below-floor text renders when Jev verdict below 0.6 confidence and rule stands', () => {
        const belowFloorData = {
            ...sampleData,
            pills: {
                ...sampleData.pills,
                regime: { verdict: 'risk-on', by: 'rule', reason: 'Score 2 → risk-on', jev: { verdict: 'narrow', p: 0.32 } },
            },
        };
        render(<JevPills data={belowFloorData} loading={false} />);
        fireEvent.click(screen.getByText('Regime'));
        expect(screen.getByRole('dialog')).toBeInTheDocument();
        // The header says "Decided by the rule" (since by: 'rule')
        expect(screen.getByText('Decided by the rule')).toBeInTheDocument();
        // Jev's view section shows below-floor text
        expect(screen.getByText(/below the 0.6 confidence floor/)).toBeInTheDocument();
        expect(screen.getByText(/Jev said Narrow/)).toBeInTheDocument();
    });

    test('modal explains what the pill measures, each input, and data freshness', () => {
        const withMeta = { ...sampleData, _meta: { ...(sampleData._meta || {}), jev: 'ok', dataAsOf: { breadth: '2026-09-18' } } };
        render(<JevPills data={withMeta} loading={false} />);
        fireEvent.click(screen.getByRole('button', { name: /Regime/ }));
        const dlg = screen.getByRole('dialog');
        expect(dlg).toHaveTextContent('What this measures');
        expect(dlg).toHaveTextContent(/risk-on\) or "no-go"/);
        expect(dlg).toHaveTextContent(/Junk bonds \(HYG\) vs high-quality bonds/);
        expect(dlg).toHaveTextContent('Data through: ETF ratios 2026-09-18');
    });

    // ── Backups in use ──

    test('shows "Backups in use" line in Sources section when inputSources has non-default entries', () => {
        const withBackups = {
            ...sampleData,
            _meta: {
                ...sampleData._meta,
                inputSources: {
                    t10y3m: 'fredcsv',
                    nfci: 'fredcsv',
                    claims: 'horsemen',
                    sahm: 'fred-route',
                },
            },
        };
        render(<JevPills data={withBackups} loading={false} />);
        fireEvent.click(screen.getByText('Regime'));
        expect(screen.getByRole('dialog')).toHaveTextContent('Backups in use: t10y3m fredcsv · nfci fredcsv · claims horsemen');
    });

    test('does NOT show "Backups in use" line when all inputSources are fred-route', () => {
        const allDefault = {
            ...sampleData,
            _meta: {
                ...sampleData._meta,
                inputSources: {
                    t10y3m: 'fred-route',
                    nfci: 'fred-route',
                    claims: 'fred-route',
                    sahm: 'fred-route',
                },
            },
        };
        render(<JevPills data={allDefault} loading={false} />);
        fireEvent.click(screen.getByText('Regime'));
        expect(screen.getByRole('dialog')).not.toHaveTextContent('Backups in use');
    });

    test('does NOT show "Backups in use" when inputSources is missing', () => {
        render(<JevPills data={sampleData} loading={false} />);
        fireEvent.click(screen.getByText('Regime'));
        expect(screen.getByRole('dialog')).not.toHaveTextContent('Backups in use');
    });
});

describe('PILL_FEEDS lists every feed a pill reads', () => {
    const { PILL_FEEDS, FEED_NAMES } = require('../jevExplain');
    test('regime and conflict include SPY and Fear & Greed', () => {
        expect(PILL_FEEDS.regime).toEqual(expect.arrayContaining(['spy', 'fg', 'breadth']));
        expect(PILL_FEEDS.conflict).toEqual(expect.arrayContaining(['spy', 'fg', 'fred', 'breadth']));
    });
    test('every feed key has a friendly name', () => {
        for (const feeds of Object.values(PILL_FEEDS)) for (const f of feeds) expect(FEED_NAMES[f]).toBeTruthy();
    });
});
