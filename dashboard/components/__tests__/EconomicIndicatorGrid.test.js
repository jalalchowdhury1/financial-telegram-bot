import { render, screen } from '@testing-library/react';
import EconomicIndicatorGrid from '../EconomicIndicatorGrid';

const statusColor = () => '';

// All six FRED indicator tiles present & fresh by default; override per test.
const baseInd = (over = {}) => ({
    sahmRule: { value: 0.2, asOf: '2026-05-01', stale: false, unavailable: false, staleDays: 0, status: 'safe' },
    sentiment: { value: 49.8, asOf: '2026-04-01', stale: false, unavailable: false, staleDays: 0, status: 'weak' },
    claims: { value: 223, asOf: '2026-06-13', stale: false, unavailable: false, staleDays: 0, status: 'healthy' },
    creditSpread: { value: 0.92, asOf: '2026-06-17', stale: false, unavailable: false, staleDays: 0, status: 'tight' },
    realYields: { value: 2.23, asOf: '2026-06-17', stale: false, unavailable: false, staleDays: 0, status: 'restrictive' },
    copperGold: { value: 1.5, asOf: '2026-06-20', stale: false, unavailable: false, status: 'rising', changePct: 1, changePct3mo: 2, copper: 5, gold: 3300, source: 'cnbc' },
    ...over,
});

const fredWith = (indOver) => ({ indicators: baseInd(indOver), peRatio: 32.2, peRatioAsOf: '2026-06-21T00:00:00Z' });

test('stale tile shows 🕐 + last value (not N/A)', () => {
    const fred = fredWith({ sentiment: { value: 49.8, asOf: '2026-04-01', stale: true, unavailable: false, staleDays: 2, status: 'weak' } });
    render(<EconomicIndicatorGrid fred={fred} loading={false} statusColor={statusColor} />);
    expect(screen.getByText('🕐 49.8')).toBeInTheDocument(); // value kept, clock-prefixed
    expect(screen.queryByText('N/A')).toBeNull();             // not nulled
});

test('unavailable tile shows N/A', () => {
    const fred = fredWith({ creditSpread: { value: null, asOf: null, stale: false, unavailable: true, staleDays: 0, status: 'normal' } });
    render(<EconomicIndicatorGrid fred={fred} loading={false} statusColor={statusColor} />);
    expect(screen.getByText('N/A')).toBeInTheDocument();
    expect(screen.queryByText(/🕐/)).toBeNull();              // no clock for unavailable
});

test('all-fresh grid has no clocks and no N/A', () => {
    render(<EconomicIndicatorGrid fred={fredWith({})} loading={false} statusColor={statusColor} />);
    expect(screen.queryByText(/🕐/)).toBeNull();
    expect(screen.queryByText('N/A')).toBeNull();
});
