import { render, screen } from '@testing-library/react';
import ExtraMarketsGrid from '../ExtraMarketsGrid';
import live from '../../lib/__tests__/fixtures/market-extra-2026-10-04.json';

// Live /api/market-extra on 2026-10-04 (history trimmed to the last 5 bars per row).
const clone = () => JSON.parse(JSON.stringify(live));
const chg = (ticker) => screen.queryByTestId(`mkt-chg-${ticker}`)?.textContent ?? null;

describe('Global Markets: each change says which window it covers', () => {
    test.each([
        ['TNX', '−5bp · Thu'],
        ['T2Y', '−10bp · Thu'],
        ['MORT30', '+25bp · 1w'],
        ['MTGPMT', '+2.53% · 1w'],
        ['ZRI', '+0.23% · 1mo'],
        ['ATNHPI', '+1.87% · 3mo'],
        ['CL', '−0.09% · Sun'],
        ['GOLD', '−1.01% · Fri'],
        ['BTC', '+0.28% · Sat'],
        ['USD/CAD', '+0.19% · Fri'],
        ['USD/INR', '−0.10% · Fri'],
    ])('%s reads %s', (ticker, text) => {
        render(<ExtraMarketsGrid data={live} loading={false} />);
        expect(chg(ticker)).toBe(text);
    });

    test('the five once-a-day FX rows say "daily rate", never "live" or a fake +0.00%', () => {
        const { container } = render(<ExtraMarketsGrid data={live} loading={false} />);
        for (const t of ['USD/BDT', 'INR/BDT', 'CAD/INR', 'CAD/BDT', 'DXY']) {
            expect(chg(t)).toBeNull();
            expect(screen.getByTestId(`mkt-spot-${t}`).textContent).toBe('daily rate');
        }
        expect(container.textContent).not.toMatch(/\blive\b/);
        expect(container.textContent).not.toContain('+0.00%');
        expect(screen.getAllByText('daily rate')).toHaveLength(5);
    });

    test('a Sunday bar that only repeats Friday\'s oil close shows Friday\'s real move', () => {
        const d = clone();
        const cl = d.commodities.cl;
        cl.history = [...cl.history.slice(0, -1), { date: '2026-10-04', price: 91.11 }];
        cl.current = 91.11;
        cl.dailyChange = { value: 0, pct: 0 };
        render(<ExtraMarketsGrid data={d} loading={false} />);
        expect(chg('CL')).toBe('−1.90% · Fri');
    });

    test('a row whose change cannot be worked out shows nothing, not +0.00%', () => {
        const d = clone();
        d.commodities.gc.history = [{ date: '2026-10-01', price: 0 }, { date: '2026-10-02', price: 4139.49 }];
        d.commodities.gc.dailyChange = undefined;
        const { container } = render(<ExtraMarketsGrid data={d} loading={false} />);
        expect(chg('GOLD')).toBeNull();
        expect(screen.queryByTestId('mkt-spot-GOLD')).toBeNull();
        expect(container.textContent).toContain('GOLD');
        expect(container.textContent).not.toContain('0.00%');
    });

    test('a Gold / BTC spot fallback with no history still says live', () => {
        const d = clone();
        d.commodities.btc = { current: 84742.22, dailyChange: { value: 0, pct: 0 }, history: [] };
        render(<ExtraMarketsGrid data={d} loading={false} />);
        expect(chg('BTC')).toBeNull();
        expect(screen.getByTestId('mkt-spot-BTC').textContent).toBe('live');
    });

    test('missing payloads render nothing harmful', () => {
        const { container: a } = render(<ExtraMarketsGrid data={null} loading={false} />);
        expect(a.textContent).toBe('');
        const { container: b } = render(<ExtraMarketsGrid data={{}} loading={false} />);
        expect(b.textContent).not.toMatch(/NaN|undefined|Infinity|0\.00%/);
        const { container: c } = render(<ExtraMarketsGrid data={{ rates: { tnx: { current: 5.24 } }, fx: { dxy: {} } }} loading={false} />);
        expect(c.textContent).not.toMatch(/NaN|undefined|Infinity|0\.00%/);
    });
});

describe('Global Markets: a row filled from the last-known-good copy says so', () => {
    test('stale row shows an orange 🕐 date tag; live rows show none', () => {
        const d = { ...live, commodities: { ...live.commodities, cl: { ...live.commodities.cl, stale: true, savedAt: '2026-10-08T20:15:00Z' } } };
        render(<ExtraMarketsGrid data={d} loading={false} />);
        expect(screen.getByTestId('mkt-stale-CL').textContent).toBe('🕐 Oct 8');
        expect(screen.queryByTestId('mkt-stale-BTC')).toBeNull();
    });
});
